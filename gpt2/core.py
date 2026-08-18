"""Shared machinery for all experiments: model+lens loading, exact lens
readouts, J-lens vectors, and residual-stream interventions.

Conventions (identical to Anthropic's reference implementation `jlens`,
validated in experiments/00_validate.py):

- `layer` L = the residual stream at the OUTPUT of transformer block L.
  gpt2-small has blocks 0..11; the lens is fitted for source layers 0..10
  with target = final layer (the Neuronpedia recipe).
- Lens readout at layer L: unembed(J_L @ h), where unembed is the model's
  own final LayerNorm + LM head. Exactly:
      logit_tau(h) = <v_tau, h> / sigma(J_L h)  +  beta_tau
  with v_tau = J_L^T C (g * w_tau)   (w_tau = unembedding row,
       g = ln_f gain, C = mean-centering over dims — LN's mu subtraction),
  sigma = ln_f's per-position std, and beta_tau = <w_tau, ln_f.bias>.
  v_tau is "the J-lens vector of token tau at layer L" (paper: rows of
  W_U J_L). Rankings are always computed from the exact logits.

- The dictionary {v_tau} is used in two flavors:
    raw:      v_tau
    centered: v_tau - vbar   (vbar = vocabulary-mean lens vector)
  Readout rankings are *exactly* invariant to centering (the <vbar, h>
  term is constant across tau, and softmax kills constants). Geometry —
  decompositions, projections, pseudoinverse coordinates — is not.
  See experiments/10_cone.py for why centered is the meaningful geometry
  at GPT-2 scale.

- Interventions are forward hooks on blocks, applied at absolute token
  positions (KV-cache safe, batch 1). Swaps use the paper's CLAMPED
  semantics: coordinates are held at (swapped) clean-pass values at every
  edited layer, not re-swapped per layer (re-swapping oscillates — the
  swap permutation is an involution).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

import jlens
from jlens.hooks import ActivationRecorder

LENS_DIR = "/tiny-jlens/lenses"

LENS_FILES = {
    "gpt2": f"{LENS_DIR}/gpt2-small/gpt2_jacobian_lens.pt",  # authors' release
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

class Kit:
    """Model + lens + precomputed pieces. Everything float32 on GPU."""

    def __init__(self, model_name: str, lens_path: str | None = None,
                 need_lens: bool = True):
        import transformers

        self.name = model_name
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        hf = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32
        ).cuda()
        self.model = jlens.from_hf(hf, self.tokenizer)
        self.n_layers = self.model.n_layers
        self.d_model = self.model.d_model
        if need_lens:
            self.lens = jlens.JacobianLens.load(lens_path or LENS_FILES[model_name])
            for l in self.lens.source_layers:
                self.lens.jacobians[l] = self.lens.jacobians[l].float().cuda()
            self.layers = self.lens.source_layers  # fitted layers (0 .. n_layers-2)
        else:
            self.lens, self.layers = None, []

        # Effective unembedding rows u_tau = C (g * w_tau), and bias beta_tau.
        norm: nn.Module = self.model._final_norm
        U = self.model._lm_head.weight.detach().float().cuda()  # [vocab, d]
        g = norm.weight.detach().float().cuda()
        U = U * g[None, :]
        U = U - U.mean(dim=1, keepdim=True)  # LN centering, folded
        self.U_eff = U
        b = norm.bias.detach().float().cuda() if getattr(norm, "bias", None) is not None else None
        self.beta = (self.model._lm_head.weight.detach().float().cuda() @ b) if b is not None else None

        self._V_cache: dict[int, torch.Tensor] = {}    # full dictionaries
        self._vbar_cache: dict[int, torch.Tensor] = {}
        self._mean_norm_cache: dict[int, float] = {}

    # ---------- tokens ----------

    def encode(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt").input_ids.cuda()

    def decode(self, ids) -> str:
        return self.tokenizer.decode(ids)

    def tok_id(self, word: str) -> int:
        """Token id of a string that must be a single token (asserts)."""
        ids = self.tokenizer(word, add_special_tokens=False).input_ids
        assert len(ids) == 1, f"{word!r} is {len(ids)} tokens"
        return ids[0]

    def first_content_id(self, text: str) -> int:
        """First non-whitespace token id of a string (grading helper: GPT-2
        tokenizes ' 5' as [' ', '5'] in some contexts)."""
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        for i in ids:
            if self.tokenizer.decode([i]).strip():
                return i
        return ids[0]

    # ---------- forward / readout ----------

    @torch.no_grad()
    def residuals(self, input_ids: torch.Tensor, layers=None) -> dict[int, torch.Tensor]:
        """{L: [seq, d] float32} for one prompt [1, seq]."""
        record = sorted(set(layers if layers is not None else self.layers))
        with ActivationRecorder(self.model.layers, at=record) as rec:
            self.model.forward(input_ids)
            return {l: rec.activations[l][0].detach().float() for l in record}

    @torch.no_grad()
    def lens_logits(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """Exact lens readout logits [..., vocab] for residuals h [..., d]."""
        return self.model.unembed(h @ self.lens.jacobians[layer].T).float()

    @torch.no_grad()
    def ranks(self, h: torch.Tensor, layer: int, token_ids: list[int]) -> torch.Tensor:
        """Rank (0 = top) of each token in the lens readout. h [seq, d] ->
        [seq, len(token_ids)]."""
        logits = self.lens_logits(h, layer)
        target = logits[:, token_ids]
        return (logits[:, None, :] > target[:, :, None]).sum(dim=2)

    @torch.no_grad()
    def topk_tokens(self, h: torch.Tensor, layer: int, k: int = 10) -> list[list[str]]:
        """Top-k readout tokens per position (h [seq, d])."""
        logits = self.lens_logits(h, layer)
        top = logits.topk(k, dim=-1).indices
        return [[self.tokenizer.decode([t]) for t in row] for row in top.tolist()]

    @torch.no_grad()
    def model_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Final logits [seq, vocab] (no interventions)."""
        final = self.n_layers - 1
        with ActivationRecorder(self.model.layers, at=[final]) as rec:
            self.model.forward(input_ids)
            h = rec.activations[final][0].detach().float()
        return self.model.unembed(h).float()

    # ---------- J-lens vectors ----------

    def V(self, layer: int) -> torch.Tensor:
        """Full raw dictionary [vocab, d] at `layer` (cached)."""
        if layer not in self._V_cache:
            self._V_cache[layer] = self.U_eff @ self.lens.jacobians[layer]
        return self._V_cache[layer]

    def vbar(self, layer: int) -> torch.Tensor:
        if layer not in self._vbar_cache:
            self._vbar_cache[layer] = self.V(layer).mean(dim=0)
        return self._vbar_cache[layer]

    def vectors(self, layer: int, token_ids: list[int], centered: bool) -> torch.Tensor:
        """[len(ids), d] lens vectors, raw or centered."""
        out = self.U_eff[token_ids] @ self.lens.jacobians[layer]
        if centered:
            out = out - self.vbar(layer)[None, :]
        return out

    def drop_V_cache(self) -> None:
        self._V_cache.clear()
        torch.cuda.empty_cache()

    # ---------- misc ----------

    @torch.no_grad()
    def mean_resid_norm(self, layer: int) -> float:
        """Mean residual norm at `layer` over a fixed wikitext sample
        (steering scale reference)."""
        if layer not in self._mean_norm_cache:
            import datasets

            ds = datasets.load_dataset(
                "Salesforce/wikitext", "wikitext-103-raw-v1", split="validation"
            )
            texts = [r["text"].strip() for r in ds if len(r["text"].strip()) > 400][:8]
            norms = []
            for t in texts:
                ids = self.encode(t[:1500])[:, :128]
                h = self.residuals(ids, [layer])[layer]
                norms.append(h[16:].norm(dim=1).mean().item())
            self._mean_norm_cache[layer] = float(sum(norms) / len(norms))
        return self._mean_norm_cache[layer]


# --------------------------------------------------------------------------
# Interventions (forward hooks on blocks; absolute positions; batch 1)
# --------------------------------------------------------------------------

def _get_hidden(output):
    return output if torch.is_tensor(output) else output[0]


def _set_hidden(output, hidden):
    return hidden if torch.is_tensor(output) else (hidden,) + tuple(output[1:])


@dataclass
class Edit:
    """fn(h_slice [n, d], abs_positions [n]) -> new h_slice, applied to the
    output of block `layer` at `positions` (None = every position, including
    generated ones)."""

    layer: int
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    positions: list[int] | None = None


class Session:
    """Installs Edits as hooks for the duration of a `with` block."""

    def __init__(self, kit: Kit, edits: list[Edit]):
        self.kit = kit
        self.by_layer: dict[int, list[Edit]] = {}
        for e in edits:
            self.by_layer.setdefault(e.layer, []).append(e)
        self._handles = []
        self.offset = 0

    def __enter__(self):
        first = min(self.by_layer) if self.by_layer else None

        def make_hook(layer: int):
            def hook(module, inputs, output):
                hidden = _get_hidden(output)
                assert hidden.shape[0] == 1, "batch must be 1 under interventions"
                seq = hidden.shape[1]
                if layer == first:
                    self._base = self.offset
                    self.offset += seq
                base = getattr(self, "_base", 0)
                abs_pos = torch.arange(base, base + seq, device=hidden.device)
                h = hidden[0]
                for e in self.by_layer[layer]:
                    if e.positions is None:
                        idx = torch.arange(seq, device=hidden.device)
                    else:
                        want = torch.tensor(sorted(set(e.positions)), device=hidden.device)
                        idx = torch.isin(abs_pos, want).nonzero(as_tuple=True)[0]
                    if len(idx):
                        new = e.fn(h[idx].float(), abs_pos[idx]).to(h.dtype)
                        h = h.index_copy(0, idx, new)
                return _set_hidden(output, h.unsqueeze(0))

            return hook

        for layer in self.by_layer:
            self._handles.append(self.kit.model.layers[layer].register_forward_hook(make_hook(layer)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()


# ---------- edit factories ----------

def steer(kit: Kit, layer: int, token_id: int, strength: float,
          positions=None, centered: bool = False) -> Edit:
    """h += strength * mean_resid_norm * unit(v_token)."""
    v = kit.vectors(layer, [token_id], centered)[0]
    delta = (strength * kit.mean_resid_norm(layer)) * v / v.norm().clamp_min(1e-8)

    def fn(h, pos):
        return h + delta.to(h.device)

    return Edit(layer, fn, positions)


def add_delta(delta: torch.Tensor, layer: int, positions=None) -> Edit:
    def fn(h, pos):
        return h + delta.to(h.device)

    return Edit(layer, fn, positions)


def swap_clamped(kit: Kit, input_ids: torch.Tensor, layers: list[int],
                 source_ids: list[int], target_ids: list[int],
                 alpha: float = 1.0, positions=None,
                 centered: bool = False) -> list[Edit]:
    """The paper's clamped lens-coordinate swap (Fig. 4C): at each edited
    layer, coordinates along [v_src..., v_tgt...] are held at the swapped
    clean-pass values. `centered` uses the centered dictionary for both the
    coordinate read (pseudoinverse) and the write-back."""
    assert len(source_ids) == len(target_ids)
    m = len(source_ids)
    resid = kit.residuals(input_ids, layers)
    seq = input_ids.shape[1]
    pos_list = list(range(seq)) if positions is None else sorted(set(positions))
    perm = torch.cat([torch.arange(m, 2 * m), torch.arange(0, m)])
    edits = []
    for l in layers:
        Vmat = kit.vectors(l, list(source_ids) + list(target_ids), centered)  # [2m, d]
        pinv = torch.linalg.pinv(Vmat.T)  # [2m, d]
        c_clean = resid[l][pos_list] @ pinv.T  # [n, 2m]
        ref_vals = c_clean + alpha * (c_clean[:, perm] - c_clean)
        ref = {p: ref_vals[i] for i, p in enumerate(pos_list)}

        def fn(h, pos, Vmat=Vmat, pinv=pinv, ref=ref):
            c = h @ pinv.T
            tgt = torch.stack([ref[int(p)] for p in pos]).to(h.device)
            return h + (tgt - c) @ Vmat

        edits.append(Edit(l, fn, pos_list))
    return edits


def swap_projection(kit: Kit, input_ids: torch.Tensor, layers: list[int],
                    source_ids: list[int], target_ids: list[int],
                    alpha: float = 1.0, positions=None,
                    centered: bool = False) -> list[Edit]:
    """The paper's §3.1 projection swap, clamped to clean-pass values:
    h += alpha * <h_clean, u_s> (u_t - u_s), unit vectors u."""
    assert len(source_ids) == len(target_ids)
    resid = kit.residuals(input_ids, layers)
    seq = input_ids.shape[1]
    pos_list = list(range(seq)) if positions is None else sorted(set(positions))
    edits = []
    for l in layers:
        Vs = kit.vectors(l, list(source_ids), centered)
        Vt = kit.vectors(l, list(target_ids), centered)
        Us = Vs / Vs.norm(dim=1, keepdim=True).clamp_min(1e-8)
        Ut = Vt / Vt.norm(dim=1, keepdim=True).clamp_min(1e-8)
        c_clean = resid[l][pos_list] @ Us.T  # [n, m]
        delta = alpha * (c_clean @ (Ut - Us))  # [n, d]
        ref = {p: delta[i] for i, p in enumerate(pos_list)}

        def fn(h, pos, ref=ref):
            return h + torch.stack([ref[int(p)] for p in pos]).to(h.device)

        edits.append(Edit(l, fn, pos_list))
    return edits


def clamp_coords(kit: Kit, input_ids: torch.Tensor, layers: list[int],
                 token_ids: list[int], positions=None,
                 centered: bool = False) -> list[Edit]:
    """Hold coordinates along the given tokens' lens vectors at clean-pass
    values (blocks re-entry of a concept; the mediation control)."""
    resid = kit.residuals(input_ids, layers)
    seq = input_ids.shape[1]
    pos_list = list(range(seq)) if positions is None else sorted(set(positions))
    edits = []
    for l in layers:
        Vmat = kit.vectors(l, token_ids, centered)
        pinv = torch.linalg.pinv(Vmat.T)
        c_clean = resid[l][pos_list] @ pinv.T
        ref = {p: c_clean[i] for i, p in enumerate(pos_list)}

        def fn(h, pos, Vmat=Vmat, pinv=pinv, ref=ref):
            c = h @ pinv.T
            tgt = torch.stack([ref[int(p)] for p in pos]).to(h.device)
            return h + (tgt - c) @ Vmat

        edits.append(Edit(l, fn, pos_list))
    return edits


def ablate_topk(kit: Kit, layer: int, k: int,
                protected: dict[int, set[int]] | None = None,
                positions=None, centered: bool = False) -> Edit:
    """At each position, project out the span of the k highest-scoring lens
    vectors (readout ranking; identical for raw/centered), sparing protected
    token ids. `centered` projects out centered vectors — removes
    token-discriminative content only, not the shared axis."""
    Jl = kit.lens.jacobians[layer]

    def fn(h, pos):
        out = h.clone()
        scores = kit.U_eff @ (Jl @ h.T)  # [vocab, n]
        for j in range(h.shape[0]):
            s = scores[:, j].clone()
            p = int(pos[j])
            if protected and p in protected:
                s[list(protected[p])] = -torch.inf
            top = s.topk(k).indices.tolist()
            Vmat = kit.vectors(layer, top, centered)
            Q, _ = torch.linalg.qr(Vmat.T)
            out[j] = h[j] - Q @ (Q.T @ h[j])
        return out

    return Edit(layer, fn, positions)


# ---------- running under edits ----------

@torch.no_grad()
def logits_with(kit: Kit, input_ids: torch.Tensor, edits: list[Edit]) -> torch.Tensor:
    """Final logits [seq, vocab] with edits active."""
    final = kit.n_layers - 1
    with Session(kit, edits):
        with ActivationRecorder(kit.model.layers, at=[final]) as rec:
            kit.model.forward(input_ids)
            h = rec.activations[final][0].detach().float()
    return kit.model.unembed(h).float()


@torch.no_grad()
def residuals_with(kit: Kit, input_ids: torch.Tensor, edits: list[Edit],
                   layers) -> dict[int, torch.Tensor]:
    record = sorted(set(layers))
    with Session(kit, edits):
        with ActivationRecorder(kit.model.layers, at=record) as rec:
            kit.model.forward(input_ids)
            return {l: rec.activations[l][0].detach().float() for l in record}


@torch.no_grad()
def generate_with(kit: Kit, input_ids: torch.Tensor, edits: list[Edit],
                  max_new_tokens: int = 16) -> str:
    """Greedy generation with edits active; returns the generated text."""
    with Session(kit, edits):
        out = kit.model._hf_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=kit.tokenizer.eos_token_id,
        )
    return kit.decode(out[0, input_ids.shape[1]:])


# --------------------------------------------------------------------------
# Sparse decomposition (gradient pursuit, paper §Methods)
# --------------------------------------------------------------------------

@torch.no_grad()
def gradient_pursuit(kit: Kit, x: torch.Tensor, layer: int, k: int,
                     centered: bool = False, n_candidates: int = 4096):
    """Greedy sparse non-negative decomposition of x [d] into k lens vectors.
    Candidates = top n_candidates tokens by readout score of x.
    Returns (token_ids [<=k], coeffs, reconstruction [d])."""
    Jl = kit.lens.jacobians[layer]
    scores = kit.U_eff @ (Jl @ x)
    candidate_ids = scores.topk(n_candidates).indices
    Vmat = kit.vectors(layer, candidate_ids.tolist(), centered)  # [C, d]
    Vn = Vmat / Vmat.norm(dim=1, keepdim=True).clamp_min(1e-8)

    selected: list[int] = []
    coeffs = torch.zeros(0, device=x.device)
    r = x.clone()
    for _ in range(k):
        corr = Vn @ r
        if selected:
            corr[torch.tensor(selected, device=x.device)] = -torch.inf
        best = int(corr.argmax())
        if corr[best] <= 0:
            break
        selected.append(best)
        A = Vmat[selected]
        G, b = A @ A.T, A @ x
        c = torch.linalg.lstsq(
            G + 1e-6 * torch.eye(len(selected), device=x.device), b.unsqueeze(1)
        ).solution.squeeze(1).clamp_min(0.0)
        step = 1.0 / (torch.linalg.matrix_norm(G, 2) + 1e-6)
        prev = None
        for _ in range(400):
            c = (c - step * (G @ c - b)).clamp_min(0.0)
            if prev is not None and (c - prev).norm() < 1e-6 * (c.norm() + 1e-9):
                break
            prev = c.clone()
        coeffs = c
        r = x - A.T @ c
    if not selected:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0), torch.zeros_like(x)
    ids = candidate_ids[torch.tensor(selected, device=candidate_ids.device)]
    return ids.cpu(), coeffs.cpu(), Vmat[selected].T @ coeffs
