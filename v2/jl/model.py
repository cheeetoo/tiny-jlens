"""GPT-2 small + the Jacobian lens: loading, J-lens vectors, readout.

Conventions (identical to Anthropic's reference implementation in ref/jacobian-lens):

* layer L  := the residual stream at the OUTPUT of transformer block L, L = 0..11.
* The lens file holds J_L for source layers 0..10, fitted with target layer 11 (the
  output of the last block, before the final LayerNorm).
* lens readout at layer L:   lens_L(h) = W_U · ln_f(J_L h)
  (paper, Methods:  lens(h_l) = softmax(W_U norm(J_l h_l)) ).

J-lens vectors.  As in the paper: the rows of W_U J_L, i.e. token t's unembedding row
pushed back through J_L,  v_t = J_L^T w_t.  (GPT-2's final LayerNorm sits between J_L h and
W_U in the readout; it is applied in the readout, not folded into v_t.)

Centering.  GPT-2's unembedding rows share a large common component, so the raw v_t all
point in nearly the same direction (a cone).  We subtract the vocabulary mean,
      v_t  <-  v_t - mean_{t'} v_{t'} .
Every readout (logit differences, softmax, ranks) is exactly unchanged by this, because
<mean_{t'} v_{t'}, h> is the same number for every t.  Only the geometry used by
interventions (projections, pseudoinverse coordinates, decompositions) changes.
"""
from __future__ import annotations

import torch
import transformers

import jlens
from jlens.hooks import ActivationRecorder

from .hooks import Edit, Session

MODEL_ID = "openai-community/gpt2"
LENS_PATH = (
    "/tiny-jlens/lenses/_hf/gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"
)
BAND = [7, 8, 9]  # workspace band for gpt2-small (see v2/README.md)


class Lensed:
    """The model, its lens, and the (centered) J-lens dictionary.  Everything fp32 on GPU."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
        hf = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.float32
        ).to(device).eval()
        # Anthropic's adapter: .layers (blocks), .forward(ids), .unembed(h) = lm_head(ln_f(h))
        self.m = jlens.from_hf(hf, self.tok)
        self.n_layers = self.m.n_layers  # 12
        self.d = self.m.d_model  # 768
        self.final = self.n_layers - 1  # 11

        lens = jlens.JacobianLens.load(LENS_PATH)
        self.layers = list(lens.source_layers)  # [0, ..., 10]
        self.J = {L: lens.jacobians[L].float().to(device) for L in self.layers}
        # J_11 := identity (layer 11 is the lens target, so its "lens" is the model output)
        self.J[self.final] = torch.eye(self.d, device=device)

        # unembedding rows w_t, [vocab, d]
        self.U = self.m._lm_head.weight.detach().float().to(device)
        self.vocab = self.U.shape[0]
        self._V: dict[int, torch.Tensor] = {}

        self.bos = self.tok.bos_token_id  # <|endoftext|>, 50256

    # ------------------------------------------------------------------ tokens
    def encode(self, text: str) -> torch.Tensor:
        """[1, T] input ids, with <|endoftext|> prepended as an attention-sink BOS
        (the reference implementation's default, `force_bos=True`)."""
        ids = self.tok(text, add_special_tokens=False).input_ids
        return torch.tensor([[self.bos] + ids], device=self.device)

    def tid(self, s: str) -> int:
        """Id of a string that must be exactly one token."""
        ids = self.tok(s, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f"{s!r} is {len(ids)} tokens: {[self.tok.decode([i]) for i in ids]}")
        return ids[0]

    def is_single(self, s: str) -> bool:
        return len(self.tok(s, add_special_tokens=False).input_ids) == 1

    def dec(self, t: int) -> str:
        return self.tok.decode([t])

    # ------------------------------------------------------------------ forward
    @torch.no_grad()
    def residuals(
        self, ids: torch.Tensor, layers=None, edits: list[Edit] | None = None
    ) -> dict[int, torch.Tensor]:
        """{L: [T, d]} residuals (block outputs) for one prompt, optionally under edits."""
        record = sorted(set(self.layers if layers is None else layers))
        with Session(self.m.layers, edits or []):
            with ActivationRecorder(self.m.layers, at=record) as rec:
                self.m.forward(ids)
                return {L: rec.activations[L][0].detach().float() for L in record}

    @torch.no_grad()
    def logits(self, ids: torch.Tensor, edits: list[Edit] | None = None) -> torch.Tensor:
        """Model output logits [T, vocab], optionally under edits."""
        h = self.residuals(ids, [self.final], edits)[self.final]
        return self.m.unembed(h).float()

    @torch.no_grad()
    def lens_logits(self, h: torch.Tensor, L: int) -> torch.Tensor:
        """Exact lens readout  W_U ln_f(J_L h)  for residuals h [..., d] -> [..., vocab]."""
        return self.m.unembed(h @ self.J[L].T).float()

    # ------------------------------------------------------------------ vectors
    def V(self, L: int) -> torch.Tensor:
        """Centered J-lens dictionary at layer L: [vocab, d], row t = v_t - mean_t' v_t'."""
        if L not in self._V:
            V = self.U @ self.J[L]  # row t = J_L^T w_t
            self._V[L] = V - V.mean(dim=0, keepdim=True)
        return self._V[L]

    def v(self, L: int, t: int) -> torch.Tensor:
        """Centered J-lens vector of token t at layer L: [d]."""
        return self.V(L)[t]


# ---------------------------------------------------------------------- ranks
def ranks_of(logits: torch.Tensor, token_ids) -> torch.Tensor:
    """1-indexed rank of each token id in a [vocab] logit vector (1 = top)."""
    t = torch.as_tensor(list(token_ids), device=logits.device)
    return (logits[None, :] > logits[t][:, None]).sum(dim=1) + 1
