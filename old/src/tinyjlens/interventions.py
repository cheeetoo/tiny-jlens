"""Residual-stream interventions in J-lens coordinates.

Implements the paper's intervention repertoire:

- steer:      h <- h + strength * unit(v_t) * mean_resid_norm(layer)
              (verbal-introspection protocol, data README)
- swap:       V = [v_s..., v_t...], c = V^+ h, h <- h + alpha * V (sigma(c) - c)
              with sigma exchanging the source and target coefficient blocks
              (paper Fig. 4C "patching in lens coordinates")
- clamp:      coordinates along V are set to reference values from a clean pass
- project_out: h <- h - P_span{v...} h  (concept ablation)
- topk_ablate: at each position, project out the span of the k most active
              lens vectors, sparing a protected token set (paper §3.5)

Interventions are installed as forward hooks on the residual blocks of a
layer band and apply at absolute sequence positions, correctly tracking
position offsets during incremental decoding (batch size 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch


def _get_hidden(output):
    return output if torch.is_tensor(output) else output[0]


def _set_hidden(output, hidden):
    if torch.is_tensor(output):
        return hidden
    return (hidden,) + tuple(output[1:])


@dataclass
class LayerEdit:
    """One edit applied to the residual output of one block.

    fn(hidden_slice [n_pos, d], abs_positions [n_pos]) -> new hidden_slice.
    positions: absolute token indices, or None for "every position, including
    generated ones".
    """

    layer: int
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    positions: list[int] | None = None


class InterventionSession:
    """Context manager installing a set of LayerEdits on a jlens HF model."""

    def __init__(self, model, edits: list[LayerEdit]):
        self.model = model
        self.edits_by_layer: dict[int, list[LayerEdit]] = {}
        for e in edits:
            self.edits_by_layer.setdefault(e.layer, []).append(e)
        self._handles = []
        self.offset = 0  # tokens already processed (for incremental decoding)
        self._seen_this_pass: dict[int, int] = {}

    def __enter__(self):
        first_layer = min(self.edits_by_layer) if self.edits_by_layer else None

        def make_hook(layer: int):
            def hook(module, inputs, output):
                hidden = _get_hidden(output)
                assert hidden.shape[0] == 1, "batch size must be 1 under interventions"
                seq_len = hidden.shape[1]
                # advance offset bookkeeping once per forward pass, keyed on the
                # first hooked layer
                if layer == first_layer:
                    self._current_base = self.offset
                    self.offset += seq_len
                base = getattr(self, "_current_base", 0)
                abs_pos = torch.arange(base, base + seq_len, device=hidden.device)
                h = hidden[0]
                for e in self.edits_by_layer[layer]:
                    if e.positions is None:
                        mask = torch.ones(seq_len, dtype=torch.bool, device=hidden.device)
                    else:
                        want = torch.tensor(sorted(set(e.positions)), device=hidden.device)
                        mask = torch.isin(abs_pos, want)
                    if mask.any():
                        idx = mask.nonzero(as_tuple=True)[0]
                        new = e.fn(h[idx].float(), abs_pos[idx]).to(h.dtype)
                        h = h.index_copy(0, idx, new)
                hidden = h.unsqueeze(0)
                return _set_hidden(output, hidden)

            return hook

        for layer in self.edits_by_layer:
            self._handles.append(
                self.model.layers[layer].register_forward_hook(make_hook(layer))
            )
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []


# ---------- edit factories ----------


def steer_edit(kit, layer: int, token_id: int, strength: float,
               mean_norm: float, positions: list[int] | None) -> LayerEdit:
    v = kit.jlens_vector(layer, token_id)
    v = v / v.norm().clamp_min(1e-8)
    delta = (strength * mean_norm) * v

    def fn(h, pos):
        return h + delta.to(h.device)

    return LayerEdit(layer=layer, fn=fn, positions=positions)


def steer_vector_edit(vec: torch.Tensor, strength: float, mean_norm: float,
                      layer: int, positions: list[int] | None) -> LayerEdit:
    v = vec / vec.norm().clamp_min(1e-8)
    delta = (strength * mean_norm) * v

    def fn(h, pos):
        return h + delta.to(h.device)

    return LayerEdit(layer=layer, fn=fn, positions=positions)


def swap_edit(kit, layer: int, source_ids: list[int], target_ids: list[int],
              alpha: float, positions: list[int] | None) -> LayerEdit:
    """DYNAMIC pseudoinverse coordinate swap (re-swaps current coordinates).
    NOTE: applying this at several consecutive layers oscillates (sigma is an
    involution); use `clamped_swap_edits` for band interventions — it matches
    the paper's clamped semantics. Kept for single-layer/generated-position use.
    """
    assert len(source_ids) == len(target_ids)
    m = len(source_ids)
    V = kit.jlens_vectors(layer, list(source_ids) + list(target_ids))  # [2m, d]
    pinv = torch.linalg.pinv(V.T)  # [2m, d]
    perm = torch.arange(2 * m)
    perm = torch.cat([perm[m:], perm[:m]])  # swap blocks

    def fn(h, pos):
        c = h @ pinv.T  # [n, 2m]
        delta = (c[:, perm] - c) @ V  # [n, d]
        return h + alpha * delta

    return LayerEdit(layer=layer, fn=fn, positions=positions)


def clamped_swap_edits(kit, input_ids: torch.Tensor, layers: list[int],
                       source_ids: list[int], target_ids: list[int],
                       alpha: float = 1.0,
                       positions: list[int] | None = None) -> list[LayerEdit]:
    """The paper's clamped lens-coordinate swap (Fig. 4C + data README):
    at every band layer, the coordinates along [v_src..., v_tgt...] are HELD at
    the swapped clean-pass values sigma(c_clean) (scaled by alpha beyond the
    clean values), rather than re-swapped dynamically. Runs one clean forward
    to record c_clean per (layer, position).
    """
    assert len(source_ids) == len(target_ids)
    m = len(source_ids)
    resid = kit.residuals(input_ids, list(layers))
    seq = input_ids.shape[1]
    pos_list = list(range(seq)) if positions is None else sorted(set(positions))
    perm = torch.arange(2 * m)
    perm = torch.cat([perm[m:], perm[:m]])
    edits = []
    for l in layers:
        V = kit.jlens_vectors(l, list(source_ids) + list(target_ids))  # [2m, d]
        pinv = torch.linalg.pinv(V.T)  # [2m, d]
        c_clean = resid[l][pos_list] @ pinv.T  # [n, 2m]
        ref_vals = c_clean + alpha * (c_clean[:, perm] - c_clean)
        ref = {p: ref_vals[i] for i, p in enumerate(pos_list)}

        def fn(h, pos, V=V, pinv=pinv, ref=ref):
            c = h @ pinv.T
            tgt = torch.stack([ref[int(p)] for p in pos]).to(h.device)
            return h + (tgt - c) @ V

        edits.append(LayerEdit(layer=l, fn=fn, positions=pos_list))
    return edits


def project_out_edit(vectors: torch.Tensor, layer: int,
                     positions: list[int] | None) -> LayerEdit:
    """Remove the orthogonal projection onto span(vectors [k, d])."""
    Q, _ = torch.linalg.qr(vectors.T)  # [d, k]

    def fn(h, pos):
        return h - (h @ Q) @ Q.T

    return LayerEdit(layer=layer, fn=fn, positions=positions)


def add_delta_edit(delta: torch.Tensor, layer: int,
                   positions: list[int] | None) -> LayerEdit:
    def fn(h, pos):
        return h + delta.to(h.device)

    return LayerEdit(layer=layer, fn=fn, positions=positions)


def clamp_coords_edit(kit, layer: int, token_ids: list[int],
                      ref_coords: dict[int, torch.Tensor],
                      positions: list[int]) -> LayerEdit:
    """Clamp coordinates along the given tokens' lens vectors to reference
    values recorded from a clean pass. ref_coords: {abs_pos: [k] coeffs}."""
    V = kit.jlens_vectors(layer, token_ids)  # [k, d]
    pinv = torch.linalg.pinv(V.T)  # [k, d]

    def fn(h, pos):
        c = h @ pinv.T  # [n, k]
        ref = torch.stack([ref_coords[int(p)] for p in pos]).to(h.device)  # [n, k]
        return h + (ref - c) @ V

    return LayerEdit(layer=layer, fn=fn, positions=positions)


def topk_ablate_edit(kit, layer: int, k: int,
                     protected: dict[int, set[int]] | None,
                     positions: list[int] | None) -> LayerEdit:
    """Project out, at each position independently, the span of the k J-lens
    vectors with the highest lens-readout score there, skipping protected
    token ids for that position (paper §3.5: spare the clean top-10 output)."""
    Jl = kit.lens.jacobians[layer]

    def fn(h, pos):
        out = h.clone()
        scores = kit.U_eff @ (Jl @ h.T)  # [vocab, n]
        for j in range(h.shape[0]):
            p = int(pos[j])
            s = scores[:, j].clone()
            if protected and p in protected:
                s[list(protected[p])] = -torch.inf
            top = s.topk(k).indices
            V = (kit.U_eff[top] @ Jl)  # [k, d]
            Q, _ = torch.linalg.qr(V.T)
            out[j] = h[j] - Q @ (Q.T @ h[j])
        return out

    return LayerEdit(layer=layer, fn=fn, positions=positions)


def proj_swap_edits(kit, input_ids: torch.Tensor, layers: list[int],
                    source_ids: list[int], target_ids: list[int],
                    alpha: float = 1.0,
                    positions: list[int] | None = None) -> list[LayerEdit]:
    """The paper's §3.1 projection swap, clamped to clean-pass values:
    h <- h - alpha*<h_clean, u_s> u_s + alpha*<h_clean, u_s> u_t
    with u_* unit lens vectors (pairwise over source/target lists)."""
    assert len(source_ids) == len(target_ids)
    resid = kit.residuals(input_ids, list(layers))
    seq = input_ids.shape[1]
    pos_list = list(range(seq)) if positions is None else sorted(set(positions))
    edits = []
    for l in layers:
        Vs = kit.jlens_vectors(l, list(source_ids))
        Vt = kit.jlens_vectors(l, list(target_ids))
        Us = Vs / Vs.norm(dim=1, keepdim=True).clamp_min(1e-8)  # [m, d]
        Ut = Vt / Vt.norm(dim=1, keepdim=True).clamp_min(1e-8)
        c_clean = resid[l][pos_list] @ Us.T  # [n, m]
        delta = alpha * (c_clean @ (Ut - Us))  # [n, d]
        ref = {p: delta[i] for i, p in enumerate(pos_list)}

        def fn(h, pos, ref=ref):
            d = torch.stack([ref[int(p)] for p in pos]).to(h.device)
            return h + d

        edits.append(LayerEdit(layer=l, fn=fn, positions=pos_list))
    return edits


# ---------- generation under interventions ----------


@torch.no_grad()
def generate_with_edits(kit, input_ids: torch.Tensor, edits: list[LayerEdit],
                        max_new_tokens: int = 24, temperature: float = 0.0):
    """Greedy (or sampled) generation with edits active; returns generated ids."""
    hf = kit.model._hf_model
    tok_ids = input_ids
    with InterventionSession(kit.model, edits):
        out = hf.generate(
            tok_ids,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=kit.model.tokenizer.eos_token_id,
        )
    return out[:, tok_ids.shape[1]:]


@torch.no_grad()
def logits_with_edits(kit, input_ids: torch.Tensor, edits: list[LayerEdit]) -> torch.Tensor:
    """Final-layer logits [seq, vocab] under edits (single forward)."""
    from jlens.hooks import ActivationRecorder

    final = kit.model.n_layers - 1
    with InterventionSession(kit.model, edits):
        with ActivationRecorder(kit.model.layers, at=[final]) as rec:
            kit.model.forward(input_ids)
            resid = rec.activations[final][0].detach().float()
    return kit.model.unembed(resid).float()
