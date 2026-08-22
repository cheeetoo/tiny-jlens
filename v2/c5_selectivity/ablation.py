"""Whole-J-space ablation, and its matched-norm random control (paper §3.5.2, Fig 22/24).

Kept local to this folder (the shared `jl` library carries criterion 1's operations; c3 added
the coordinate swap; c5 adds the top-k ablation, which the others did not need).

Paper §3.5.2: *"at each token position, across a band of layers, we identify the k=10 most
strongly activated J-lens vectors and zero out the residual stream's projection onto each, then
allow the forward pass to continue. To avoid confounds from ablating tokens the model intended
to output, we do not ablate any tokens that appear in the top-10 tokens of a clean forward
pass, so as to specifically target the J-space's effects on internal reasoning rather than
report."*

So, at each (position, band layer):

  * "most strongly activated J-lens vectors" — the top of the lens readout at that (position,
    layer), i.e. the tokens whose centered lens vector v_t scores highest.  We walk down the
    lens readout, skipping any token in the model's clean output top-10 at that position, until
    we have collected k = 10 directions.
  * "zero out the projection onto each" — we remove the residual's component in the span of
    those k directions:  h <- h - V (V^+ h)  with V = [v_1 .. v_k] (centered lens vectors).
    (For a non-orthogonal set, removing the span is the well-defined reading of "zero out the
    projection onto each"; it is the same span-removal the criterion-1 clamp uses.)

Selection (which directions, and each position's removed-norm) is read from a single clean
forward pass, then applied as fixed per-position edits — deterministic, and not chasing its own
tail across band layers.  `select` does that once; `edits_from` turns a selection into either
the J ablation or its matched-norm random control, so both share one selection pass.

Matched-norm random control (paper: "equal-sized, layer-matched sets of randomly chosen
directions").  We remove the residual's component in the span of k random directions, rescaled
per position to the exact norm the J-space ablation removes there.  This isolates *which
subspace* is removed from *how much norm*: any damage beyond the random control is attributable
to the J-space specifically, not to the size of the perturbation.
"""
from __future__ import annotations

import torch

from jl import Edit, Lensed


def _span_projection(A: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Project rows of h [n, d] onto the per-row column space of A [n, d, k]. -> [n, d]."""
    pinvA = torch.linalg.pinv(A)                       # [n, k, d]
    coeff = torch.bmm(pinvA, h.unsqueeze(-1))          # [n, k, 1]
    return torch.bmm(A, coeff).squeeze(-1)             # [n, d]


def select(lm: Lensed, ids: torch.Tensor, layers, *, k: int = 10,
           exclude_output_top: int = 10, clean=None) -> dict:
    """Per-layer (directions [T,d,k], removed-norm [T]) for the top-k J-space ablation, read
    from the clean pass.  Directions = the k highest lens tokens not in the clean output top-10."""
    clean = clean or lm.residuals(ids, layers)
    out_logits = lm.logits(ids)                        # [T, vocab] clean output
    T = out_logits.shape[0]
    out_top = [set(out_logits[p].topk(exclude_output_top).indices.tolist()) for p in range(T)]
    sel = {}
    for L in layers:
        h = clean[L]                                   # [T, d]
        order = lm.lens_logits(h, L).argsort(dim=-1, descending=True)  # [T, vocab]
        V = torch.empty(T, k, lm.d, device=lm.device)
        for p in range(T):                             # k lens directions, skipping output-top
            chosen = []
            for tok in order[p].tolist():
                if tok in out_top[p]:
                    continue
                chosen.append(tok)
                if len(chosen) == k:
                    break
            V[p] = lm.V(L)[chosen]                      # centered lens vectors [k, d]
        A = V.transpose(1, 2)                           # [T, d, k]  (columns = directions)
        sel[L] = (A, _span_projection(A, h).norm(dim=-1))  # (directions, ||removed||)
    return sel


def edits_from(sel: dict, lm: Lensed, *, random: bool = False, seed: int = 0) -> list[Edit]:
    """Turn a `select` result into the J-space ablation (random=False) or its matched-norm
    random control (random=True)."""
    gen = torch.Generator(device=lm.device).manual_seed(seed)
    edits = []
    for L, (A, removed_norm) in sel.items():
        R = torch.randn(A.shape[0], lm.d, A.shape[2], generator=gen, device=lm.device) if random else None

        def fn(hh, pos, A=A, R=R, removed_norm=removed_norm):
            if R is None:                               # J-space ablation
                return hh - _span_projection(A[pos], hh)
            proj = _span_projection(R[pos], hh)         # matched-norm random control
            scale = (removed_norm[pos] / proj.norm(dim=-1).clamp_min(1e-8)).unsqueeze(-1)
            return hh - proj * scale

        edits.append(Edit(L, fn))
    return edits


def ablate_edits(lm: Lensed, ids: torch.Tensor, layers, *, k: int = 10,
                 exclude_output_top: int = 10, random: bool = False, seed: int = 0,
                 clean=None) -> list[Edit]:
    """Convenience: select + edits_from in one call."""
    sel = select(lm, ids, layers, k=k, exclude_output_top=exclude_output_top, clean=clean)
    return edits_from(sel, lm, random=random, seed=seed)
