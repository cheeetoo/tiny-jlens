"""Interventions and measurements specific to criterion 4 (flexible generalization).
Kept local to this folder, mirroring c3's `swaps.py`.  All reuse `jl.Edit`/`jl.Lensed`.

The swap (paper §3.4).  §3.4 specifies its swap by the alpha language: "'double
strength' swap ('alpha = 2', doubling the strength with which we subtract the source
lens vector and add in the target)".  So the operation is the subtract-and-add form
--- the same one criterion 1 uses for verbal report (`jl.swap_edits`) --- with a
scalar alpha:

    delta_L = alpha * <v_s, h> * (v_t - v_s)          (v_s, v_t unit centered lens vectors)

<v_s, h> is read from the clean pass; delta is applied at every band layer and every
token position ("swap ... at every token position across a band of intermediate
layers, applying the identical swap regardless of which prompt we are in").  alpha=1
is the default swap, alpha=2 the paper's double-strength swap.  This is exactly the
criterion-1 operation generalized by alpha, and is the operation §3.4 describes.

Coordinate swap (comparison only).  We also run the Fig 4C coordinate swap that
criterion 3 used (`coord_swap_edits`), so the two forms can be compared on the same
trials.  §3.4 names the subtract-and-add form, so that is the headline; the
coordinate swap is reported alongside for transparency.

Workspace loading (paper §3.4, Fig 19 right).  "a concept's workspace loading [is]
the cosine similarity between the residual stream and that concept's lens vector,
averaged over the argument and readout positions in the unmodified forward pass."
`loading()` computes exactly this over the band, averaged over the argument-token
position and the readout (final) position.
"""
from __future__ import annotations

import torch

from jl import Edit, Lensed


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def subadd_swap_edits(lm: Lensed, ids: torch.Tensor, s: int, t: int, layers, *,
                      alpha: float = 1.0, clean=None) -> list[Edit]:
    """Paper §3.4 subtract-and-add swap s->t, at every position of every layer in
    `layers`:  h <- h + alpha * <v_s,h> * (v_t - v_s), with v_s, v_t unit centered
    lens vectors and <v_s,h> from the clean pass."""
    clean = clean or lm.residuals(ids, layers)
    edits = []
    for L in layers:
        vs, vt = _unit(lm.v(L, s)), _unit(lm.v(L, t))
        mag = clean[L] @ vs                              # [T]  clean projection onto source
        delta = alpha * mag[:, None] * (vt - vs)[None, :]  # [T, d]
        edits.append(Edit(L, lambda h, pos, d=delta: h + d[pos]))
    return edits


def coord_swap_edits(lm: Lensed, ids: torch.Tensor, s: int, t: int, layers, *,
                     alpha: float = 1.0, clean=None) -> list[Edit]:
    """Fig 4C coordinate swap s<->t (the criterion-3 operation), for comparison.
    Reads the two oblique coordinates c = V^+ h and drives them to the swapped
    clean values, leaving the component orthogonal to span{v_s, v_t} untouched."""
    clean = clean or lm.residuals(ids, layers)
    edits = []
    for L in layers:
        V = torch.stack([lm.v(L, s), lm.v(L, t)])        # [2, d]
        P = torch.linalg.pinv(V.T)                        # [2, d]
        c_clean = clean[L] @ P.T                          # [T, 2]
        ref = c_clean + alpha * (c_clean[:, [1, 0]] - c_clean)
        edits.append(Edit(L, lambda h, pos, P=P, V=V, ref=ref: h + (ref[pos] - h @ P.T) @ V))
    return edits


@torch.no_grad()
def loading(lm: Lensed, ids: torch.Tensor, arg_token: int, arg_pos: int, layers,
            *, clean=None) -> float:
    """Workspace loading of `arg_token`: mean over `layers` and over
    {arg_pos, readout(final) pos} of cos(residual, centered lens vector v_arg)."""
    clean = clean or lm.residuals(ids, layers)
    T = clean[layers[0]].shape[0]
    positions = [arg_pos, T - 1]
    cos = []
    for L in layers:
        v = _unit(lm.v(L, arg_token))
        for p in positions:
            cos.append(float(torch.dot(_unit(clean[L][p]), v)))
    return sum(cos) / len(cos)
