"""Interventions specific to criterion 3.  Kept local to this folder (the shared
`jl` library carries the criterion-1 operations; c3 needs the coordinate swap,
which c1 did not).  All reuse `jl.Edit` / `jl.Lensed`.

Coordinate swap (paper Methods "patching in lens coordinates", Fig. 4C, and the
"clamped lens-coordinate swap" of §3.3 / Fig. 13, 15).  For a source token s and
target token t with centered lens vectors v_s, v_t at layer L, form V = [v_s v_t],
read the oblique coordinates c = V^+ h (V^+ = pseudoinverse), and drive them to the
swapped clean-pass values sigma(c_clean):

    h  <-  h + (ref - V^+ h) V,      ref = c_clean + alpha (sigma(c_clean) - c_clean)

with sigma exchanging the two entries.  The component of h orthogonal to
span{v_s, v_t} is untouched.  alpha=1 is a full swap; the coordinates are held
("clamped") at the swapped clean values at every band layer and position, so the
concept cannot be re-derived downstream.

This is the operation §3.3 names; unlike the subtract-and-add form used for
criterion 1's verbal report, it does not assume the target is absent before the
swap.  (On gpt2-small the subtract-and-add form gives ~0% here; see PROTOCOL.md.)

Component swap (§3.3 privileging, Fig. 16).  A probe direction is not a single
lens vector, so its swap is an additive delta along the component difference,
rescaled to the magnitude of the full-probe swap ("every perturbation rescaled to
the same magnitude").  `delta_edits` applies given per-layer deltas at every
position.
"""
from __future__ import annotations

import torch

from jl import Edit, Lensed


def coord_swap_edits(lm: Lensed, ids: torch.Tensor, s: int, t: int, layers, *,
                     alpha: float = 1.0, clean=None) -> list[Edit]:
    """Paper Fig. 4C coordinate swap s<->t, clamped to swapped clean values, at
    every position of every layer in `layers`."""
    clean = clean or lm.residuals(ids, layers)
    edits = []
    for L in layers:
        V = torch.stack([lm.v(L, s), lm.v(L, t)])          # [2, d] centered lens vectors
        P = torch.linalg.pinv(V.T)                          # [2, d]  coordinate readers (V^+)
        c_clean = clean[L] @ P.T                            # [T, 2]  clean-pass coordinates
        ref = c_clean + alpha * (c_clean[:, [1, 0]] - c_clean)  # swap the two, alpha-scaled
        edits.append(Edit(L, lambda h, pos, P=P, V=V, ref=ref: h + (ref[pos] - h @ P.T) @ V))
    return edits


def delta_edits(deltas: dict[int, torch.Tensor]) -> list[Edit]:
    """Additive edit h <- h + deltas[L] at every position, for each L in deltas."""
    return [Edit(L, lambda h, pos, d=d: h + d[None, :]) for L, d in deltas.items()]
