"""Interventions used for criterion 1 (paper §3.1 and Methods, "Technical details of J-lens use
cases").

Swap.  §3.1: "we subtract the projection onto the Soccer lens vector and add an equal-magnitude
projection onto the Rugby lens vector":
        delta = <v_s, h> (v_t - v_s)            (v unit-normalised)
computed from the clean pass and applied at every band layer and every token position
("clamped lens-coordinate swap at every position", Fig. 13 caption).
Component swap.  Fig. 8: "substituting each component for the J-lens vectors used previously,
with every perturbation rescaled to the same magnitude" — the same magnitude, along the
component difference:
        delta_a = <v_s, h> ||v_t - v_s|| * unit(a_t - a_s)
where a is the J-space part or the non-J-space remainder of the two concept vectors.
Clamp-to-clean.  §3.1: "clamping the relevant J-lens coordinates to their clean-pass values at
every position and layer" — the coordinates along a set of lens vectors are held at their
clean-pass values.
Pursuit.  The paper's "gradient pursuit" is not specified beyond the name; ours is a
non-negative greedy pursuit over the full dictionary with a non-negative least-squares refit
on the support at each of the k steps.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import nnls

from .hooks import Edit
from .model import Lensed


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def swap_edits(lm: Lensed, ids: torch.Tensor, s: int, t: int, layers, *, comp=None, clean=None) -> list[Edit]:
    """Swap source token s -> target token t at every position of every layer in `layers`.
    `comp` = {L: (a_s, a_t)} substitutes component directions for the lens vectors."""
    clean = clean or lm.residuals(ids, layers)
    edits = []
    for L in layers:
        vs, vt = unit(lm.v(L, s)), unit(lm.v(L, t))
        magnitude = (clean[L] @ vs) * (vt - vs).norm()  # [T]
        direction = unit(vt - vs) if comp is None else unit(comp[L][1] - comp[L][0])
        delta = magnitude[:, None] * direction[None, :]  # [T, d]
        edits.append(Edit(L, lambda h, pos, delta=delta: h + delta[pos]))
    return edits


def clamp_edits(lm: Lensed, ids: torch.Tensor, dirs: dict[int, torch.Tensor], *, clean=None) -> list[Edit]:
    """Hold the coordinates along dirs[L] ([m, d]) at their clean-pass values, every position."""
    clean = clean or lm.residuals(ids, sorted(dirs))
    edits = []
    for L, D in dirs.items():
        Vm = unit(D).T  # [d, m]
        P = torch.linalg.pinv(Vm)  # [m, d]
        c_clean = clean[L] @ P.T  # [T, m]
        edits.append(Edit(L, lambda h, pos, P=P, Vm=Vm, c_clean=c_clean: h + (c_clean[pos] - h @ P.T) @ Vm.T))
    return edits


@torch.no_grad()
def pursuit(x: torch.Tensor, V: torch.Tensor, k: int):
    """Non-negative greedy pursuit of x [d] over the dictionary V [N, d] -> (support, recon)."""
    Vn = unit(V)
    support: list[int] = []
    r = x.clone()
    recon = torch.zeros_like(x)
    for _ in range(k):
        corr = Vn @ r
        if support:
            corr[torch.as_tensor(support, device=x.device)] = -torch.inf
        j = int(corr.argmax())
        if corr[j] <= 0:
            break
        support.append(j)
        A = V[support].T.cpu().double().numpy()  # [d, |support|]
        coeffs, _ = nnls(A, x.cpu().double().numpy())
        recon = torch.as_tensor(A @ coeffs, device=x.device, dtype=x.dtype)
        r = x - recon
        if r.norm() < 1e-6 * x.norm():
            break
    return support, recon
