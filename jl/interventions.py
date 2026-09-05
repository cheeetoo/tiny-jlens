"""Residual-stream interventions: every operation the five criteria perform on the J-space.

All of them are built from `Edit`s (see hooks.py) and act on a band of layers at every token
position, with the magnitudes read from a clean forward pass.

Subtract-and-add swap (§3.1, and §3.4 with alpha).  §3.1: "we subtract the projection onto the
Soccer lens vector and add an equal-magnitude projection onto the Rugby lens vector":

        delta = alpha * <v_s, h> (v_t - v_s)          (v unit-normalised, <v_s,h> from the clean pass)

applied at every band layer and every token position ("clamped lens-coordinate swap at every
position", Fig. 13 caption).  alpha = 1 is criterion 1's operation; §3.4's "'double strength'
swap ('alpha = 2', doubling the strength with which we subtract the source lens vector and add
in the target)" is the same thing at alpha = 2.

Component swap (Fig. 8, and §3.3's Fig. 16).  "substituting each component for the J-lens
vectors used previously, with every perturbation rescaled to the same magnitude" -- the same
magnitude, along the component difference:

        delta_a = alpha * <v_s, h> ||v_t - v_s|| * unit(a_t - a_s)

where a is the J-space part or the non-J-space remainder of the two concept vectors.  A probe
direction is not a single lens vector, so `delta_edits` applies given per-layer deltas directly.

Coordinate swap (Methods "patching in lens coordinates", Fig. 4C; the operation §3.3 names).
For a source token s and target token t with centered lens vectors v_s, v_t at layer L, form
V = [v_s v_t], read the oblique coordinates c = V^+ h, and drive them to the swapped clean-pass
values sigma(c_clean):

        h  <-  h + (ref - V^+ h) V,      ref = c_clean + alpha (sigma(c_clean) - c_clean)

with sigma exchanging the two entries.  The component of h orthogonal to span{v_s, v_t} is
untouched.  The coordinates are held ("clamped") at the swapped clean values at every band layer
and position, so the concept cannot be re-derived downstream.  Unlike the subtract-and-add form
it does not assume the target is absent before the swap.

Clamp-to-clean (§3.1).  "clamping the relevant J-lens coordinates to their clean-pass values at
every position and layer" -- the coordinates along a set of lens vectors are held at their
clean-pass values.

Pursuit.  The paper's "gradient pursuit" is not specified beyond the name; ours is a
non-negative greedy pursuit over the full dictionary with a non-negative least-squares refit on
the support at each of the k steps.

Workspace loading (§3.4, Fig 19 right).  "a concept's workspace loading [is] the cosine
similarity between the residual stream and that concept's lens vector, averaged over the
argument and readout positions in the unmodified forward pass."

Top-k J-space ablation and its matched-norm random control (§3.5.2, Fig 22/24) -- see
`ablation_select` below for the paper's specification and how we read it.
"""
from __future__ import annotations

import torch
from scipy.optimize import nnls

from .hooks import Edit
from .model import Lensed


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


# ------------------------------------------------------------------------------- swaps
def swap_edits(lm: Lensed, ids: torch.Tensor, s: int, t: int, layers, *,
               alpha: float = 1.0, comp=None, clean=None) -> list[Edit]:
    """Subtract-and-add swap of source token s -> target token t at every position of every
    layer in `layers`:  h <- h + alpha <v_s,h> (v_t - v_s), with <v_s,h> read from the clean
    pass.  `comp` = {L: (a_s, a_t)} substitutes component directions for the lens vectors,
    rescaled to the magnitude of the full swap."""
    clean = clean or lm.residuals(ids, layers)
    edits = []
    for L in layers:
        vs, vt = unit(lm.v(L, s)), unit(lm.v(L, t))
        mag = clean[L] @ vs                                        # [T]
        if comp is None:
            delta = alpha * mag[:, None] * (vt - vs)[None, :]      # [T, d]
        else:
            direction = unit(comp[L][1] - comp[L][0])
            delta = alpha * (mag * (vt - vs).norm())[:, None] * direction[None, :]
        edits.append(Edit(L, lambda h, pos, delta=delta: h + delta[pos]))
    return edits


def coord_swap_edits(lm: Lensed, ids: torch.Tensor, s: int, t: int, layers, *,
                     alpha: float = 1.0, clean=None) -> list[Edit]:
    """Fig. 4C coordinate swap s<->t, clamped to the swapped clean values, at every position of
    every layer in `layers`."""
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


def clamp_edits(lm: Lensed, ids: torch.Tensor, dirs: dict[int, torch.Tensor], *,
                clean=None) -> list[Edit]:
    """Hold the coordinates along dirs[L] ([m, d]) at their clean-pass values, every position."""
    clean = clean or lm.residuals(ids, sorted(dirs))
    edits = []
    for L, D in dirs.items():
        Vm = unit(D).T  # [d, m]
        P = torch.linalg.pinv(Vm)  # [m, d]
        c_clean = clean[L] @ P.T  # [T, m]
        edits.append(Edit(L, lambda h, pos, P=P, Vm=Vm, c_clean=c_clean: h + (c_clean[pos] - h @ P.T) @ Vm.T))
    return edits


# ------------------------------------------------------------------------------- pursuit
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


# ------------------------------------------------------------------------------- loading
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
        v = unit(lm.v(L, arg_token))
        for p in positions:
            cos.append(float(torch.dot(unit(clean[L][p]), v)))
    return sum(cos) / len(cos)


# ------------------------------------------------------------------------------- ablation
def _span_projection(A: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Project rows of h [n, d] onto the per-row column space of A [n, d, k]. -> [n, d]."""
    pinvA = torch.linalg.pinv(A)                       # [n, k, d]
    coeff = torch.bmm(pinvA, h.unsqueeze(-1))          # [n, k, 1]
    return torch.bmm(A, coeff).squeeze(-1)             # [n, d]


def ablation_select(lm: Lensed, ids: torch.Tensor, layers, *, k: int = 10,
                    exclude_output_top: int = 10, clean=None) -> dict:
    """Per-layer (directions [T,d,k], removed-norm [T]) for the top-k J-space ablation, read
    from the clean pass.  Directions = the k highest lens tokens not in the clean output top-10.

    Paper §3.5.2: *"at each token position, across a band of layers, we identify the k=10 most
    strongly activated J-lens vectors and zero out the residual stream's projection onto each,
    then allow the forward pass to continue.  To avoid confounds from ablating tokens the model
    intended to output, we do not ablate any tokens that appear in the top-10 tokens of a clean
    forward pass, so as to specifically target the J-space's effects on internal reasoning
    rather than report."*  So, at each (position, band layer):

      * "most strongly activated J-lens vectors" -- the top of the lens readout at that
        (position, layer), i.e. the tokens whose centered lens vector v_t scores highest.  We
        walk down the lens readout, skipping any token in the model's clean output top-10 at
        that position, until we have collected k = 10 directions.
      * "zero out the projection onto each" -- we remove the residual's component in the span of
        those k directions:  h <- h - V (V^+ h)  with V = [v_1 .. v_k] (centered lens vectors).
        (For a non-orthogonal set, removing the span is the well-defined reading of "zero out
        the projection onto each"; it is the same span-removal the criterion-1 clamp uses.)

    Selection (which directions, and each position's removed-norm) is read from a single clean
    forward pass, then applied as fixed per-position edits -- deterministic, and not chasing its
    own tail across band layers.  `ablation_edits` turns a selection into either the J ablation
    or its matched-norm random control, so both share one selection pass."""
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


def ablation_edits(sel: dict, lm: Lensed, *, random: bool = False, seed: int = 0) -> list[Edit]:
    """Turn an `ablation_select` result into the J-space ablation (random=False) or its
    matched-norm random control (random=True).

    Matched-norm random control (paper: "equal-sized, layer-matched sets of randomly chosen
    directions").  We remove the residual's component in the span of k random directions,
    rescaled per position to the exact norm the J-space ablation removes there.  This isolates
    *which subspace* is removed from *how much norm*: any damage beyond the random control is
    attributable to the J-space specifically, not to the size of the perturbation."""
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
