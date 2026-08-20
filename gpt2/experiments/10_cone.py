"""The cone: diagnose the degenerate geometry of the GPT-2 J-lens dictionary,
and test the justification for centering.

Background. At each layer the dictionary is 50k vectors v_tau in d=768 dims.
Previous work found them massively correlated ("a cone"), which breaks every
geometry-based operation (sparse decomposition, occupancy, top-k ablation,
matched-norm J/non-J splits) while leaving readouts and difference-based
interventions intact.

The gauge argument. The exact readout is
    logit_tau(h) = <v_tau, h>/sigma(Jh) + beta_tau.
Replacing every v_tau by v_tau + u (any fixed u) shifts all logits by the
same per-position constant <u,h>/sigma — the softmax readout is EXACTLY
invariant (checked in 00_validate.py). The invariance group is translations
only: a shared rescale a*v_tau is NOT softmax-invisible, because beta_tau
and sigma do not transform with it. So the dictionary is only defined up to
translation, and any geometric quantity that changes under it (pairwise
cosines, spans, pseudoinverse coordinates, projections) is not a property
of the lens until a gauge is fixed. Centering (u = -vbar) is the canonical
translate: the minimum-total-norm representative of the gauge class. The empirical question this script answers: how much of the
raw dictionary's geometry is gauge artifact (shared component vbar), and is
the shared component token-discriminative at all?

Measurements per layer:
  1. share of dictionary second moment carried by vbar; pairwise |cos| raw
     vs centered; top principal-component shares raw vs centered.
  2. uniformity: the logit profile of the cone axis across the vocabulary
     (std/|mean| of <v_tau, p>) vs random directions vs centered vectors.
     Near-zero => the axis writes the same logit to every token => carries
     no token-discriminative content.
  3. provenance: vbar = J^T ubar exactly (ubar = mean effective unembedding
     row). Is the dominance from the unembedding cone (ubar large) and/or
     J amplifying ubar's direction?
  4. consequences preview on natural activations: norm removed by top-k
     ablation and variance captured by gradient pursuit, raw vs centered.

Run:  python experiments/10_cone.py [model]
"""

import json
import sys

sys.path.insert(0, "/tiny-jlens/gpt2")

import torch

import core

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
kit = core.Kit(MODEL)
torch.manual_seed(0)

# natural activations: final-position residuals of a few short prompts
PROMPTS = [
    "The capital of the country where Polish is the primary language is",
    "When Mary and John went to the store, John gave a drink to",
    "The Eiffel Tower is located in the city of",
    "In 1969, the first humans landed on the",
    "The primary language spoken in Portugal is",
    "The scientist mixed the two chemicals and watched the solution turn",
    "After the long drought, the farmers finally saw clouds and hoped for",
    "The largest planet in the solar system is called",
    "She opened the letter slowly, afraid of what it might",
    "The recipe said to add two cups of flour and one cup of",
    "The team celebrated loudly after scoring the winning",
    "On the first day of school, the teacher asked everyone to say their",
]
resids = {p: kit.residuals(kit.encode(p)) for p in PROMPTS}

SAMPLE = 4096
vocab = kit.U_eff.shape[0]
idx = torch.randperm(vocab)[:SAMPLE].cuda()

results = {}
for L in kit.layers:
    V = kit.V(L)                      # [vocab, d]
    vbar = kit.vbar(L)
    Vc = V - vbar[None, :]

    # -- 1. geometry
    mean_sq = (V.norm(dim=1) ** 2).mean().item()
    share_vbar = (vbar.norm() ** 2).item() / mean_sq

    Vs, Vcs = V[idx], Vc[idx]
    Vsn = Vs / Vs.norm(dim=1, keepdim=True).clamp_min(1e-8)
    Vcsn = Vcs / Vcs.norm(dim=1, keepdim=True).clamp_min(1e-8)
    cos_raw = (Vsn @ Vsn.T).abs()
    cos_cen = (Vcsn @ Vcsn.T).abs()
    off = ~torch.eye(SAMPLE, dtype=torch.bool, device="cuda")
    mean_cos_raw = cos_raw[off].mean().item()
    mean_cos_cen = cos_cen[off].mean().item()

    def pc_shares(M, n=3):
        # top singular-value shares of the (uncentered) second moment
        s = torch.linalg.svdvals(M)
        tot = (s ** 2).sum()
        return [((s[i] ** 2) / tot).item() for i in range(n)]

    pcs_raw = pc_shares(Vs)
    pcs_cen = pc_shares(Vcs)
    U_svd, S_svd, _ = torch.linalg.svd(Vs, full_matrices=False)
    # top right-singular vector of the raw sample = the raw "cone axis"
    axis = (Vs.T @ U_svd[:, 0]) / S_svd[0]
    cos_axis_vbar = torch.nn.functional.cosine_similarity(axis, vbar, dim=0).abs().item()

    # -- 2. uniformity of logit profiles
    def uniformity(p):
        prof = V @ (p / p.norm())
        return (prof.std() / prof.mean().abs().clamp_min(1e-9)).item()

    unif_vbar = uniformity(vbar)
    unif_rand = float(torch.tensor([uniformity(torch.randn(kit.d_model, device="cuda"))
                                    for _ in range(5)]).median())
    unif_cen = float(torch.tensor([uniformity(Vc[i]) for i in idx[:5]]).median())

    # -- 3. provenance
    ubar = kit.U_eff.mean(dim=0)
    u_mean_sq = (kit.U_eff.norm(dim=1) ** 2).mean().item()
    share_ubar = (ubar.norm() ** 2).item() / u_mean_sq
    J = kit.lens.jacobians[L]
    amp_ubar = (J.T @ ubar).norm().item() / ubar.norm().item()
    r = torch.randn(64, kit.d_model, device="cuda")
    amp_rand = ((r @ J).norm(dim=1) / r.norm(dim=1)).mean().item()

    # -- 4. consequences on natural activations (final position of each prompt)
    def norm_removed(h, k, centered):
        scores = kit.U_eff @ (J @ h)
        top = scores.topk(k).indices.tolist()
        W = kit.vectors(L, top, centered)
        Q, _ = torch.linalg.qr(W.T)
        return ((Q @ (Q.T @ h)).norm() / h.norm()).item()

    def gp_var(h, centered):
        _, _, recon = core.gradient_pursuit(kit, h, L, k=16, centered=centered,
                                            n_candidates=2048)
        return (1 - (h - recon).norm() ** 2 / h.norm() ** 2).item()

    hs = [resids[p][L][-1] for p in PROMPTS]
    removed_raw_k1 = sum(norm_removed(h, 1, False) for h in hs) / len(hs)
    removed_cen_k1 = sum(norm_removed(h, 1, True) for h in hs) / len(hs)
    removed_raw_k10 = sum(norm_removed(h, 10, False) for h in hs) / len(hs)
    removed_cen_k10 = sum(norm_removed(h, 10, True) for h in hs) / len(hs)
    gpv_raw = sum(gp_var(h, False) for h in hs) / len(hs)
    gpv_cen = sum(gp_var(h, True) for h in hs) / len(hs)

    results[L] = dict(
        share_vbar=share_vbar, mean_cos_raw=mean_cos_raw, mean_cos_cen=mean_cos_cen,
        pcs_raw=pcs_raw, pcs_cen=pcs_cen, cos_axis_vbar=cos_axis_vbar,
        unif_vbar=unif_vbar, unif_rand=unif_rand, unif_cen=unif_cen,
        share_ubar=share_ubar, amp_ubar=amp_ubar, amp_rand=amp_rand,
        removed_k1=(removed_raw_k1, removed_cen_k1),
        removed_k10=(removed_raw_k10, removed_cen_k10),
        gp_var16=(gpv_raw, gpv_cen),
    )
    r_ = results[L]
    print(f"L{L:2d}  |vbar|^2 share {r_['share_vbar']:.2f}  "
          f"|cos| raw/cen {r_['mean_cos_raw']:.2f}/{r_['mean_cos_cen']:.3f}  "
          f"PC1 raw/cen {r_['pcs_raw'][0]:.2f}/{r_['pcs_cen'][0]:.3f}  "
          f"cos(axis,vbar) {r_['cos_axis_vbar']:.3f}  "
          f"unif axis/rand/cen {r_['unif_vbar']:.3f}/{r_['unif_rand']:.1f}/{r_['unif_cen']:.1f}  "
          f"ampJ vbar/rand {r_['amp_ubar']:.2f}/{r_['amp_rand']:.2f}  "
          f"ablk1 {r_['removed_k1'][0]:.2f}/{r_['removed_k1'][1]:.3f}  "
          f"ablk10 {r_['removed_k10'][0]:.2f}/{r_['removed_k10'][1]:.3f}  "
          f"gp16 {r_['gp_var16'][0]:.3f}/{r_['gp_var16'][1]:.3f}", flush=True)
    kit.drop_V_cache()

with open(f"/tiny-jlens/gpt2/results/cone_{MODEL}.json", "w") as f:
    json.dump({str(k): v for k, v in results.items()}, f, indent=2)
print("saved results/cone_%s.json" % MODEL)
