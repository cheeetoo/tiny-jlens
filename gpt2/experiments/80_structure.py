"""Structural profile of the lens across layers.

Phases:
  band      lens-vs-model agreement per layer on held-out wikitext (where is
            the "motor" regime?), plus lens top-1 autocorrelation across
            positions vs a shuffled null (abstract-content persistence).
  occupancy sparse-decomposition occupancy in the centered gauge (the
            paper's capacity measure that the raw cone breaks): number of
            atoms at which gradient-pursuit's marginal gain drops to that of
            a matched random dictionary.

Run:  python experiments/80_structure.py band|occupancy [model]
"""

import json
import sys

sys.path.insert(0, "/tiny-jlens/gpt2")

import datasets
import torch

import core

PHASE = sys.argv[1] if len(sys.argv) > 1 else "band"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "gpt2"
kit = core.Kit(MODEL)

ds = datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                           split="validation")
texts = [r["text"].strip() for r in ds if len(r["text"].strip()) > 600][:48]

if PHASE == "band":
    stats = {l: dict(top1=0, top10=0, n=0, auto=0, auto_null=0, npairs=0)
             for l in kit.layers}
    torch.manual_seed(0)
    for t in texts:
        ids = kit.encode(t[:2000])[:, :128]
        resid = kit.residuals(ids)
        model_lg = kit.model_logits(ids)
        model_top1 = model_lg.argmax(dim=-1)  # [seq]
        for l in kit.layers:
            lens_lg = kit.lens_logits(resid[l], l)
            lens_sorted = lens_lg.topk(10, dim=-1).indices  # [seq, 10]
            pos = range(16, ids.shape[1] - 1)
            for p in pos:
                stats[l]["n"] += 1
                stats[l]["top1"] += int(lens_sorted[p, 0] == model_top1[p])
                stats[l]["top10"] += int((lens_sorted[p] == model_top1[p]).any())
            # autocorrelation: same lens top-1 at adjacent positions,
            # vs a position-shuffled null
            t1 = lens_sorted[list(pos), 0]
            same = (t1[:-1] == t1[1:]).float().mean().item()
            perm = t1[torch.randperm(len(t1))]
            null = (perm[:-1] == perm[1:]).float().mean().item()
            stats[l]["auto"] += same
            stats[l]["auto_null"] += null
            stats[l]["npairs"] += 1
    print(f"{MODEL}: lens-vs-model agreement + top-1 autocorrelation "
          f"({len(texts)} wikitext seqs)")
    print(f"{'L':>3} {'top1':>6} {'top10':>6} {'autocorr':>9} {'null':>6}")
    out = {}
    for l in kit.layers:
        s = stats[l]
        out[l] = dict(top1=s["top1"] / s["n"], top10=s["top10"] / s["n"],
                      auto=s["auto"] / s["npairs"], auto_null=s["auto_null"] / s["npairs"])
        print(f"{l:>3} {out[l]['top1']:>6.2f} {out[l]['top10']:>6.2f} "
              f"{out[l]['auto']:>9.2f} {out[l]['auto_null']:>6.2f}")
    json.dump(out, open(f"/tiny-jlens/gpt2/results/band_{MODEL}.json", "w"))

if PHASE == "occupancy":
    # occupancy: run gradient pursuit (centered) with k up to 40 on natural
    # residuals; occupancy = last k where the k-th atom's marginal variance
    # gain exceeds the matched random-dictionary gain at the same step.
    torch.manual_seed(0)
    K = 40
    N_POS = 96
    results = {}
    for l in kit.layers:
        gains_lens, gains_rand = [], []
        count = 0
        for t in texts:
            if count >= N_POS:
                break
            ids = kit.encode(t[:2000])[:, :128]
            resid = kit.residuals(ids, [l])[l]
            for p in (32, 96):
                if count >= N_POS:
                    break
                h = resid[p]
                # centered-dictionary pursuit, tracking residual norms stepwise
                def stepwise(vectors_from):
                    r = h.clone()
                    norms = [r.norm().item() ** 2]
                    Vsel = []
                    for _ in range(K):
                        v = vectors_from(r, Vsel)
                        if v is None:
                            break
                        Vsel.append(v)
                        A = torch.stack(Vsel)
                        c = torch.linalg.lstsq(A @ A.T + 1e-6 * torch.eye(len(Vsel), device=h.device),
                                               (A @ h).unsqueeze(1)).solution.squeeze(1).clamp_min(0)
                        r = h - A.T @ c
                        norms.append(r.norm().item() ** 2)
                    return norms

                Jl = kit.lens.jacobians[l]
                vbar = kit.vbar(l)

                def from_lens(r, Vsel):
                    scores = kit.U_eff @ (Jl @ r)
                    cand = scores.topk(64).indices.tolist()
                    W = kit.vectors(l, cand, centered=True)
                    Wn = W / W.norm(dim=1, keepdim=True).clamp_min(1e-8)
                    corr = Wn @ r
                    best = int(corr.argmax())
                    if corr[best] <= 0:
                        return None
                    return W[best]

                rand_dict = torch.randn(4096, kit.d_model, device=h.device)

                def from_rand(r, Vsel):
                    Wn = rand_dict / rand_dict.norm(dim=1, keepdim=True)
                    corr = Wn @ r
                    best = int(corr.argmax())
                    if corr[best] <= 0:
                        return None
                    return rand_dict[best]

                nl = stepwise(from_lens)
                nr = stepwise(from_rand)
                gains_lens.append([-(nl[i + 1] - nl[i]) for i in range(len(nl) - 1)])
                gains_rand.append([-(nr[i + 1] - nr[i]) for i in range(len(nr) - 1)])
                count += 1
        # occupancy per position: last step where lens gain > rand gain
        occs = []
        for gl, gr in zip(gains_lens, gains_rand):
            occ = 0
            for i in range(min(len(gl), len(gr))):
                if gl[i] > gr[i]:
                    occ = i + 1
                else:
                    break
            occs.append(occ)
        occs.sort()
        results[l] = dict(median=occs[len(occs) // 2],
                          q25=occs[len(occs) // 4], q75=occs[3 * len(occs) // 4])
        print(f"L{l:2d}  occupancy median {results[l]['median']:>3}  "
              f"IQR [{results[l]['q25']}, {results[l]['q75']}]", flush=True)
    json.dump(results, open(f"/tiny-jlens/gpt2/results/occupancy_{MODEL}.json", "w"))
