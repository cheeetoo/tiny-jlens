"""J-space occupancy (paper §4.2): the K at which the marginal reconstruction
improvement of a sparse nonnegative K-atom lens decomposition falls below that
of a same-size random-direction control. Also reports variance explained.

Usage: python scripts/occupancy.py --lens runs/smollm2-135m-it/lens_explore.pt
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import transformers

import jlens

from tinyjlens.lensops import LensKit
from tinyjlens.corpus import wikitext_prompts
from tinyjlens.prompts import build_raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--lens", default="runs/smollm2-135m-it/lens_explore.pt")
    ap.add_argument("--layers", default="20,23,26")
    ap.add_argument("--kmax", type=int, default=60)
    ap.add_argument("--n-positions", type=int, default=40)
    ap.add_argument("--out", default="runs/occupancy.json")
    args = ap.parse_args()

    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    hf = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda()
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(args.lens)
    kit = LensKit.build(model, lens)
    layers = [int(x) for x in args.layers.split(",")]

    prompts = wikitext_prompts(6, skip=9000)
    acts = {l: [] for l in layers}
    for p in prompts:
        bp = build_raw(tok, p[:400])
        resid = kit.residuals(bp.input_ids, layers)
        for l in layers:
            n = resid[l].shape[0]
            idx = torch.linspace(8, n - 2, min(args.n_positions // len(prompts) + 1, n - 9)).long()
            acts[l].extend(resid[l][idx])

    g = torch.Generator(device="cuda").manual_seed(0)
    out = {}
    for l in layers:
        occs, var_at_occ = [], []
        for x in acts[l][: args.n_positions]:
            # GP incremental reconstruction curve
            gp_frac = [0.0]
            ids, co, recon = kit.gradient_pursuit(x, l, 1)
            for K in range(1, args.kmax + 1):
                ids, co, recon = kit.gradient_pursuit(x, l, K)
                gp_frac.append(float(recon.norm() ** 2 / x.norm() ** 2))
            # random control: SAME greedy nonneg pursuit over a random dictionary
            Rdict = torch.randn(4096, x.shape[0], device="cuda", generator=g)
            Rdict = Rdict / Rdict.norm(dim=1, keepdim=True)
            rand_frac = [0.0]
            sel, r = [], x.clone()
            import torch as _t
            for K in range(1, args.kmax + 1):
                corr = Rdict @ r
                if sel:
                    corr[_t.tensor(sel, device=x.device)] = -_t.inf
                b_i = int(corr.argmax())
                sel.append(b_i)
                A = Rdict[sel]
                G = A @ A.T
                bb = A @ x
                cc = _t.linalg.lstsq(G + 1e-6 * _t.eye(len(sel), device=x.device), bb.unsqueeze(1)).solution.squeeze(1).clamp_min(0)
                step = 1.0 / (_t.linalg.matrix_norm(G, 2) + 1e-6)
                for _ in range(200):
                    cc = (cc - step * (G @ cc - bb)).clamp_min(0)
                r = x - A.T @ cc
                rand_frac.append(float(1 - r.norm() ** 2 / x.norm() ** 2))
            occ = args.kmax
            for K in range(2, args.kmax + 1):
                dgp = gp_frac[K] - gp_frac[K - 1]
                drand = rand_frac[K] - rand_frac[K - 1]
                if dgp < drand:
                    occ = K - 1
                    break
            occs.append(occ)
            var_at_occ.append(gp_frac[occ] - rand_frac[occ])
        occs.sort()
        out[l] = {"occupancy_median": occs[len(occs) // 2],
                  "occupancy_p25": occs[len(occs) // 4],
                  "occupancy_p75": occs[3 * len(occs) // 4],
                  "excess_var_at_occ_mean": sum(var_at_occ) / len(var_at_occ)}
        print(f"L{l}: occupancy median {out[l]['occupancy_median']} "
              f"[{out[l]['occupancy_p25']}-{out[l]['occupancy_p75']}], "
              f"excess var {out[l]['excess_var_at_occ_mean']:.3f}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
