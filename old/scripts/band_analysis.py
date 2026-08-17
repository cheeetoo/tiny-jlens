"""Layer-band analysis driver (paper §4.1): locate the workspace band.

Usage:
  python scripts/band_analysis.py --lens runs/smollm2-135m-it/lens.pt
  python scripts/band_analysis.py --ckpt runs/smollm2-135m-it/lens_ckpt.pt  # partial fit
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
from tinyjlens.bands import readout_stats, effective_dim, cka_matrix
from tinyjlens.corpus import wikitext_prompts


def load_lens(args):
    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        jac = {l: state["jacobian_sum"][l] / state["n_done"] for l in state["jacobian_sum"]}
        print(f"checkpoint lens from {state['n_done']} prompts")
        d = next(iter(jac.values())).shape[0]
        return jlens.JacobianLens(jacobians=jac, n_prompts=state["n_done"], d_model=d)
    return jlens.JacobianLens.load(args.lens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--lens", default="runs/smollm2-135m-it/lens.pt")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-prompts", type=int, default=24)
    ap.add_argument("--out", default="runs/band_analysis.json")
    args = ap.parse_args()

    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    hf = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda()
    model = jlens.from_hf(hf, tok)
    lens = load_lens(args)
    kit = LensKit.build(model, lens)
    layers = lens.source_layers

    prompts = wikitext_prompts(args.n_prompts, skip=6000)  # held out from fit slice
    stats = readout_stats(kit, prompts, layers)
    mean = lambda xs: sum(xs) / len(xs)
    table = {}
    for l in layers:
        table[l] = {
            "kurtosis": round(mean(stats["kurtosis"][l]), 1),
            "top1_next": round(mean(stats["top1_next"][l]), 3),
            "top10_next": round(mean(stats["top10_next"][l]), 3),
            "autocorr": round(mean(stats["autocorr_top1"][l]), 3),
            "null": round(mean(stats["null_top1"][l]), 3),
            "logitlens_agree": round(mean(stats["logitlens_top1_agree"][l]), 3),
        }
    print(f"{'L':>3} {'kurt':>8} {'top1':>6} {'top10':>6} {'autoc':>6} {'null':>6} {'ll_agree':>8}")
    for l in layers:
        t = table[l]
        print(f"{l:>3} {t['kurtosis']:>8} {t['top1_next']:>6} {t['top10_next']:>6} "
              f"{t['autocorr']:>6} {t['null']:>6} {t['logitlens_agree']:>8}")

    ed = effective_dim(kit, layers)
    print("\neffective dim (frac dims for 80% var):",
          {l: round(ed[l]["0.8"], 3) for l in layers})

    cka = cka_matrix(kit, layers)
    print("\nCKA row L{first} vs all:".format(first=layers[0]))
    for i, l in enumerate(layers):
        print(f"L{l:>3}", " ".join(f"{cka[i, j]:.2f}" for j in range(len(layers))))

    out = {"table": table, "effective_dim": ed, "cka": cka.tolist(),
           "n_prompts_eval": args.n_prompts,
           "lens_prompts": lens.n_prompts}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
