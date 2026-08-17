"""Methodological comparison at tiny scale (paper §A quantitative comparisons):
J-lens vs logit lens at recovering known unspoken intermediates.

Items: the C3 capability-filtered two-hop pool (ref probe-swap + custom).
Metric: pass@k (intermediate variant tokens' min rank over layers <= k) at the
final prompt position, per lens; plus per-layer recovery profiles.

Usage: python scripts/lenseval_compare.py --lens runs/smollm2-135m-it/lens_explore.pt
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
from tinyjlens.prompts import build_raw, variant_token_ids
from tinyjlens.twohop_pool import build_items


def first_token_id(tok, word):
    ids = tok(" " + word, add_special_tokens=False)["input_ids"]
    for i in ids:
        if tok.decode([i]).strip():
            return i
    return ids[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--lens", default="runs/smollm2-135m-it/lens_explore.pt")
    ap.add_argument("--out", default="runs/lenseval_compare.json")
    args = ap.parse_args()

    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    hf = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda()
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(args.lens)
    kit = LensKit.build(model, lens)
    layers = lens.source_layers

    REF = os.path.join(os.path.dirname(__file__), "..", "ref", "jacobian-lens", "data")
    with open(os.path.join(REF, "experiments", "probe-swap.json")) as f:
        ref_items = json.load(f)["items"]
    items = ref_items + build_items()

    # capability filter (same as C3)
    kept = []
    for it in items:
        bp = build_raw(tok, it["prompt"].rstrip())
        resid = kit.residuals(bp.input_ids, [model.n_layers - 1])
        lg = kit.model.unembed(resid[model.n_layers - 1][-1]).float()
        if int(lg.argmax()) == first_token_id(tok, it["answer"]):
            ids = variant_token_ids(tok, it["intermediate"])
            if ids:
                kept.append((it, ids))
    print(f"{len(kept)} capability-filtered items")

    rows = []
    for it, inter_ids in kept:
        bp = build_raw(tok, it["prompt"].rstrip())
        resid = kit.residuals(bp.input_ids, list(layers))
        jr, lr = {}, {}
        for l in layers:
            h = resid[l][-1]
            jlogits = kit.lens_logits(h, l)
            llogits = kit.model.unembed(h).float()
            jr[l] = min(int((jlogits > jlogits[t]).sum()) for t in inter_ids)
            lr[l] = min(int((llogits > llogits[t]).sum()) for t in inter_ids)
        rows.append({"name": it.get("name", it["intermediate"]),
                     "jlens_min": min(jr.values()), "logit_min": min(lr.values()),
                     "jlens_by_layer": jr, "logit_by_layer": lr})

    out = {"n": len(rows), "rows": rows}
    for k in (1, 5, 10, 25, 100):
        ja = sum(r["jlens_min"] < k for r in rows)
        la = sum(r["logit_min"] < k for r in rows)
        out[f"pass@{k}"] = {"jlens": ja, "logit": la}
        print(f"pass@{k}: J-lens {ja}/{len(rows)}  logit {la}/{len(rows)}")

    # per-layer recovery: fraction of items with intermediate in top-10 at layer l
    print("\nper-layer top-10 recovery (J-lens | logit lens):")
    for l in layers:
        jf = sum(r["jlens_by_layer"][l] < 10 for r in rows) / len(rows)
        lf = sum(r["logit_by_layer"][l] < 10 for r in rows) / len(rows)
        print(f"  L{l:>2}: {jf:.2f} | {lf:.2f}")
        out[f"L{l}"] = {"jlens": jf, "logit": lf}

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
