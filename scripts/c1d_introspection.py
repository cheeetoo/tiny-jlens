"""C1d — Injected-thought introspection (paper §3.1 / Fig. 7; secondary per BRIEF).

Protocol (mirrors data/experiments/verbal-introspection.json):
- multi-turn framing + trial question; assistant prefill ends in an open quote
- steer the concept's J-lens direction (unit-normalized x layer mean residual
  norm x strength) at every band layer on every user-question token
- score: rank of the concept's surface token in the next-token distribution at
  the open quote; strength 0 = control
- selectivity: the concept must NOT become the top prediction at the other
  prefill positions (position control, paper Fig. 7)

Usage: python scripts/c1d_introspection.py --lens runs/smollm2-135m-it/lens.pt
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
from tinyjlens.prompts import build_raw
from tinyjlens.interventions import steer_edit, logits_with_edits
from tinyjlens.corpus import wikitext_prompts

REF = os.path.join(os.path.dirname(__file__), "..", "ref", "jacobian-lens", "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--lens", default="runs/smollm2-135m-it/lens.pt")
    ap.add_argument("--band", default=None)
    ap.add_argument("--strengths", default="0,1,2,4,8,16")
    ap.add_argument("--n-concepts", type=int, default=40)
    ap.add_argument("--simple-prompt", action="store_true",
                    help="single-turn simplified framing for tiny models")
    ap.add_argument("--out", default="runs/c1d_introspection.json")
    args = ap.parse_args()
    strengths = [float(s) for s in args.strengths.split(",")]

    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    hf = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda()
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(args.lens)
    kit = LensKit.build(model, lens)
    n_layers = model.n_layers
    if args.band:
        lo, hi = map(int, args.band.split(":"))
    else:
        lo, hi = int(n_layers * 0.4), int(n_layers * 0.87)
    band = [l for l in range(lo, hi + 1) if l in lens.source_layers]

    with open(os.path.join(REF, "experiments", "verbal-introspection.json")) as f:
        vi = json.load(f)

    # mean residual norms per layer over wikitext
    norm_cache_path = f"runs/{os.path.basename(args.lens).split('.')[0]}_residnorms.json"
    if os.path.exists(norm_cache_path):
        mean_norms = {int(k): v for k, v in json.load(open(norm_cache_path)).items()}
    else:
        sums = {l: [] for l in lens.source_layers}
        for p in wikitext_prompts(8, skip=8000):
            bp = build_raw(tok, p[:500])
            resid = kit.residuals(bp.input_ids, lens.source_layers)
            for l in lens.source_layers:
                sums[l].append(float(resid[l][8:].norm(dim=1).mean()))
        mean_norms = {l: sum(v) / len(v) for l, v in sums.items()}
        json.dump(mean_norms, open(norm_cache_path, "w"))
    print("mean resid norms (band):", {l: round(mean_norms[l], 1) for l in band})

    # build the prompt
    if args.simple_prompt:
        msgs = [{"role": "user",
                 "content": "A thought has been injected into your mind. What is the thought about? Answer with one word."}]
        prefill = ' The thought is about "'
    else:
        msgs = [dict(m) for m in vi["intro_prompt"] if m["content"]]
        prefill = vi["prefills"]["default"].lstrip()
        prefill = " " + prefill if not prefill.startswith(" ") else prefill
    rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = rendered + prefill
    bp = build_raw(tok, full)
    # user question span: last user message content
    q_text = msgs[-1]["content"].strip()
    qa, qb = bp.find_span(q_text[:60])
    q_positions = list(range(qa, bp.n_tokens - len(tok(prefill, add_special_tokens=False)["input_ids"])))
    quote_pos = bp.n_tokens - 1
    prefill_positions = [p for p in range(quote_pos - 8, quote_pos)]
    print(f"prompt tokens: {bp.n_tokens}, steer span: {q_positions[0]}..{q_positions[-1]}, readout at {quote_pos}")

    concepts = []
    for c in vi["concepts"][: args.n_concepts]:
        ids = tok(c["surface"], add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            concepts.append((c["surface"], ids[0]))
    print(f"{len(concepts)} single-token concepts")

    results = {"band": [band[0], band[-1]], "strengths": strengths, "trials": []}
    for surface, tid in concepts:
        space_tid = tok(" " + surface, add_special_tokens=False)["input_ids"]
        steer_tid = space_tid[0] if len(space_tid) == 1 else tid
        row = {"surface": surface}
        for s in strengths:
            edits = [] if s == 0 else [
                steer_edit(kit, l, steer_tid, s, mean_norms[l], q_positions) for l in band]
            lg = logits_with_edits(kit, bp.input_ids, edits)
            r = int((lg[quote_pos] > lg[quote_pos, tid]).sum())
            others = [int((lg[p] > lg[p, tid]).sum()) for p in prefill_positions]
            row[f"rank_s{s:g}"] = r
            row[f"other_min_rank_s{s:g}"] = min(others)
        results["trials"].append(row)
        print(row)

    for s in strengths:
        rr = [1.0 / (1 + r[f"rank_s{s:g}"]) for r in results["trials"]]
        rr.sort()
        med = rr[len(rr) // 2]
        top1 = sum(r[f"rank_s{s:g}"] == 0 for r in results["trials"])
        blurt = sum(r[f"other_min_rank_s{s:g}"] == 0 for r in results["trials"])
        print(f"strength {s:g}: median RR {med:.3f}, top1 {top1}/{len(results['trials'])}, "
              f"blurt-at-other-positions {blurt}")
        results[f"summary_s{s:g}"] = {"median_rr": med, "top1": top1, "blurt": blurt}

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
