"""C2 exploration variants — giving directed modulation its strongest fair shot
at 135M scale before any verdict (BRIEF R3: exploration).

Variants beyond scripts/c2_modulation.py:
  V1 hold-then-report: "Think of {a concept}. Copy this sentence: ... Then
     name the thing you thought of." The concept is behaviorally REQUIRED
     after the carrier, so maintenance during copying has task demand
     (paper's withheld-answer family, §6/appendix). Readout at carrier span:
     is the eventually-reported word in the lens DURING copying?
  V2 full-depth scan: per-layer target log-prob profile (not just the band)
     under focus vs baseline, paired by (carrier, target).
  V3 continuous metric: mean log-softmax of target over carrier span,
     focus - baseline, per layer.

Usage: python scripts/c2_variants.py --lens <lens.pt>
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

REF = os.path.join(os.path.dirname(__file__), "..", "ref", "jacobian-lens", "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--lens", default="runs/smollm2-135m-it/lens.pt")
    ap.add_argument("--out", default="runs/c2_variants.json")
    args = ap.parse_args()

    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    hf = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda()
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(args.lens)
    kit = LensKit.build(model, lens)
    layers = lens.source_layers

    with open(os.path.join(REF, "experiments", "directed-modulation.json")) as f:
        dm = json.load(f)
    carriers = dm["carrier_sentences"][:4]
    topics = dm["topic_categories"][:8]

    results = {"v1": [], "v3": {}}

    def render_chat(user, prefill=""):
        return tok.apply_chat_template([{"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True) + prefill

    # ---------- V1: hold-then-report ----------
    @torch.no_grad()
    def greedy(full, n=8):
        ids = tok(full, return_tensors="pt")["input_ids"].cuda()
        out = hf.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    for cat, hint in [("animal", "animal"), ("color", "color"), ("fruit", "fruit"),
                      ("sport", "sport"), ("country", "country"), ("beverage", "beverage"),
                      ("vegetable", "vegetable"), ("instrument", "instrument")]:
        for carrier in carriers[:3]:
            user = (f"Think of one {cat} and keep it in mind. "
                    f'First, copy this sentence exactly: "{carrier}" '
                    f"Then say which {cat} you were thinking of.")
            user_nodemand = (f"Think of one {cat} and keep it in mind. "
                             f'Copy this sentence exactly: "{carrier}"')
            full = render_chat(user, f'"{carrier}." The {cat} I was thinking of is')
            words = [w.strip('."?,!') for w in greedy(full, 10).strip().split()]
            words = [w for w in words if w and w[0].isalpha() and w.lower() not in
                     {"a", "an", "the", "my", "one", "called", "of", "was", "is"}]
            if not words:
                continue
            answer = words[0]
            tids = variant_token_ids(tok, answer)
            if not tids:
                continue
            bp = build_raw(tok, full)
            a, b = bp.find_span(carrier, occurrence=full.count(carrier) - 1)
            resid = kit.residuals(bp.input_ids, layers)
            best = min(int(kit.lens_ranks_of(resid[l][a:b], l, tids).min()) for l in layers)
            # matched no-demand control: same instruction to hold, no later use
            fullb = render_chat(user_nodemand, '"' + carrier + '."')
            bpb = build_raw(tok, fullb)
            ab, bb = bpb.find_span(carrier, occurrence=fullb.count(carrier) - 1)
            residb = kit.residuals(bpb.input_ids, layers)
            bestb = min(int(kit.lens_ranks_of(residb[l][ab:bb], l, tids).min()) for l in layers)
            results["v1"].append({"cat": cat, "carrier": carrier[:20], "answer": answer,
                                  "best_rank_demand": best, "best_rank_nodemand": bestb})
            print(f"V1 {cat}: answer={answer!r} rank demand={best} nodemand={bestb}")

    # ---------- V3: focus - baseline log-prob profile ----------
    focus_t = "Concentrate on {x} while you write the sentence."
    diffs = {l: [] for l in layers}
    for topic in topics:
        tids = []
        for m in topic["members"]:
            tids += variant_token_ids(tok, m)
        if not tids:
            continue
        for carrier in carriers[:3]:
            rows = {}
            for cond, instr in (("focus", focus_t.format(x=topic["name"]) + " "), ("base", "")):
                user = f'{instr}Copy this sentence exactly: "{carrier}"'
                full = render_chat(user, '"' + carrier + '."')
                bp = build_raw(tok, full)
                a, b = bp.find_span(carrier, occurrence=full.count(carrier) - 1)
                resid = kit.residuals(bp.input_ids, layers)
                rows[cond] = {}
                for l in layers:
                    lg = torch.log_softmax(kit.lens_logits(resid[l][a:b], l), dim=-1)
                    rows[cond][l] = float(lg[:, tids].max(dim=1).values.mean())
            for l in layers:
                diffs[l].append(rows["focus"][l] - rows["base"][l])
    results["v3"] = {l: sum(v) / len(v) for l, v in diffs.items() if v}
    print("\nV3 focus-minus-baseline mean target logprob by layer:")
    for l in layers:
        if l in results["v3"]:
            print(f"  L{l:>2}: {results['v3'][l]:+.3f}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
