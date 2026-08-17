"""C2 — Directed modulation (paper §3.2, BRIEF C2).

The model copies a carrier sentence (teacher-forced) while an instruction
tells it to think about / ignore a target. We read the lens at the copied
tokens and measure target hit rates by condition:
  focus > mention > ignore > baseline(≈0)   [paper Fig. 10/65 pattern]

Usage: python scripts/c2_modulation.py --lens runs/smollm2-135m-it/lens.pt
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
    ap.add_argument("--band", default=None)
    ap.add_argument("--hit-rank", type=int, default=1, help="hit if rank < this")
    ap.add_argument("--n-carriers", type=int, default=4)
    ap.add_argument("--n-topics", type=int, default=10)
    ap.add_argument("--n-math", type=int, default=8)
    ap.add_argument("--out", default="runs/c2_modulation.json")
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args()

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

    with open(os.path.join(REF, "experiments", "directed-modulation.json")) as f:
        dm = json.load(f)
    group_kind = dm["group_kind"]
    carriers = dm["carrier_sentences"][: args.n_carriers]
    if args.confirm:
        from tinyjlens.confirm_pools import CONFIRM_CARRIERS
        carriers = carriers + CONFIRM_CARRIERS
    topics = dm["topic_categories"][: args.n_topics]
    maths = [m for m in dm["math_problems"] if m["tier"] == 1][: args.n_math]

    def render(instruction: str | None, carrier: str) -> tuple[str, str]:
        if instruction:
            user = f'{instruction} Copy this sentence exactly: "{carrier}"'
        else:
            user = f'Copy this sentence exactly: "{carrier}"'
        prefix = tok.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        return prefix + '"' + carrier + '."', carrier

    @torch.no_grad()
    def trial(instruction: str | None, carrier: str, target_ids: list[int]) -> dict:
        full, _ = render(instruction, carrier)
        bp = build_raw(tok, full)
        # carrier span inside the assistant turn = last occurrence
        a, b = bp.find_span(carrier, occurrence=full.count(carrier) - 1)
        resid = kit.residuals(bp.input_ids, band)
        best = 10**9
        per_layer_best = {}
        lp_sum, lp_n, lp_max = 0.0, 0, -1e9
        for l in band:
            ranks = kit.lens_ranks_of(resid[l][a:b], l, target_ids)
            r = int(ranks.min())
            per_layer_best[l] = r
            best = min(best, r)
            lg = torch.log_softmax(kit.lens_logits(resid[l][a:b], l), dim=-1)
            tv = lg[:, target_ids].max(dim=1).values
            lp_sum += float(tv.sum()); lp_n += tv.shape[0]
            lp_max = max(lp_max, float(tv.max()))
        return {"best_rank": best, "hit": best < args.hit_rank,
                "hit10": best < 10, "mean_lp": lp_sum / max(lp_n, 1),
                "max_lp": lp_max, "per_layer_best": per_layer_best}

    # choose one phrasing per group for the main sweep + extras for robustness
    by_group: dict[str, list[str]] = {}
    for ph in dm["phrasings"]:
        by_group.setdefault(ph["group"], []).append(ph["text"])
    print("phrasing groups:", {g: len(v) for g, v in by_group.items()},
          "| group_kind:", group_kind)

    results = {"band": [band[0], band[-1]], "hit_rank": args.hit_rank, "trials": []}
    conditions = [("baseline", None)] + [
        (g, t) for g, texts in by_group.items() for t in texts[:2]]

    # topic targets
    for topic in topics:
        tids = []
        for m in topic["members"]:
            tids += variant_token_ids(tok, m)
        tids = list(dict.fromkeys(tids))
        if not tids:
            continue
        for cond, template in conditions:
            instr = template.format(x=topic["name"]) if template else None
            for carrier in carriers:
                t = trial(instr, carrier, tids)
                results["trials"].append({
                    "kind": "topic", "target": topic["name"],
                    "cond": cond, "kindof": group_kind.get(cond, "baseline") if cond != "baseline" else "baseline",
                    "carrier": carrier[:25], "best_rank": t["best_rank"], "hit": t["hit"],
                    "hit10": t["hit10"], "mean_lp": t["mean_lp"], "max_lp": t["max_lp"]})

    # math targets
    for m in maths:
        num = m["answer"]
        words = {"1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
                 "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
                 "12": "twelve", "14": "fourteen", "15": "fifteen", "16": "sixteen",
                 "18": "eighteen", "20": "twenty", "24": "twentyfour", "25": "twentyfive"}
        tids = variant_token_ids(tok, num)
        if num in words:
            tids += variant_token_ids(tok, words[num])
        tids = list(dict.fromkeys(tids))
        for cond, template in conditions:
            instr = template.format(x=f"evaluating {m['expr']}") if template else None
            for carrier in carriers:
                t = trial(instr, carrier, tids)
                results["trials"].append({
                    "kind": "math", "target": m["expr"], "answer": num,
                    "cond": cond, "kindof": group_kind.get(cond, "baseline") if cond != "baseline" else "baseline",
                    "carrier": carrier[:25], "best_rank": t["best_rank"], "hit": t["hit"],
                    "hit10": t["hit10"], "mean_lp": t["mean_lp"], "max_lp": t["max_lp"]})

    # summarize by (kind, kindof)
    summary = {}
    for tr in results["trials"]:
        key = f"{tr['kind']}/{tr['kindof']}"
        s = summary.setdefault(key, {"n": 0, "hits": 0, "hits10": 0, "ranks": [], "lps": []})
        s["n"] += 1
        s["hits"] += tr["hit"]
        s["hits10"] += tr.get("hit10", False)
        s["ranks"].append(tr["best_rank"])
        s["lps"].append(tr.get("mean_lp", 0.0))
    for k, s in sorted(summary.items()):
        rs = sorted(s["ranks"])
        s["median_rank"] = rs[len(rs) // 2]
        s["hit_rate"] = round(s["hits"] / s["n"], 3)
        s["hit10_rate"] = round(s["hits10"] / s["n"], 3)
        s["mean_lp"] = round(sum(s["lps"]) / s["n"], 3)
        del s["ranks"], s["lps"]
        print(k, s)
    results["summary"] = summary

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
