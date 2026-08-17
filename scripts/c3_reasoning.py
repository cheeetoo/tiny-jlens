"""C3 — Internal reasoning (paper §3.3, BRIEF C3).

Phases:
  filter   capability filter: greedy next token == answer (paper's baseline)
  readout  intermediate's best lens rank over band layers x prompt positions
  swap     intermediate coordinate swap -> does the answer follow? (a2 too)
  timing   sliding-window swaps: intermediate bites earlier than answer swap
  probe    mean-difference probe split: J-component vs non-J remainder, clamp

Usage: python scripts/c3_reasoning.py --lens runs/smollm2-135m-it/lens.pt
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
from tinyjlens.interventions import clamped_swap_edits, add_delta_edit, clamp_coords_edit, logits_with_edits
from tinyjlens.twohop_pool import build_items, LANG_CAPITAL, CITY_LANGUAGE

REF = os.path.join(os.path.dirname(__file__), "..", "ref", "jacobian-lens", "data")


def first_token_id(tok, word):
    ids = tok(" " + word, add_special_tokens=False)["input_ids"]
    for i in ids:
        if tok.decode([i]).strip():
            return i
    return ids[0]


def rank_of(logits, tid):
    return int((logits > logits[tid]).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--lens", default="runs/smollm2-135m-it/lens.pt")
    ap.add_argument("--band", default=None)
    ap.add_argument("--alphas", default="1,2")
    ap.add_argument("--out", default="runs/c3_reasoning.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phases", default="filter,readout,swap,timing,probe")
    ap.add_argument("--k-gp", type=int, default=25)
    ap.add_argument("--timing-window", type=int, default=4)
    args = ap.parse_args()
    phases = set(args.phases.split(","))
    alphas = [float(a) for a in args.alphas.split(",")]

    torch.manual_seed(args.seed)
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
    results = {"band": [band[0], band[-1]]}

    # ---------- assemble items ----------
    with open(os.path.join(REF, "experiments", "probe-swap.json")) as f:
        ref_items = json.load(f)["items"]
    for it in ref_items:
        it["family"] = "ref/" + it["category"]
    custom = build_items()
    items = ref_items + custom

    # ---------- filter ----------
    kept = []
    for it in items:
        bp = build_raw(tok, it["prompt"].rstrip())
        logits = logits_with_edits(kit, bp.input_ids, [])[-1]
        ans_tid = first_token_id(tok, it["answer"])
        ok = int(logits.argmax()) == ans_tid
        inter_ids = variant_token_ids(tok, it["intermediate"])
        if ok and inter_ids:
            it = dict(it, ans_tid=ans_tid, inter_ids=inter_ids, bp_text=bp.text)
            kept.append(it)
    results["filter"] = {"n_total": len(items), "n_kept": len(kept),
                         "kept": [it["family"] + ":" + it.get("name", it["intermediate"]) for it in kept]}
    print(f"filter: {len(kept)}/{len(items)} items pass"
          f" ({len([i for i in kept if i['family'].startswith('ref/')])} ref)")

    # assign swap partners within family (custom) or from item fields (ref)
    g = torch.Generator().manual_seed(args.seed)
    by_family = {}
    for it in kept:
        by_family.setdefault(it["family"], []).append(it)
    for it in kept:
        if "swap_to" in it:
            it["swap_inter"], it["swap_answer_"] = it["swap_to"], it["swap_answer"]
        else:
            fam = [x for x in by_family[it["family"]]
                   if x["intermediate"] != it["intermediate"] and x["answer"] != it["answer"]]
            if not fam:
                it["swap_inter"] = None
                continue
            j = int(torch.randint(len(fam), (1,), generator=g))
            it["swap_inter"], it["swap_answer_"] = fam[j]["intermediate"], fam[j]["answer"]
    kept = [it for it in kept if it.get("swap_inter")]

    # ---------- readout ----------
    if "readout" in phases:
        n_top10_final, n_top10_any = 0, 0
        for it in kept:
            bp = build_raw(tok, it["bp_text"])
            resid = kit.residuals(bp.input_ids, band)
            # primary: final prompt position (paper's lens-eval convention),
            # min over band layers. secondary: best over all positions (echo-
            # contaminated near the cue token; diagnostic only)
            fin_best, fin_layer = 10**9, None
            any_best = (10**9, None, None)
            for l in band:
                ranks = kit.lens_ranks_of(resid[l], l, it["inter_ids"])  # [seq, k]
                rf = int(ranks[-1].min())
                if rf < fin_best:
                    fin_best, fin_layer = rf, l
                r = int(ranks.min())
                if r < any_best[0]:
                    pos = int(ranks.min(dim=1).values.argmin())
                    any_best = (r, l, pos)
            it["inter_final_rank"], it["inter_final_layer"] = fin_best, fin_layer
            it["inter_best_rank"], it["inter_best_layer"], it["inter_best_pos"] = any_best
            n_top10_final += fin_best < 10
            n_top10_any += any_best[0] < 10
        results["readout"] = {
            "n": len(kept), "n_top10_final": n_top10_final, "n_top10_any": n_top10_any,
            "items": [{k: it[k] for k in ("family", "intermediate", "answer",
                                          "inter_final_rank", "inter_final_layer",
                                          "inter_best_rank", "inter_best_layer", "inter_best_pos")}
                      for it in kept]}
        print(f"readout: intermediate in lens top-10 — final pos {n_top10_final}/{len(kept)}, "
              f"any pos {n_top10_any}/{len(kept)}")

    # ---------- swap ----------
    def paired_forms(a: str, b: str):
        pairs = []
        for pre in (" ", ""):
            ia = tok(pre + a, add_special_tokens=False)["input_ids"]
            ib = tok(pre + b, add_special_tokens=False)["input_ids"]
            if len(ia) == 1 and len(ib) == 1:
                pairs.append((ia[0], ib[0]))
        return pairs

    if "swap" in phases:
        swap_results = []
        for it in kept:
            pairs = paired_forms(it["intermediate"], it["swap_inter"])
            if not pairs:
                continue
            tgt_tid = first_token_id(tok, it["swap_answer_"])
            if tgt_tid == it["ans_tid"]:
                continue  # degenerate: swap answer's first token == original's
            bp = build_raw(tok, it["bp_text"])
            clean = logits_with_edits(kit, bp.input_ids, [])[-1]
            row = {"family": it["family"], "inter": it["intermediate"],
                   "swap_to": it["swap_inter"], "want": it["swap_answer_"],
                   "pre_rank": rank_of(clean, tgt_tid)}
            src_ids = [p[0] for p in pairs]
            tgt_ids = [p[1] for p in pairs]
            for a in alphas:
                edits = clamped_swap_edits(kit, bp.input_ids, band, src_ids, tgt_ids, a)
                logits = logits_with_edits(kit, bp.input_ids, edits)[-1]
                row[f"post_rank_a{a:g}"] = rank_of(logits, tgt_tid)
                row[f"dlp_a{a:g}"] = float(
                    torch.log_softmax(logits, -1)[tgt_tid]
                    - torch.log_softmax(clean, -1)[tgt_tid])
            swap_results.append(row)
        n = len(swap_results)
        summary = {f"top1_a{a:g}": sum(r[f"post_rank_a{a:g}"] == 0 for r in swap_results) for a in alphas}
        results["swap"] = {"n": n, **summary, "trials": swap_results}
        print(f"swap: {summary} of n={n}")

    # ---------- crossfn: cross-function consistency (anti-smuggling) ----------
    if "crossfn" in phases:
        # country pairs present in BOTH lang-capital and city-language kept sets
        lc = {it["intermediate"]: it for it in kept if it["family"] == "lang-capital"}
        cl = {it["intermediate"]: it for it in kept if it["family"] == "city-language"}
        shared = sorted(set(lc) & set(cl))
        cross = []
        for src_c in shared:
            for tgt_c in shared:
                if src_c == tgt_c:
                    continue
                pairs = paired_forms(src_c, tgt_c)
                if not pairs:
                    continue
                row = {"src": src_c, "tgt": tgt_c}
                ok_both = True
                any_flip = False
                for fam, pool in (("lang-capital", lc), ("city-language", cl)):
                    it_s, it_t = pool[src_c], pool[tgt_c]
                    want = first_token_id(tok, it_t["answer"])
                    orig = first_token_id(tok, it_s["answer"])
                    if want == orig:
                        ok_both = False
                        break
                    bp = build_raw(tok, it_s["bp_text"])
                    edits = clamped_swap_edits(kit, bp.input_ids, band,
                                               [p[0] for p in pairs], [p[1] for p in pairs], 1.0)
                    lg = logits_with_edits(kit, bp.input_ids, edits)[-1]
                    flipped = int(lg.argmax()) == want
                    row[fam] = {"want": it_t["answer"], "flipped": flipped,
                                "post_rank": rank_of(lg, want)}
                    any_flip = any_flip or flipped
                    ok_both = ok_both and flipped
                if "lang-capital" in row and "city-language" in row:
                    row["both"] = ok_both
                    row["any"] = any_flip
                    cross.append(row)
        n_any = sum(r["any"] for r in cross)
        n_both = sum(r["both"] for r in cross)
        results["crossfn"] = {"n_pairs": len(cross), "n_any_flip": n_any,
                              "n_both_flip": n_both, "trials": cross}
        print(f"crossfn: {n_both}/{len(cross)} pairs flip BOTH functions "
              f"({n_any} flip at least one)")

    # ---------- timing ----------
    if "timing" in phases:
        window = args.timing_window
        starts = [l for l in range(0, n_layers - window) if all(
            (l + w) in lens.source_layers for w in range(window))]
        timing = []
        for it in kept:
            pairs_i = paired_forms(it["intermediate"], it["swap_inter"])
            pairs_a = paired_forms(it["answer"], it["swap_answer_"])
            if not pairs_i or not pairs_a:
                continue
            bp = build_raw(tok, it["bp_text"])
            clean = torch.log_softmax(logits_with_edits(kit, bp.input_ids, [])[-1], -1)
            tgt_tid = first_token_id(tok, it["swap_answer_"])

            def depth_profile(pairs):
                effs = []
                for s in starts:
                    lays = [s + w for w in range(window)]
                    edits = clamped_swap_edits(kit, bp.input_ids, lays,
                                               [p[0] for p in pairs], [p[1] for p in pairs], 1.0)
                    lg = torch.log_softmax(logits_with_edits(kit, bp.input_ids, edits)[-1], -1)
                    effs.append(float(lg[tgt_tid] - clean[tgt_tid]))
                return effs

            eff_i = depth_profile(pairs_i)
            eff_a = depth_profile(pairs_a)

            def onset(effs):
                mx = max(effs)
                if mx <= 0.5:
                    return None
                for s, e in zip(starts, effs):
                    if e >= 0.5 * mx:
                        return s
                return None

            oi, oa = onset(eff_i), onset(eff_a)
            timing.append({"family": it["family"], "inter": it["intermediate"],
                           "onset_inter": oi, "onset_answer": oa,
                           "eff_inter": eff_i, "eff_answer": eff_a})
        diffs = [t["onset_answer"] - t["onset_inter"] for t in timing
                 if t["onset_inter"] is not None and t["onset_answer"] is not None]
        results["timing"] = {"n": len(timing), "n_both_onsets": len(diffs),
                             "median_answer_minus_inter": (sorted(diffs)[len(diffs) // 2] if diffs else None),
                             "diffs": diffs, "trials": timing}
        print(f"timing: median(answer_onset - inter_onset) = "
              f"{results['timing']['median_answer_minus_inter']} over {len(diffs)} items")

    # ---------- probe ----------
    if "probe" in phases:
        # country probes from multiple implication prompts
        # probe prompts use surface cues DIFFERENT from both trial families
        # (currency/continent/food/borders), so no trial prompt is in its own
        # probe mean (paper: "different surface cues... different questions")
        probe_prompts = {}
        for lang, country, capital in LANG_CAPITAL:
            probe_prompts.setdefault(country, []).extend([
                f"Fact: The currency used in the country where people speak {lang} is the",
                f"Fact: The continent where {lang} is the main national language is",
                f"Fact: The flag of the country where most people speak {lang} has the colors",
            ])
        for city, country, language in CITY_LANGUAGE:
            probe_prompts.setdefault(country, []).append(
                f"Fact: The continent of the country whose capital city is {city} is")
        countries = sorted(probe_prompts)
        raw_means = {}
        for c in countries:
            resids = []
            for p in probe_prompts[c]:
                bp = build_raw(tok, p)
                r = kit.residuals(bp.input_ids, band)
                resids.append({l: r[l][-1] for l in band})
            raw_means[c] = {l: torch.stack([r[l] for r in resids]).mean(0) for l in band}
        grand = {l: torch.stack([raw_means[c][l] for c in countries]).mean(0) for l in band}
        probes = {c: {l: raw_means[c][l] - grand[l] for l in band} for c in countries}

        probe_trials = []
        cand_items = [it for it in kept if it["family"] in ("lang-capital", "city-language")
                      and it["intermediate"] in probes and it["swap_inter"] in probes]
        for it in cand_items:
            bp = build_raw(tok, it["bp_text"])
            tgt_tid = first_token_id(tok, it["swap_answer_"])
            clean = logits_with_edits(kit, bp.input_ids, [])[-1]
            deltas_J, deltas_N, clamp_ids, var_share = {}, {}, {}, []
            for l in band:
                ids_s, co_s, rec_s = kit.gradient_pursuit(probes[it["intermediate"]][l], l, args.k_gp)
                ids_t, co_t, rec_t = kit.gradient_pursuit(probes[it["swap_inter"]][l], l, args.k_gp)
                var_share.append(float((rec_s.norm() ** 2) /
                                       (probes[it["intermediate"]][l].norm() ** 2 + 1e-9)))
                dJ = rec_t - rec_s
                dN = (probes[it["swap_inter"]][l] - rec_t) - (probes[it["intermediate"]][l] - rec_s)
                dN = dN * (dJ.norm() / dN.norm().clamp_min(1e-8))
                deltas_J[l], deltas_N[l] = dJ, dN
                name_ids = variant_token_ids(tok, it["intermediate"]) + variant_token_ids(tok, it["swap_inter"])
                clamp_ids[l] = list(dict.fromkeys(ids_s.tolist() + ids_t.tolist() + name_ids))
            row = {"family": it["family"], "inter": it["intermediate"], "swap_to": it["swap_inter"],
                   "want": it["swap_answer_"], "pre_rank": rank_of(clean, tgt_tid),
                   "J_var_share": sum(var_share) / len(var_share)}
            for nm, deltas in (("full", {l: probes[it["swap_inter"]][l] - probes[it["intermediate"]][l] for l in band}),
                               ("J", deltas_J), ("nonJ", deltas_N)):
                edits = [add_delta_edit(deltas[l], l, None) for l in band]
                lg = logits_with_edits(kit, bp.input_ids, edits)[-1]
                row[nm + "_rank"] = rank_of(lg, tgt_tid)
            clean_resid = kit.residuals(bp.input_ids, band)
            edits = []
            for l in band:
                V = kit.jlens_vectors(l, clamp_ids[l])
                pinv = torch.linalg.pinv(V.T)
                ref = {p: (pinv @ clean_resid[l][p]) for p in range(bp.n_tokens)}
                edits.append(add_delta_edit(deltas_N[l], l, None))
                edits.append(clamp_coords_edit(kit, l, clamp_ids[l], ref, list(range(bp.n_tokens))))
            lg = logits_with_edits(kit, bp.input_ids, edits)[-1]
            row["nonJ_clamped_rank"] = rank_of(lg, tgt_tid)
            probe_trials.append(row)
            print(row)
        results["probe"] = {
            "n": len(probe_trials),
            "full_top1": sum(r["full_rank"] == 0 for r in probe_trials),
            "J_top1": sum(r["J_rank"] == 0 for r in probe_trials),
            "nonJ_top1": sum(r["nonJ_rank"] == 0 for r in probe_trials),
            "nonJ_clamped_top1": sum(r["nonJ_clamped_rank"] == 0 for r in probe_trials),
            "mean_J_var_share": (sum(r["J_var_share"] for r in probe_trials) / max(len(probe_trials), 1)),
            "trials": probe_trials}
        print("probe:", {k: v for k, v in results["probe"].items() if k != "trials"})

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
