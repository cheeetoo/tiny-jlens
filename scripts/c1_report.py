"""C1 — Verbal report (paper §3.1, BRIEF C1).

Sub-experiments:
  1a correlation: Spearman(lens ranking, output ranking) over category candidates
     at the readout anchor, per layer.
  1b swap-to-report: J-lens coordinate swap source->target across prompt
     positions and band layers; does the report follow?
  1c privilege: concept-vector split into J-component (gradient pursuit k=16)
     vs non-J remainder; matched-norm swaps; clamp control for the remainder.

Usage: python scripts/c1_report.py --lens runs/smollm2-135m-it/lens.pt [--explore]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import transformers

import jlens
from scipy.stats import spearmanr

from tinyjlens.lensops import LensKit
from tinyjlens.prompts import build_chat, single_token_id
from tinyjlens.interventions import (
    LayerEdit, clamped_swap_edits, proj_swap_edits, add_delta_edit, clamp_coords_edit, logits_with_edits,
)

REF = os.path.join(os.path.dirname(__file__), "..", "ref", "jacobian-lens", "data")


def rank_of(logits: torch.Tensor, token_id: int) -> int:
    return int((logits > logits[token_id]).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--lens", default="runs/smollm2-135m-it/lens.pt")
    ap.add_argument("--band", default=None, help="lo:hi band layers, e.g. 12:26")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--swap-mode", default="coord", choices=["coord", "proj"])
    ap.add_argument("--k-gp", type=int, default=16)
    ap.add_argument("--out", default="runs/c1_report.json")
    ap.add_argument("--n-targets", type=int, default=6, help="swap targets per category")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args()

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
    print(f"band: L{band[0]}..L{band[-1]}")

    with open(os.path.join(REF, "experiments", "verbal-report.json")) as f:
        candidates = json.load(f)["candidates"]
    if args.confirm:
        from tinyjlens.confirm_pools import CONFIRM_CATEGORIES
        candidates = {**candidates, **CONFIRM_CATEGORIES}

    # candidates as variant token-id sets; answers follow "... is" so the
    # leading-space forms are the relevant ones
    from tinyjlens.prompts import variant_token_ids

    cand_tokens = {}
    for cat, members in candidates.items():
        toks = {}
        for m in members:
            ids = [t for t in variant_token_ids(tok, m)
                   if tok.decode([t]).startswith(" ")]
            if ids:
                toks[m] = ids
        if len(toks) >= 6:
            cand_tokens[cat] = toks
    print("categories with >=6 single-token candidates:", list(cand_tokens))

    ARTICLES = {" a", " an", " the", " my", " one", " called", ":", " :", '"', ' "'}

    def report_prompt(cat):
        """Build the report prompt, extending the prefill through any article
        tokens so the readout anchor sits immediately before the content word
        (the model's actual report)."""
        prefill = f"The {cat} I am thinking of is"
        for _ in range(3):
            bp = build_chat(tok, f"Think of a {cat}. Answer with one word.",
                            assistant_prefill=prefill)
            lg = logits_with_edits(kit, bp.input_ids, [])[-1]
            top = tok.decode([int(lg.argmax())])
            if top in ARTICLES or top.strip() in {"a", "an", "the"}:
                prefill += top
                continue
            return bp
        return bp

    results = {"band": [band[0], band[-1]], "alpha": args.alpha}

    # ---------- 1a: correlation ----------
    def variant_max(logits, variant_ids):
        return max(float(logits[t]) for t in variant_ids)

    corr = {l: [] for l in band}
    per_cat = {}
    for cat, toks in cand_tokens.items():
        bp = report_prompt(cat)
        pos = bp.n_tokens - 1
        resid = kit.residuals(bp.input_ids, band + [n_layers - 1])
        out_logits = kit.model.unembed(resid[n_layers - 1][pos]).float()
        out_scores = [variant_max(out_logits, v) for v in toks.values()]
        per_cat[cat] = {}
        for l in band:
            ll = kit.lens_logits(resid[l][pos], l)
            lens_scores = [variant_max(ll, v) for v in toks.values()]
            rho = spearmanr(out_scores, lens_scores).statistic
            corr[l].append(float(rho))
            per_cat[cat][l] = float(rho)
    results["corr_1a"] = {
        "per_layer_mean": {l: sum(v) / len(v) for l, v in corr.items()},
        "per_cat": per_cat,
    }
    print("1a mean Spearman by layer:",
          {l: round(sum(v) / len(v), 3) for l, v in corr.items()})

    # ---------- 1b: swap-to-report ----------
    g = torch.Generator().manual_seed(args.seed)

    def pick_target_form(source_id: int, variant_ids: list[int]) -> int:
        """Prefer the target variant whose case matches the source token's."""
        src = tok.decode([source_id])
        for t in variant_ids:
            if tok.decode([t])[1:2].isupper() == src[1:2].isupper():
                return t
        return variant_ids[0]

    def min_rank(logits, variant_ids):
        return min(rank_of(logits, t) for t in variant_ids)

    def pick_source(clean_logits, toks):
        """The model's spontaneously chosen item: the raw argmax if it is a
        clean word token, else the highest-probability candidate-list token
        (fragment/markup argmaxes like ' co', '**' have no usable lens vector)."""
        raw = int(clean_logits.argmax())
        dec = tok.decode([raw])
        if dec.startswith(" ") and dec[1:].isalpha() and len(dec) >= 4:
            return raw
        all_variants = [t for v in toks.values() for t in v]
        return max(all_variants, key=lambda t: float(clean_logits[t]))

    swap_trials = []
    for cat, toks in cand_tokens.items():
        bp = report_prompt(cat)
        pos = bp.n_tokens - 1
        clean_logits = logits_with_edits(kit, bp.input_ids, [])[pos]
        top10 = set(clean_logits.topk(10).indices.tolist())
        source = pick_source(clean_logits, toks)
        eligible = [(m, v) for m, v in toks.items()
                    if not (set(v) & top10) and source not in v]
        order = torch.randperm(len(eligible), generator=g).tolist()
        for j in order[: args.n_targets]:
            m, variants = eligible[j]
            target = pick_target_form(source, variants)
            swap_fn = proj_swap_edits if args.swap_mode == 'proj' else clamped_swap_edits
            edits = swap_fn(kit, bp.input_ids, band, [source], [target], args.alpha)
            logits = logits_with_edits(kit, bp.input_ids, edits)[pos]
            swap_trials.append({
                "cat": cat, "source": tok.decode([source]), "target": m,
                "pre_rank": min_rank(clean_logits, variants),
                "post_rank": min_rank(logits, variants),
            })
    n_top1 = sum(t["post_rank"] == 0 for t in swap_trials)
    n_top5 = sum(t["post_rank"] < 5 for t in swap_trials)
    results["swap_1b"] = {"trials": swap_trials, "n": len(swap_trials),
                          "top1": n_top1, "top5": n_top5}
    print(f"1b: {n_top1}/{len(swap_trials)} top-1, {n_top5}/{len(swap_trials)} top-5")

    # ---------- 1c: concept-vector split ----------
    # concept vectors: residual at last prompt position of "Tell me about {m}."
    concepts = [(cat, m, t) for cat, toks in cand_tokens.items() for m, t in toks.items()]
    resid_cache = {}
    for cat, m, t in concepts:
        bp = build_chat(tok, f"Tell me about {m}.")
        resid = kit.residuals(bp.input_ids, band)
        resid_cache[(cat, m)] = {l: resid[l][-1] for l in band}
    mean_all = {l: torch.stack([r[l] for r in resid_cache.values()]).mean(0) for l in band}
    concept_vec = {k: {l: v[l] - mean_all[l] for l in band} for k, v in resid_cache.items()}

    def split(vec, l):
        ids_gp, coeffs, recon = kit.gradient_pursuit(vec, l, args.k_gp)
        return recon, vec - recon, ids_gp

    split_trials = []
    for cat, toks in cand_tokens.items():
        bp = report_prompt(cat)
        pos = bp.n_tokens - 1
        clean_logits = logits_with_edits(kit, bp.input_ids, [])[pos]
        top10 = set(clean_logits.topk(10).indices.tolist())
        source_tok = pick_source(clean_logits, toks)
        source_name = tok.decode([source_tok]).strip()
        src_key = next((k for k in concept_vec if k[0] == cat and k[1].lower() == source_name.lower()), None)
        if src_key is None:
            continue
        eligible = [(m, v) for m, v in toks.items()
                    if not (set(v) & top10) and m.lower() != source_name.lower()]
        order = torch.randperm(len(eligible), generator=g).tolist()
        for j in order[:3]:
            m, variants = eligible[j]
            target_tok_id = pick_target_form(source_tok, variants)
            deltas_J, deltas_nonJ, clamp_ids = {}, {}, {}
            for l in band:
                J_s, nonJ_s, ids_s = split(concept_vec[src_key][l], l)
                J_t, nonJ_t, ids_t = split(concept_vec[(cat, m)][l], l)
                dJ = J_t - J_s
                dN = nonJ_t - nonJ_s
                dN = dN * (dJ.norm() / dN.norm().clamp_min(1e-8))
                deltas_J[l], deltas_nonJ[l] = dJ, dN
                clamp_ids[l] = list(dict.fromkeys(
                    ids_s.tolist() + ids_t.tolist() + [source_tok, target_tok_id]))
            trial = {"cat": cat, "source": source_name, "target": m}
            for name, deltas in [("J_comp", deltas_J), ("nonJ_comp", deltas_nonJ)]:
                edits = [add_delta_edit(deltas[l], l, None) for l in band]
                logits = logits_with_edits(kit, bp.input_ids, edits)[pos]
                trial[name + "_rank"] = rank_of(logits, target_tok_id)
            # clamp control for the non-J swap
            clean_resid = kit.residuals(bp.input_ids, band)
            edits = []
            for l in band:
                pinv = torch.linalg.pinv(kit.jlens_vectors(l, clamp_ids[l]).T)  # [k, d]
                coords = clean_resid[l] @ pinv.T  # [seq, k]
                ref = {p: coords[p] for p in range(bp.n_tokens)}
                edits.append(add_delta_edit(deltas_nonJ[l], l, None))
                edits.append(clamp_coords_edit(kit, l, clamp_ids[l], ref,
                                               list(range(bp.n_tokens))))
            logits = logits_with_edits(kit, bp.input_ids, edits)[pos]
            trial["nonJ_clamped_rank"] = rank_of(logits, target_tok_id)
            split_trials.append(trial)
            print(trial)
    results["split_1c"] = {
        "trials": split_trials,
        "J_top5": sum(t["J_comp_rank"] < 5 for t in split_trials),
        "nonJ_top5": sum(t["nonJ_comp_rank"] < 5 for t in split_trials),
        "nonJ_clamped_top5": sum(t["nonJ_clamped_rank"] < 5 for t in split_trials),
        "n": len(split_trials),
    }
    print("1c:", {k: v for k, v in results["split_1c"].items() if k != "trials"})

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
