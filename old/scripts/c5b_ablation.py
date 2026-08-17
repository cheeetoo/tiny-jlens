"""C5(b) — J-space ablation battery (paper §3.5.2, BRIEF C5b).

Ablate the top-k=10 J-lens directions at every position over a layer band
(sparing each position's clean top-10 output tokens), and measure task
retention: shallow tasks should survive, flexible tasks should collapse,
random-direction and norm-matched-noise controls should leave both intact.

Tasks (capability-filtered):
  shallow:  sentiment / forced-copy agreement / wikitext next-token agreement
  flexible: two-hop (C3-kept items) / language->country / analogy
  uncertain: one-hop recall (flex-gen cells)  [reported, not in verdict]

Usage: python scripts/c5b_ablation.py --lens runs/smollm2-135m-it/lens.pt
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
from tinyjlens.interventions import LayerEdit, topk_ablate_edit, logits_with_edits
from tinyjlens.corpus import wikitext_prompts
from tinyjlens.twohop_pool import build_items
from tinyjlens.confirm_pools import (
    CONFIRM_SENTIMENT, CONFIRM_ANALOGIES, CONFIRM_TWOHOP, CONFIRM_PASSAGES)

REF = os.path.join(os.path.dirname(__file__), "..", "ref", "jacobian-lens", "data")


def first_token_id(tok, word):
    ids = tok(" " + word, add_special_tokens=False)["input_ids"]
    for i in ids:
        if tok.decode([i]).strip():
            return i
    return ids[0]


SENTIMENT = [
    ("I loved this movie, it was wonderful and heartwarming.", "positive"),
    ("This was a terrible waste of time, I hated every minute.", "negative"),
    ("An absolute delight from start to finish.", "positive"),
    ("Boring, predictable, and badly acted.", "negative"),
    ("The best meal I have had in years.", "positive"),
    ("The service was awful and the food was cold.", "negative"),
    ("A stunning, beautiful film with a big heart.", "positive"),
    ("Dreadful. I walked out halfway through.", "negative"),
    ("Everyone should see this, it is fantastic.", "positive"),
    ("A complete disaster of a production.", "negative"),
]

ANALOGIES = [
    ("big is to small as hot is to", "cold"),
    ("up is to down as left is to", "right"),
    ("day is to night as summer is to", "winter"),
    ("happy is to sad as fast is to", "slow"),
    ("cat is to kitten as dog is to", "puppy"),
    ("hand is to glove as foot is to", "shoe"),
    ("king is to queen as man is to", "woman"),
    ("open is to closed as light is to", "dark"),
]

LANG_COUNTRY = [
    ("French", "France"), ("German", "Germany"), ("Spanish", "Spain"),
    ("Italian", "Italy"), ("Russian", "Russia"), ("Japanese", "Japan"),
    ("Polish", "Poland"), ("Swedish", "Sweden"), ("Greek", "Greece"),
    ("Chinese", "China"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--lens", default="runs/smollm2-135m-it/lens.pt")
    ap.add_argument("--band", default=None, help="full band lo:hi (heavy)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--confirm", action="store_true", help="add held-out confirm pools (confirmatory run)")
    ap.add_argument("--out", default="runs/c5b_ablation.json")
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
    full_band = [l for l in range(lo, hi + 1) if l in lens.source_layers]
    third = max(1, len(full_band) // 3)
    strengths = {
        "light": full_band[:third],
        "medium": full_band[: 2 * third],
        "heavy": full_band,
    }
    print({k: (v[0], v[-1]) for k, v in strengths.items()})

    def clean_top10(bp):
        lg = logits_with_edits(kit, bp.input_ids, [])
        return {p: set(lg[p].topk(10).indices.tolist()) for p in range(lg.shape[0])}

    def make_condition_edits(bp, cond):
        if cond == "none":
            return []
        if cond.startswith("ablate_"):
            layers = strengths[cond.split("_")[1]]
            prot = clean_top10(bp)
            return [topk_ablate_edit(kit, l, args.k, prot, None) for l in layers]
        if cond == "random_dirs":
            layers = strengths["medium"]
            g = torch.Generator(device="cuda").manual_seed(0)
            edits = []
            for l in layers:
                R = torch.randn(args.k, kit.model.d_model, generator=g, device="cuda")
                Q, _ = torch.linalg.qr(R.T)

                def fn(h, pos, Q=Q):
                    return h - (h @ Q) @ Q.T
                edits.append(LayerEdit(layer=l, fn=fn, positions=None))
            return edits
        if cond == "noise_matched":
            # add noise with norm matched to what heavy ablation removes
            layers = strengths["medium"]
            prot = clean_top10(bp)
            abl_edits = {e.layer: e for e in
                         [topk_ablate_edit(kit, l, args.k, prot, None) for l in layers]}
            g = torch.Generator(device="cuda").manual_seed(1)
            edits = []
            for l in layers:
                abl_fn = abl_edits[l].fn

                def fn(h, pos, abl_fn=abl_fn, g=g):
                    removed = h - abl_fn(h.clone(), pos)
                    noise = torch.randn(h.shape, generator=g, device=h.device)
                    noise = noise / noise.norm(dim=1, keepdim=True).clamp_min(1e-8)
                    return h + noise * removed.norm(dim=1, keepdim=True)
                edits.append(LayerEdit(layer=l, fn=fn, positions=None))
            return edits
        raise ValueError(cond)

    CONDS = ["none", "ablate_light", "ablate_medium", "ablate_heavy",
             "random_dirs", "noise_matched"]

    # ---------- assemble tasks ----------
    tasks = {}

    def chat_prompt(user, prefill):
        return tok.apply_chat_template([{"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True) + prefill

    tasks["sentiment"] = [
        (chat_prompt(f'Review: "{t}"\nIs this review positive or negative? Answer with one word.',
                     " The review is"), first_token_id(tok, w))
        for t, w in SENTIMENT]
    tasks["analogy"] = [("Fact: " + p, first_token_id(tok, w)) for p, w in ANALOGIES]
    tasks["lang_country"] = [
        ("Fact: The country where most people speak " + lang + " is", first_token_id(tok, c))
        for lang, c in LANG_COUNTRY]
    twohop = build_items() + (CONFIRM_TWOHOP if args.confirm else [])
    tasks["twohop"] = [(it["prompt"], first_token_id(tok, it["answer"])) for it in twohop]
    if args.confirm:
        tasks["sentiment"] += [
            (chat_prompt(f'Review: "{t}"\nIs this review positive or negative? Answer with one word.',
                         " The review is"), first_token_id(tok, w)) for t, w in CONFIRM_SENTIMENT]
        tasks["analogy"] += [("Fact: " + p_, first_token_id(tok, w)) for p_, w in CONFIRM_ANALOGIES]
    # passage tasks (C5a stimuli in the ablation battery; protection-clean by design)
    with open(os.path.join(REF, "experiments", "selectivity-language.json")) as f:
        _passages = json.load(f)["passages"]
    if args.confirm:
        _passages = _passages + CONFIRM_PASSAGES
    LANG_COUNTRY_MAP = {"French": "France", "German": "Germany", "Spanish": "Spain", "Italian": "Italy"}
    tasks["passage_report"] = [
        (chat_prompt(f"Here is a passage of text:\n\n{p_['text']}\n\nWhat language is this passage written in? Answer with one word.",
                     " The language is"), first_token_id(tok, p_["category"]))
        for p_ in _passages]
    tasks["passage_country"] = [
        (chat_prompt(f"Here is a passage of text:\n\n{p_['text']}\n\nIn which country is the language of this passage mainly spoken? Answer with one word.",
                     " The language of this passage is mainly spoken in"),
         first_token_id(tok, LANG_COUNTRY_MAP[p_["category"]]))
        for p_ in _passages]
    with open(os.path.join(REF, "experiments", "flexible-generalization.json")) as f:
        cats = json.load(f)["categories"]
    tasks["onehop_recall"] = [
        ("Fact: " + fn["template"].format(arg=arg), first_token_id(tok, fn["answers"][arg]))
        for c in cats for fn in c["funcs"] for arg in c["args"]]

    # capability filter: keep items the clean model gets right
    results = {"band": [full_band[0], full_band[-1]], "k": args.k, "tasks": {}}
    for name, items in tasks.items():
        kept = []
        for prompt, want in items:
            bp = build_raw(tok, prompt)
            lg = logits_with_edits(kit, bp.input_ids, [])[-1]
            if int(lg.argmax()) == want:
                kept.append((prompt, want))
        tasks[name] = kept
        print(f"task {name}: {len(kept)}/{len(items)} clean-correct")

    # copy task: forced carrier, score next-token agreement over carrier span
    with open(os.path.join(REF, "experiments", "directed-modulation.json")) as f:
        carriers = json.load(f)["carrier_sentences"][:6]

    # wikitext agreement prompts
    wiki = wikitext_prompts(6, skip=7000)

    # ---------- protection-overlap diagnostic ----------
    from tinyjlens.prompts import variant_token_ids as _vti
    overlap_stats = {}
    for name, probe_items in [("twohop", [(it["prompt"], it["intermediate"]) for it in twohop[:12]]),
                              ("lang_country", [("Fact: The country where most people speak " + l + " is", c)
                                                for l, c in LANG_COUNTRY[:6]]),
                              ("passage_report", [(tasks["passage_report"][i][0], _passages[i]["category"])
                                                  for i in range(min(6, len(_passages)))])]:
        fr = []
        for prompt, concept in probe_items:
            bp = build_raw(tok, prompt)
            clg = logits_with_edits(kit, bp.input_ids, [])
            cids = set(_vti(tok, concept))
            hits = sum(1 for p_ in range(clg.shape[0])
                       if cids & set(clg[p_].topk(10).indices.tolist()))
            fr.append(hits / clg.shape[0])
        overlap_stats[name] = sum(fr) / len(fr)
    results["protection_overlap"] = overlap_stats
    print("protection-overlap (frac positions with concept in protected set):", overlap_stats)

    # ---------- run all conditions ----------
    for cond in CONDS:
        row = {}
        for name, items in tasks.items():
            n_ok = 0
            for prompt, want in items:
                bp = build_raw(tok, prompt)
                edits = make_condition_edits(bp, cond)
                lg = logits_with_edits(kit, bp.input_ids, edits)[-1]
                n_ok += int(lg.argmax()) == want
            row[name] = {"n": len(items), "ok": n_ok}
        # copy agreement
        agree_tot, agree_n = 0, 0
        for carrier in carriers:
            full = chat_prompt(f'Copy this sentence exactly: "{carrier}"', ' "') + carrier + '."'
            bp = build_raw(tok, full)
            a, b = bp.find_span(carrier, occurrence=full.count(carrier) - 1)
            edits = make_condition_edits(bp, cond)
            lg = logits_with_edits(kit, bp.input_ids, edits)
            pred = lg.argmax(-1)
            tgt = bp.input_ids[0][1:]
            for p in range(a, b - 1):
                agree_tot += int(pred[p] == tgt[p])
                agree_n += 1
        row["copy_forced"] = {"n": agree_n, "ok": agree_tot}
        # wikitext agreement with clean model
        wa_tot, wa_n = 0, 0
        for w in wiki:
            bp = build_raw(tok, w[:400])
            clean_pred = logits_with_edits(kit, bp.input_ids, []).argmax(-1)
            edits = make_condition_edits(bp, cond)
            pred = logits_with_edits(kit, bp.input_ids, edits).argmax(-1)
            sl = slice(8, bp.n_tokens - 1)
            wa_tot += int((pred[sl] == clean_pred[sl]).sum())
            wa_n += pred[sl].shape[0]
        row["wikitext_agree"] = {"n": wa_n, "ok": wa_tot}
        results["tasks"][cond] = row
        print(cond, {k: f"{v['ok']}/{v['n']}" for k, v in row.items()})

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
