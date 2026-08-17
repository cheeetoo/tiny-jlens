"""C5(a) — Selectivity, same-latent design (paper §3.5 language experiment).

One latent variable (the passage's language) needed by four tasks:
  report      "What language is this passage written in?"        [flexible: follows swap]
  country     "In which country is this language mainly spoken?" [flexible: follows swap]
  continue    "Continue it by writing the next sentence."        [automatic: unaffected]
  anomaly     "Does the passage switch language partway?"        [automatic: unaffected]

Swap: language-name lens coordinates (true -> alt) at the question tokens
(all tokens after the passage), band layers. Presence control: language name
in lens top-k over the passage+question in every condition.

Usage: python scripts/c5a_selectivity_language.py --lens runs/smollm2-135m-it/lens.pt
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
from tinyjlens.interventions import clamped_swap_edits, logits_with_edits, generate_with_edits

REF = os.path.join(os.path.dirname(__file__), "..", "ref", "jacobian-lens", "data")

LANG_INFO = {
    "French": {"country": "France", "alt": "Spanish", "chars": "éèêàçôû",
               "stop": [" le", " la", " les", " et", " de", " un", " une", " dans", " est", " il", " elle"]},
    "German": {"country": "Germany", "alt": "French", "chars": "äöüß",
               "stop": [" der", " die", " das", " und", " ist", " ein", " eine", " nicht", " mit", " sich"]},
    "Spanish": {"country": "Spain", "alt": "French", "chars": "ñáéíóú¿¡",
                "stop": [" el", " la", " los", " y", " de", " un", " una", " es", " en", " que"]},
    "Italian": {"country": "Italy", "alt": "Spanish", "chars": "àèéìòù",
                "stop": [" il", " la", " le", " e", " di", " un", " una", " è", " che", " per"]},
}


def guess_language(text: str) -> str | None:
    """Crude language guess for continuations: stopword + charset voting."""
    scores = {}
    t = " " + text.lower()
    for lang, info in LANG_INFO.items():
        s = sum(t.count(w + " ") + t.count(w + ",") + t.count(w + ".") for w in info["stop"])
        s += sum(2 for ch in info["chars"] if ch in text)
        scores[lang] = s
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return None
    # English detector: if English stopwords dominate, call it English
    eng = sum((" " + text.lower()).count(w) for w in [" the ", " and ", " of ", " is ", " to "])
    if eng > scores[best]:
        return "English"
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--lens", default="runs/smollm2-135m-it/lens.pt")
    ap.add_argument("--band", default=None)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--single-pair", action="store_true")
    ap.add_argument("--alt-map", default="")  # e.g. German:Spanish,Italian:French
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--out", default="runs/c5a_language.json")
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

    with open(os.path.join(REF, "experiments", "selectivity-language.json")) as f:
        sl = json.load(f)
    passages = sl["passages"]
    if args.confirm:
        from tinyjlens.confirm_pools import CONFIRM_PASSAGES
        passages = passages + CONFIRM_PASSAGES

    def render(passage: str, question: str, prefill: str) -> str:
        user = f"Here is a passage of text:\n\n{passage}\n\n{question}"
        return tok.apply_chat_template([{"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True) + prefill

    TASKS = {
        "report": ("What language is this passage written in? Answer with one word.",
                   " The language is"),
        "country": ("In which country is the language of this passage mainly spoken? Answer with one word.",
                    " The language of this passage is mainly spoken in"),
        "continue": ("Continue it by writing the next sentence. Output only the continuation.", ""),
        "anomaly": ("Does the passage switch to a different language partway through? Answer yes or no.",
                    " The answer is"),
    }

    def splice(text: str, alt_lang: str) -> str:
        """Replace a middle sentence with one from an alt-language passage."""
        donor = next(x["text"] for x in passages if x["category"] == alt_lang)
        donor_sents = [s for s in donor.split(". ") if len(s) > 20]
        sents = text.split(". ")
        if len(sents) < 2 or not donor_sents:
            return text + " " + donor_sents[0] + "."
        mid = len(sents) // 2
        sents[mid] = donor_sents[0]
        return ". ".join(sents)

    yes_ids = variant_token_ids(tok, "yes")
    no_ids = variant_token_ids(tok, "no")

    def yes_no_score(logits) -> float:
        return (max(float(logits[t]) for t in yes_ids)
                - max(float(logits[t]) for t in no_ids))

    results = {"band": [band[0], band[-1]], "alpha": args.alpha, "trials": []}
    for p in passages:
        lang = p["category"]
        info = LANG_INFO[lang]
        alt = info["alt"]
        if args.alt_map:
            overrides = dict(x.split(":") for x in args.alt_map.split(","))
            alt = overrides.get(lang, alt)
        src_ids = variant_token_ids(tok, lang)
        tgt_ids = variant_token_ids(tok, alt)
        m = 1 if args.single_pair else min(len(src_ids), len(tgt_ids))
        src_ids, tgt_ids = src_ids[:m], tgt_ids[:m]

        for task, (question, prefill) in TASKS.items():
            stim_text = p["text"] if task != "anomaly" else None
            text = render(p["text"], question, prefill)
            bp = build_raw(tok, text)
            # question span: from after the passage to the end
            passage_end = bp.find_span(p["text"][-40:], occurrence=0)[1]
            q_positions = list(range(passage_end, bp.n_tokens))
            # presence: language token rank over passage+question, band layers
            resid = kit.residuals(bp.input_ids, band)
            presence = min(int(kit.lens_ranks_of(resid[l], l, src_ids).min()) for l in band)

            edits = clamped_swap_edits(kit, bp.input_ids, band, src_ids, tgt_ids, args.alpha, positions=q_positions)
            if task == "anomaly":
                # spliced vs unspliced discrimination, invariance of the margin
                row = {"key": p["key"], "lang": lang, "task": task, "presence_rank": presence}
                for variant, vtext in (("clean_passage", p["text"]),
                                       ("spliced_passage", splice(p["text"], alt))):
                    t2 = render(vtext, question, prefill)
                    bp2 = build_raw(tok, t2)
                    pe = bp2.find_span(vtext[-40:], occurrence=0)[1]
                    qp2 = list(range(pe, bp2.n_tokens))
                    e2 = clamped_swap_edits(kit, bp2.input_ids, band, src_ids, tgt_ids, args.alpha, positions=qp2)
                    row[variant + "_score"] = yes_no_score(logits_with_edits(kit, bp2.input_ids, [])[-1])
                    row[variant + "_score_swap"] = yes_no_score(logits_with_edits(kit, bp2.input_ids, e2)[-1])
                row["discrimination"] = row["spliced_passage_score"] - row["clean_passage_score"]
                row["discrimination_swap"] = row["spliced_passage_score_swap"] - row["clean_passage_score_swap"]
                results["trials"].append(row)
            elif task in ("report", "country"):
                clean_l = logits_with_edits(kit, bp.input_ids, [])[-1]
                swap_l = logits_with_edits(kit, bp.input_ids, edits)[-1]
                clean_ans = tok.decode([int(clean_l.argmax())])
                swap_ans = tok.decode([int(swap_l.argmax())])
                if task == "report":
                    want_clean, want_swap = lang, alt
                else:
                    want_clean, want_swap = info["country"], LANG_INFO[alt]["country"]
                results["trials"].append({
                    "key": p["key"], "lang": lang, "task": task, "presence_rank": presence,
                    "clean": clean_ans, "swapped": swap_ans,
                    "clean_ok": want_clean.lower().startswith(clean_ans.strip().lower()) and len(clean_ans.strip()) >= 2,
                    "followed_swap": want_swap.lower().startswith(swap_ans.strip().lower()) and len(swap_ans.strip()) >= 2,
                    "changed": clean_ans != swap_ans})
            else:  # continue: generate under clean and swap
                ids = bp.input_ids
                clean_gen = tok.decode(generate_with_edits(kit, ids, [], max_new_tokens=24)[0], skip_special_tokens=True)
                swap_gen = tok.decode(generate_with_edits(kit, ids, edits, max_new_tokens=24)[0], skip_special_tokens=True)
                results["trials"].append({
                    "key": p["key"], "lang": lang, "task": task, "presence_rank": presence,
                    "clean": clean_gen[:80], "swapped": swap_gen[:80],
                    "clean_lang": guess_language(clean_gen), "swap_lang": guess_language(swap_gen),
                    "clean_ok": guess_language(clean_gen) == lang,
                    "unchanged_lang": guess_language(swap_gen) == lang})
            print(results["trials"][-1])

    # summary
    summ = {}
    for t in results["trials"]:
        if t["task"] == "anomaly":
            s = summ.setdefault("anomaly", {"n": 0, "n_discriminates": 0,
                                            "n_sign_invariant": 0, "presence_top3": 0})
            s["n"] += 1
            s["n_discriminates"] += t["discrimination"] > 0
            s["n_sign_invariant"] += (t["discrimination"] > 0) == (t["discrimination_swap"] > 0)
            s["presence_top3"] += t["presence_rank"] < 3
            continue
        s = summ.setdefault(t["task"], {"n": 0, "clean_ok": 0, "followed": 0, "unchanged": 0,
                                        "presence_top3": 0})
        s["n"] += 1
        s["clean_ok"] += t.get("clean_ok", False)
        s["followed"] += t.get("followed_swap", False)
        s["unchanged"] += t.get("unchanged_lang", t.get("clean") == t.get("swapped"))
        s["presence_top3"] += t["presence_rank"] < 3
    results["summary"] = summ
    print(json.dumps(summ, indent=1))
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
