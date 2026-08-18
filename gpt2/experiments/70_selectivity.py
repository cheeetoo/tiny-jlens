"""C5 — selectivity: the same latent variable (the passage's language) is
causally read from the J-space by report/flexible tasks but not by automatic
continuation.

Phase same_latent:
  For each passage, three tasks over identical passage tokens:
    report        few-shot "Language:"            (explicit report)
    country       few-shot "Country ...:"         (flexible inference)
    continuation  free generation from the passage (automatic)
  A lens-coordinate swap replaces the passage's language with an alternative
  (French<->Spanish, German->Spanish, Italian->French) at post-passage
  positions (for continuation: the final 3 passage positions, where
  generation starts). Prediction: report/country flip to the alternative;
  continuation stays in the original language.
  Presence control: the language name's lens rank over passage positions is
  comparable in all conditions (the latent sits in the lens everywhere; only
  its USE differs).

Phase ablate:
  The C5b battery with the centered-gauge top-k ablation (see 71 notes in
  PLAN.md) — run separately once the battery is defined.

Run:  python experiments/70_selectivity.py same_latent [model]
"""

import json
import sys

sys.path.insert(0, "/tiny-jlens/gpt2")

import torch

import core
import pools

PHASE = sys.argv[1] if len(sys.argv) > 1 else "same_latent"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "gpt2"
kit = core.Kit(MODEL)
tok = kit.tokenizer

ALT = {"French": "Spanish", "Spanish": "French",
       "German": "Spanish", "Italian": "French"}
COUNTRY = {"French": "France", "German": "Germany",
           "Spanish": "Spain", "Italian": "Italy"}
ARTICLES = {"a", "an", "the", "called", '"', "'", ""}


def variant_ids(word: str) -> list[int]:
    out = []
    for w in {word, word.capitalize(), word.lower(), word.upper()}:
        for form in (" " + w, w):
            ids = tok(form, add_special_tokens=False)["input_ids"]
            if len(ids) == 1 and ids[0] not in out:
                out.append(ids[0])
    return out


@torch.no_grad()
def graded_top1(ids, edits=()):
    edits = list(edits)
    for _ in range(3):
        lg = core.logits_with(kit, ids, edits)[-1]
        t = int(lg.argmax())
        if tok.decode([t]).strip().lower() in ARTICLES:
            ids = torch.cat([ids, torch.tensor([[t]], device=ids.device)], dim=1)
            continue
        return t
    return t


def passage_token_count(template: str, passage: str) -> int:
    """Number of tokens up to and including the passage inside the rendered
    prompt (everything after is 'question tokens')."""
    text = template.format(passage=passage)
    cut = text.index(passage) + len(passage)
    return len(tok(text[:cut], add_special_tokens=False)["input_ids"])


if PHASE == "same_latent":
    lays = [l for l in kit.layers if 0.55 * kit.n_layers <= l <= 1.0 * kit.n_layers]
    passages = pools.c5_passages() + pools.EXTRA_PASSAGES
    rows = []
    for p in passages:
        lang, alt = p["category"], ALT[p["category"]]
        src, tgt = [kit.tok_id(" " + lang)], [kit.tok_id(" " + alt)]
        rec = dict(key=p["key"], lang=lang, alt=alt, tasks={})

        # ---- report & country (question tokens carry the swap)
        for task, tmpl, true_ans, alt_ans in [
            ("report", pools.LANG_REPORT_FEWSHOT, lang, alt),
            ("country", pools.LANG_COUNTRY_FEWSHOT, COUNTRY[lang], COUNTRY[alt]),
        ]:
            text = tmpl.format(passage=p["text"])
            ids = kit.encode(text)
            n_pass = passage_token_count(tmpl, p["text"])
            qpos = list(range(n_pass, ids.shape[1]))
            t_clean = graded_top1(ids)
            edits = core.swap_clamped(kit, ids, lays, src, tgt, alpha=1.0,
                                      positions=qpos, centered=True)
            t_swap = graded_top1(ids, edits)
            # presence: language-name lens rank over the last 10 passage
            # positions, best over layers (measured on this condition's prompt)
            resid = kit.residuals(ids, lays)
            span = list(range(max(0, n_pass - 10), n_pass))
            pres = min(int(kit.ranks(resid[l][span], l, variant_ids(lang)).min())
                       for l in lays)
            rec["tasks"][task] = dict(
                clean_ok=t_clean in variant_ids(true_ans),
                clean_got=tok.decode([t_clean]),
                swap_flipped=t_swap in variant_ids(alt_ans),
                swap_got=tok.decode([t_swap]), presence_rank=pres)

        # ---- continuation (swap at the last 3 passage positions)
        ids = kit.encode(p["text"])
        n = ids.shape[1]
        cont_clean = core.generate_with(kit, ids, [], max_new_tokens=24)
        edits = core.swap_clamped(kit, ids, lays, src, tgt, alpha=1.0,
                                  positions=list(range(n - 3, n)), centered=True)
        cont_swap = core.generate_with(kit, ids, edits, max_new_tokens=24)
        resid = kit.residuals(ids, lays)
        span = list(range(max(0, n - 10), n))
        pres = min(int(kit.ranks(resid[l][span], l, variant_ids(lang)).min())
                   for l in lays)
        rec["tasks"]["continuation"] = dict(
            clean_lang=pools.classify_language(cont_clean),
            swap_lang=pools.classify_language(cont_swap),
            clean=cont_clean, swap=cont_swap, presence_rank=pres)
        rows.append(rec)
        t = rec["tasks"]
        print(f"{p['key']:4s} {lang:8s} presence r/c/cont "
              f"{t['report']['presence_rank']:>3}/{t['country']['presence_rank']:>3}/"
              f"{t['continuation']['presence_rank']:>3}  "
              f"report {'ok' if t['report']['clean_ok'] else '--'}"
              f"->{'FLIP' if t['report']['swap_flipped'] else t['report']['swap_got'].strip()}  "
              f"country {'ok' if t['country']['clean_ok'] else '--'}"
              f"->{'FLIP' if t['country']['swap_flipped'] else t['country']['swap_got'].strip()}  "
              f"cont {t['continuation']['clean_lang'][:2]}->{t['continuation']['swap_lang'][:2]}")

    # aggregate over capability-passing trials per task
    rep = [r for r in rows if r["tasks"]["report"]["clean_ok"]]
    ctry = [r for r in rows if r["tasks"]["country"]["clean_ok"]]
    cont = [r for r in rows if r["tasks"]["continuation"]["clean_lang"] == r["lang"]]
    n_flex = sum(r["tasks"]["report"]["swap_flipped"] for r in rep) + \
        sum(r["tasks"]["country"]["swap_flipped"] for r in ctry)
    print(f"\nflexible follows swap: {n_flex}/{len(rep) + len(ctry)} "
          f"(report {sum(r['tasks']['report']['swap_flipped'] for r in rep)}/{len(rep)}, "
          f"country {sum(r['tasks']['country']['swap_flipped'] for r in ctry)}/{len(ctry)})")
    changed = [r for r in cont if r["tasks"]["continuation"]["swap_lang"] != r["lang"]]
    to_alt = [r for r in changed
              if r["tasks"]["continuation"]["swap_lang"] == r["alt"]]
    print(f"automatic continuation changed: {len(changed)}/{len(cont)} "
          f"(true redirects to alt language: {len(to_alt)}; "
          f"other changes: {[r['tasks']['continuation']['swap_lang'] for r in changed if r not in to_alt]})")
    pres_all = [r["tasks"][t]["presence_rank"] for r in rows
                for t in ("report", "country", "continuation")]
    print(f"presence (language name in lens, best rank over last-10 passage "
          f"positions): median {sorted(pres_all)[len(pres_all)//2]}")
    json.dump(rows, open(f"/tiny-jlens/gpt2/results/c5a_{MODEL}.json", "w"))
