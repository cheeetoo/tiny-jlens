"""C2 — directed modulation, rank-sensitive form: does an instruction about
what to think about move the word's lens rank at unrelated transcription
tokens?

Protocol (the form that worked on base GPT-2 in the day-2 work, itself
mirroring the user's LW study): the model transcribes a fixed sentence;
before the transcription, an instruction says to think about / not think
about an unrelated word. We read the word's J-lens rank at every
transcription token, at every layer. Conditions: think / dont / base.

Controls:
  - paired sign tests think<base (the effect), think<dont (content
    sensitivity), dont<base (white-bear: suppression incomplete);
  - blurt: the word's rank in the model's ACTUAL output distribution at the
    same positions — the claim is held-not-spoken, so trials where the model
    is about to say the word are flagged and the ordering is re-checked
    without them.

Run:  python experiments/40_modulation.py [gpt2|gpt2-medium|gpt2-large]
"""

import json
import statistics
import sys

sys.path.insert(0, "/tiny-jlens/gpt2")

import torch

import core
import pools

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
kit = core.Kit(MODEL)
tok = kit.tokenizer


def variant_ids(word: str) -> list[int]:
    out = []
    for w in {word, word.capitalize()}:
        for form in (" " + w, w):
            ids = tok(form, add_special_tokens=False)["input_ids"]
            if len(ids) == 1 and ids[0] not in out:
                out.append(ids[0])
    return out


def transcription_span(text: str, sentence: str) -> tuple[torch.Tensor, list[int]]:
    """Token ids of the full prompt + token indices of the SECOND occurrence
    of the sentence (the transcription)."""
    start = text.find(sentence, text.find(sentence) + 1)
    assert start > 0, "sentence must occur twice"
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
    ids = enc["input_ids"].cuda()
    span = [i for i, (a, b) in enumerate(enc["offset_mapping"][0].tolist())
            if a >= start and b <= start + len(sentence) and b > a]
    return ids, span


# Frame diversity: every phrasing in pools.C2_PHRASINGS runs for each
# (word, sentence); phrasing 0 of each condition is the canonical template.
# Rows are flat: one per (word, sentence, cond, phrasing).
rows = []
for w in pools.C2_WORDS:
    vids = variant_ids(w)
    if not vids:
        continue
    for s in pools.C2_SENTENCES:
        s_open = s  # transcription without a closing quote
        for cond, phrasings in pools.C2_PHRASINGS.items():
            for pi, tmpl in enumerate(phrasings):
                text = tmpl.format(s=s, w=w, s_open=s_open)
                ids, span = transcription_span(text, s_open)
                resid = kit.residuals(ids)
                per_layer = {l: int(kit.ranks(resid[l][span], l, vids).min())
                             for l in kit.layers}
                out_lg = kit.model_logits(ids)[span]
                blurt_rank = min(int((out_lg[i] > out_lg[i, v]).sum())
                                 for i in range(len(span)) for v in vids)
                rows.append(dict(word=w, sentence=s, cond=cond, phrasing=pi,
                                 best=min(per_layer.values()),
                                 per_layer=per_layer, blurt_rank=blurt_rank))
    print(".", end="", flush=True)
print()

base = {(r["word"], r["sentence"]): r for r in rows if r["cond"] == "base"}
med = statistics.median
n_pairs = len(base)
print(f"n={n_pairs} (word x sentence) trials x "
      f"{len(pools.C2_PHRASINGS['think'])} think / "
      f"{len(pools.C2_PHRASINGS['dont'])} dont phrasings")
for cond in ("think", "dont"):
    sel = [r for r in rows if r["cond"] == cond]
    print(f"{cond}: pooled median best rank {med([r['best'] for r in sel])} "
          f"(base {med([r['best'] for r in base.values()])})")
    for pi in sorted({r["phrasing"] for r in sel}):
        ph = [r for r in sel if r["phrasing"] == pi]
        wins = sum(r["best"] < base[(r["word"], r["sentence"])]["best"] for r in ph)
        print(f"  phrasing {pi}: median {med([r['best'] for r in ph]):>5}   "
              f"beats base {wins}/{len(ph)}")
think0 = {(r["word"], r["sentence"]): r for r in rows
          if r["cond"] == "think" and r["phrasing"] == 0}
blurts = sum(r["blurt_rank"] < 10 for r in think0.values())
print(f"blurt (canonical think phrasing): {blurts}/{len(think0)}")

json.dump(rows, open(f"/tiny-jlens/gpt2/results/c2_{MODEL}.json", "w"))
print(f"saved results/c2_{MODEL}.json")
