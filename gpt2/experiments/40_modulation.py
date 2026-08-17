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


rows = []
for w in pools.C2_WORDS:
    vids = variant_ids(w)
    if not vids:
        continue
    for s in pools.C2_SENTENCES:
        s_open = s  # transcription without a closing quote
        rec = dict(word=w, sentence=s)
        for cond, tmpl in pools.C2_TEMPLATES.items():
            text = tmpl.format(s=s, w=w, s_open=s_open)
            ids, span = transcription_span(text, s_open)
            resid = kit.residuals(ids)
            per_layer = {l: int(kit.ranks(resid[l][span], l, vids).min())
                         for l in kit.layers}
            out_lg = kit.model_logits(ids)[span]
            blurt_rank = min(int((out_lg[i] > out_lg[i, v]).sum())
                             for i in range(len(span)) for v in vids)
            rec[cond] = dict(best=min(per_layer.values()), per_layer=per_layer,
                             blurt_rank=blurt_rank)
        rows.append(rec)

n = len(rows)
med = lambda c: sorted(r[c]["best"] for r in rows)[n // 2]
print(f"n={n} (word x sentence) trials")
print(f"median best lens rank:  think {med('think')}   dont {med('dont')}   base {med('base')}")
for a, b in [("think", "base"), ("think", "dont"), ("dont", "base")]:
    wins = sum(r[a]["best"] < r[b]["best"] for r in rows)
    print(f"  {a} < {b}:  {wins}/{n}")
blurts = [r for r in rows if r["think"]["blurt_rank"] < 10]
print(f"blurt (word in output top-10 somewhere in span, think cond): {len(blurts)}/{n}")
nb = [r for r in rows if r["think"]["blurt_rank"] >= 10]
if nb:
    m = lambda c: sorted(r[c]["best"] for r in nb)[len(nb) // 2]
    print(f"non-blurt trials only (n={len(nb)}): think {m('think')}  dont {m('dont')}  base {m('base')}")

print("\nper-layer median rank under think / base:")
for l in kit.layers:
    mt = sorted(r["think"]["per_layer"][l] for r in rows)[n // 2]
    mb = sorted(r["base"]["per_layer"][l] for r in rows)[n // 2]
    print(f"  L{l:2d}  think {mt:>6}   base {mb:>6}")

json.dump(rows, open(f"/tiny-jlens/gpt2/results/c2_{MODEL}.json", "w"))
print(f"saved results/c2_{MODEL}.json")
