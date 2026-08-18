"""C2 demand variant: is a remembered word maintained in the lens during an
unrelated copying task ONLY when the (few-shot-established) format will ask
for it afterward? The paper's implicit-task-demand experiment, few-shot form
(day 2 found this decisive at 135M: median rank 6.5 vs 64).

The one-shot example establishes whether recall follows the copy. The word,
sentence, and copy span are identical across conditions; only the example's
ending differs. Lens rank of the word measured over the copy span, before
any recall could begin.

Run:  python experiments/43_demand.py [model]
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

# 3-shot (1-shot was too weak a format cue for base GPT-2);
# measured over the SECOND HALF of the copy span, where maintenance-vs-decay
# differentiates.
SHOTS_D = ('Remember the word "lamp". Copy: "The road was wet." -> "The road was wet." The word was "lamp".\n'
           'Remember the word "kite". Copy: "Dinner is at six." -> "Dinner is at six." The word was "kite".\n'
           'Remember the word "moss". Copy: "He lost his keys." -> "He lost his keys." The word was "moss".\n')
SHOTS_N = SHOTS_D.replace(' The word was "lamp".', ' Done.').replace(
    ' The word was "kite".', ' Done.').replace(' The word was "moss".', ' Done.')
TAIL = 'Remember the word "{w}". Copy: "{s}" -> "{s_open}'
DEMAND = SHOTS_D + TAIL
NODEMAND = SHOTS_N + TAIL


def word_ids(w):
    out = []
    for x in {w, w.capitalize()}:
        for f in (" " + x, x, ' "' + x):
            ids = tok(f, add_special_tokens=False)["input_ids"]
            if len(ids) == 1 and ids[0] not in out:
                out.append(ids[0])
    return out


def span_rank(text, sentence, w):
    start = text.rfind(sentence)  # the copy occurrence
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
    ids = enc["input_ids"].cuda()
    span = [i for i, (a, b) in enumerate(enc["offset_mapping"][0].tolist())
            if a >= start and b <= start + len(sentence) and b > a]
    span = span[len(span) // 2:]  # second half
    resid = kit.residuals(ids)
    return min(int(kit.ranks(resid[l][span], l, word_ids(w)).min())
               for l in kit.layers)


rows = []
for w in pools.C2_WORDS:
    for s in pools.C2_SENTENCES:
        rd = span_rank(DEMAND.format(w=w, s=s, s_open=s), s, w)
        rn = span_rank(NODEMAND.format(w=w, s=s, s_open=s), s, w)
        rows.append(dict(word=w, sentence=s, demand=rd, nodemand=rn))

n = len(rows)
med = lambda k: sorted(r[k] for r in rows)[n // 2]
wins = sum(r["demand"] < r["nodemand"] for r in rows)
print(f"n={n}: word's best lens rank over the copy span")
print(f"  with recall demand:    median {med('demand')}")
print(f"  without recall demand: median {med('nodemand')}")
print(f"  demand < no-demand: {wins}/{n}")
json.dump(rows, open(f"/tiny-jlens/gpt2/results/c2_demand_{MODEL}.json", "w"))
