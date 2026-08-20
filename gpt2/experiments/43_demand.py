"""C2 demand variant: is a remembered word maintained in the lens during an
unrelated copying task ONLY when the (few-shot-established) format will ask
for it afterward? A variant of the paper's implicit-task-demand design, not
a replication: the paper's version never names the tracked property, while
here the word is named in BOTH conditions and only the format-implied later
question differs (day 2 found this decisive at 135M: median rank 6.5 vs 64).

The one-shot example establishes whether recall follows the copy. The word,
sentence, and copy span are identical across conditions; only the example's
ending differs. Lens rank of the word measured over the copy span, before
any recall could begin.

Run:  python experiments/43_demand.py [model]
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

# 3-shot (1-shot was too weak a format cue for base GPT-2);
# measured over the SECOND HALF of the copy span, where maintenance-vs-decay
# differentiates. Frame diversity: pools.DEMAND_VARIANTS supplies three
# shot-set variants (variant 0 = the original lamp/kite/moss set).
TAIL = 'Remember the word "{w}". Copy: "{s}" -> "{s_open}'
VARIANTS = [pools.demand_shots(v) for v in pools.DEMAND_VARIANTS]


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
    if not word_ids(w):
        continue  # single-token filter, as everywhere else
    for s in pools.C2_SENTENCES:
        for vi, (shots_d, shots_n) in enumerate(VARIANTS):
            rd = span_rank((shots_d + TAIL).format(w=w, s=s, s_open=s), s, w)
            rn = span_rank((shots_n + TAIL).format(w=w, s=s, s_open=s), s, w)
            rows.append(dict(word=w, sentence=s, variant=vi, demand=rd, nodemand=rn))
    print(".", end="", flush=True)
print()

n = len(rows)
med = statistics.median
print(f"n={n} (word x sentence x {len(VARIANTS)} shot variants): "
      f"word's best lens rank over the copy span")
print(f"  with recall demand:    median {med([r['demand'] for r in rows])}")
print(f"  without recall demand: median {med([r['nodemand'] for r in rows])}")
print(f"  demand < no-demand: {sum(r['demand'] < r['nodemand'] for r in rows)}/{n}")
for vi in range(len(VARIANTS)):
    sel = [r for r in rows if r["variant"] == vi]
    print(f"  variant {vi}: medians {med([r['demand'] for r in sel])} vs "
          f"{med([r['nodemand'] for r in sel])}, wins "
          f"{sum(r['demand'] < r['nodemand'] for r in sel)}/{len(sel)}")
json.dump(rows, open(f"/tiny-jlens/gpt2/results/c2_demand_{MODEL}.json", "w"))
