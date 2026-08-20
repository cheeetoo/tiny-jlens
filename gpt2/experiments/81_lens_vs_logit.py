"""Is the Jacobian transport doing the work? J-lens vs logit lens on the
two-hop readout: rank of the intermediate at the answer position,
J_l @ h vs h (identity transport), same unembedding. (Capability-filtered
items only — no unspokenness filter here, unlike 50_reasoning readout.)

Run:  python experiments/81_lens_vs_logit.py [model]
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


def variant_ids(word):
    out = set()
    for w in {word, word.capitalize(), word.lower(), word.upper()}:
        for form in (" " + w, w):
            ids = tok(form, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                out.add(ids[0])
    return out


cap = json.load(open(f"/tiny-jlens/gpt2/results/capability_{MODEL}.json"))
passed = {r["prompt"] for r in cap["twohop"] if r["twohop"]}
items = [it for it in pools.twohop_items(tok) if it.prompt in passed]

stats = {l: dict(j=[], logit=[]) for l in kit.layers}
for it in items:
    ids = kit.encode(it.prompt)
    resid = kit.residuals(ids)
    vids = list(variant_ids(it.intermediate))
    for l in kit.layers:
        h = resid[l][-1:]
        jr = int(kit.ranks(h, l, vids)[0].min())
        lg = kit.model_unembed_logits = kit.model.unembed(h).float()  # logit lens
        lr = int(min((lg[0] > lg[0, v]).sum() for v in vids))
        stats[l]["j"].append(jr)
        stats[l]["logit"].append(lr)

print(f"{MODEL}: intermediate rank at answer position, J-lens vs logit lens "
      f"(n={len(items)})")
print(f"{'L':>3} {'J med':>7} {'J<10%':>6} {'logit med':>10} {'logit<10%':>10}")
out = {}
for l in kit.layers:
    j, g = stats[l]["j"], stats[l]["logit"]
    med = statistics.median
    out[l] = dict(j_med=med(j), j_top10=sum(r < 10 for r in j) / len(j),
                  logit_med=med(g), logit_top10=sum(r < 10 for r in g) / len(g))
    print(f"{l:>3} {out[l]['j_med']:>7} {100*out[l]['j_top10']:>5.0f}% "
          f"{out[l]['logit_med']:>10} {100*out[l]['logit_top10']:>9.0f}%")
json.dump(out, open(f"/tiny-jlens/gpt2/results/lens_vs_logit_{MODEL}.json", "w"))
