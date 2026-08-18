"""C4 — flexible generalization (broadcast): one and the same lens-coordinate
swap (argument A -> B), applied identically across prompts that each apply a
DIFFERENT function to the argument, redirects each function's output.

Success = the graded top-1 becomes func(B). Also records, per (A, B) pair,
how many of the functions the identical swap redirects (the broadcast
signature), and the source argument's workspace loading (cosine of the
residual with A's centered lens vector) as a success predictor.

Run:  python experiments/60_flexibility.py [model]
"""

import itertools
import json
import sys

sys.path.insert(0, "/tiny-jlens/gpt2")

import torch

import core
import pools

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
kit = core.Kit(MODEL)
tok = kit.tokenizer
LAYS = [l for l in kit.layers if 0.55 * kit.n_layers <= l <= 1.0 * kit.n_layers]
ARTICLES = {"a", "an", "the", "called", '"', "'", ""}


def variant_ids(word):
    out = set()
    for w in {word, word.capitalize(), word.lower(), word.upper()}:
        for form in (" " + w, w):
            ids = tok(form, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                out.add(ids[0])
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
        return t, lg
    return t, lg


cap = json.load(open(f"/tiny-jlens/gpt2/results/capability_{MODEL}{pools.SUFFIX}.json"))
cells = []
for c in cap["c4"]:
    ok_args = {a for a, v in c["cell"].items() if v["ok"]}
    if len(ok_args) >= 3:
        grid_entry = next((g for g in pools.c4_grid(tok)
                           if g[0] == c["category"] and g[1] == c["func"]), None)
        if grid_entry:
            cells.append(dict(category=c["category"], func=c["func"],
                              template=grid_entry[2],
                              answers={a: v for a, v in grid_entry[3].items()
                                       if a in ok_args}))
print(f"{len(cells)} capability-passing cells: "
      f"{[(c['category'], c['func'], len(c['answers'])) for c in cells]}")

by_cat = {}
for c in cells:
    by_cat.setdefault(c["category"], []).append(c)

rows = []
for cat, cs in by_cat.items():
    args = sorted(set.intersection(*[set(c["answers"]) for c in cs]))
    pairs = [(a, b) for a, b in itertools.permutations(args, 2)
             if all(not (variant_ids(c["answers"][a]) & variant_ids(c["answers"][b]))
                    for c in cs)]
    for A, B in pairs:
        src, tgt = [kit.tok_id(" " + A)], [kit.tok_id(" " + B)]
        for c in cs:
            text = c["template"].format(arg=A)
            ids = kit.encode(text)
            clean_lg = kit.model_logits(ids)[-1]
            tgt_pre = min(int((clean_lg > clean_lg[v]).sum())
                          for v in variant_ids(c["answers"][B]))
            resid = kit.residuals(ids, LAYS)
            vA = kit.vectors(LAYS[len(LAYS) // 2], src, centered=True)[0]
            loading = float(torch.nn.functional.cosine_similarity(
                resid[LAYS[len(LAYS) // 2]], vA[None, :], dim=1).max())
            edits = core.swap_clamped(kit, ids, LAYS, src, tgt,
                                      alpha=1.0, centered=True)
            t, lg = graded_top1(ids, edits)
            rows.append(dict(category=cat, func=c["func"], A=A, B=B,
                             hit=t in variant_ids(c["answers"][B]),
                             got=tok.decode([t]), tgt_pre=tgt_pre,
                             loading=loading))
    print(f"  {cat}: {len(pairs)} pairs x {len(cs)} functions")

ok = [r for r in rows if r["tgt_pre"] >= 10]  # paper guard
print(f"\noverall (target not already top-10): "
      f"{sum(r['hit'] for r in ok)}/{len(ok)} "
      f"({100 * sum(r['hit'] for r in ok) / max(1, len(ok)):.0f}%)")
for cat, cs in by_cat.items():
    for c in cs:
        sel = [r for r in ok if r["category"] == cat and r["func"] == c["func"]]
        if sel:
            print(f"  {cat:10s} {c['func']:12s} {sum(r['hit'] for r in sel):2d}/{len(sel)}")

# broadcast: same (A,B) pair across functions
pair_stats = {}
for r in ok:
    pair_stats.setdefault((r["category"], r["A"], r["B"]), []).append(r["hit"])
multi = {k: v for k, v in pair_stats.items() if len(v) >= 2}
both = sum(all(v) for v in multi.values())
print(f"same-pair-swaps evaluated on >=2 functions: {len(multi)}; "
      f"redirect ALL functions: {both} ({100 * both / max(1, len(multi)):.0f}%)")

# loading predicts success?
if ok:
    hit_load = [r["loading"] for r in ok if r["hit"]]
    miss_load = [r["loading"] for r in ok if not r["hit"]]
    if hit_load and miss_load:
        print(f"mean loading: hits {sum(hit_load)/len(hit_load):.3f} "
              f"vs misses {sum(miss_load)/len(miss_load):.3f}")
json.dump(rows, open(f"/tiny-jlens/gpt2/results/c4_{MODEL}{pools.SUFFIX}.json", "w"))
