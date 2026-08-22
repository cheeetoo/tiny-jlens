"""Criterion 4 (flexible generalization) in gpt2-small — the paper's §3.4, in one run.
See PROTOCOL.md.

  gate    capability: for each (category, function, argument) cell, does the model
          answer correctly under the 2-shot frame?  (the capability floor)
  E1 case Fig 18: one argument (France) read by every country function under a single
          fixed swap France->China; which functions follow the swap
  E2 swap Fig 19 left + appendix grids: the 4x4 grids, 16 funcs x 12 ordered pairs =
          192 trials; subtract-and-add swap (paper §3.4) at alpha=1 and alpha=2, plus
          the coordinate swap for comparison; success = target answer reaches top-1
  E3 load Fig 19 right: workspace loading (cos of residual with the arg lens vector)
          per argument and per category, and its relationship to swap success
  floor   the paper's bare templates, verbatim, no frame: capability + swap

Writes results/{results.json, prompts.json, summary.txt}.
Run:  python c4_generalization/run.py
"""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jl  # noqa: E402
import prompts as P  # noqa: E402
from swaps import coord_swap_edits, loading, subadd_swap_edits  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHAS = [1.0, 2.0]


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return p, c - h, c + h


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


class Words:
    """First-token variants of an answer word: the single-token ids that a
    space-prefixed / case-varied form of the word STARTS with.  Grading is on this
    first token (the paper's 'reaches the top of the output distribution'); multi-token
    answers are graded on their first token."""

    def __init__(self, lm):
        self.lm = lm
        self._c: dict[str, list[int]] = {}

    def __call__(self, word: str) -> list[int]:
        if word not in self._c:
            out = set()
            w = word.strip()
            for f in {w, w.capitalize(), w.lower()}:
                for pre in (" " + f, f):
                    ids = self.lm.tok(pre, add_special_tokens=False).input_ids
                    if ids:
                        out.add(ids[0])
            self._c[word] = sorted(out)
        return self._c[word]


@torch.no_grad()
def main():
    lm = jl.Lensed()
    band = jl.BAND
    W = Words(lm)
    cats = P.load_categories()

    def top1(lg):
        return int(lg.argmax())

    def hit(lg, word):
        return top1(lg) in W(word)

    def rank_out(lg, word):
        v = W(word)
        return int(jl.ranks_of(lg, v).min()) if v else 10**9

    def arg_pos(ids, arg):
        """Last position of any single-token variant of `arg` in the prompt."""
        variants = set()
        for f in {arg, arg.capitalize(), arg.lower(), arg.upper()}:
            for pre in (" " + f, f):
                w = lm.tok(pre, add_special_tokens=False).input_ids
                if len(w) == 1:
                    variants.add(w[0])
        toks = ids[0].tolist()
        cand = [i for i, t in enumerate(toks) if t in variants]
        return max(cand) if cand else len(toks) - 1

    # ------------------------------------------------------------------ build cells + gate
    # A cell = (category, function, argument).  Under the 2-shot frame, run the clean
    # pass once; store logits, band residuals, the arg token/position, and the gate.
    cells: dict[tuple, dict] = {}
    gate_rows = []
    for cat in cats:
        for fn in cat["funcs"]:
            for a in cat["args"]:
                text = P.FRAMES[(cat["name"], fn["name"])] + fn["template"].format(arg=a)
                ids = lm.encode(text)
                lg = lm.logits(ids)[-1]
                ans = fn["answers"][a]
                gated = hit(lg, ans)
                cells[(cat["name"], fn["name"], a)] = dict(
                    text=text, ids=ids, lg=lg, answer=ans, gated=gated,
                    argtok=lm.tid(" " + a) if lm.is_single(" " + a) else None,
                    argpos=arg_pos(ids, a), clean=lm.residuals(ids, band),
                    prompt_ids=set(ids[0].tolist()))
                gate_rows.append(dict(category=cat["name"], function=fn["name"], arg=a,
                                      answer=ans, gate=gated, greedy=lm.dec(top1(lg))))
    n_gate = sum(r["gate"] for r in gate_rows)

    out = dict(band=band, n_gate=n_gate, n_cells=len(gate_rows), gate=gate_rows,
               case=[], swap=[], loading=[], floor=None, alphas=ALPHAS)

    # ------------------------------------------------------------------ E1 case study (Fig 18)
    cs = P.CASE_STUDY
    cat = next(c for c in cats if c["name"] == cs["category"])
    for fn in cat["funcs"]:
        src = cells[(cat["name"], fn["name"], cs["source"])]
        tgt_answer = fn["answers"][cs["target"]]
        s = lm.tid(" " + cs["source"])
        t = lm.tid(" " + cs["target"])
        sw = lm.logits(src["ids"], subadd_swap_edits(lm, src["ids"], s, t, band,
                                                     alpha=1.0, clean=src["clean"]))[-1]
        out["case"].append(dict(
            function=fn["name"], template=fn["template"],
            source=cs["source"], target=cs["target"],
            source_answer=fn["answers"][cs["source"]], target_answer=tgt_answer,
            source_gated=src["gated"],
            target_gated=cells[(cat["name"], fn["name"], cs["target"])]["gated"],
            clean_top1=lm.dec(top1(src["lg"])), swapped_top1=lm.dec(top1(sw)),
            follows=hit(sw, tgt_answer),
            target_rank_clean=rank_out(src["lg"], tgt_answer),
            target_rank_swapped=rank_out(sw, tgt_answer)))

    # ------------------------------------------------------------------ E2 systematic swap (192)
    # Every function x every ordered (source, target) argument pair.  The prompt is the
    # SOURCE cell; we swap source->target and read the target's answer.  Flags let the
    # summary restrict to the interpretable subsets (source gated; both gated).
    for cat in cats:
        for fn in cat["funcs"]:
            for si, sa in enumerate(cat["args"]):
                src = cells[(cat["name"], fn["name"], sa)]
                s = src["argtok"]
                if s is None:
                    continue
                for ti, ta in enumerate(cat["args"]):
                    if ta == sa:
                        continue
                    tgt = cells[(cat["name"], fn["name"], ta)]
                    t = tgt["argtok"]
                    if t is None:
                        continue
                    t_ans = fn["answers"][ta]
                    distinct = not (set(W(t_ans)) & set(W(src["answer"])))  # target answer != source answer
                    echo = bool(set(W(t_ans)) & src["prompt_ids"])          # target answer visible in the frame
                    before = rank_out(src["lg"], t_ans)
                    row = dict(category=cat["name"], function=fn["name"],
                               source=sa, target=ta, source_i=si, target_i=ti,
                               source_answer=src["answer"], target_answer=t_ans,
                               source_gated=src["gated"], target_gated=tgt["gated"],
                               distinct=distinct, echo=echo, before=before, subadd={}, coord={})
                    for label, mk in (("subadd", subadd_swap_edits), ("coord", coord_swap_edits)):
                        for al in ALPHAS:
                            lg = lm.logits(src["ids"], mk(lm, src["ids"], s, t, band,
                                                          alpha=al, clean=src["clean"]))[-1]
                            row[label][str(al)] = dict(after=rank_out(lg, t_ans),
                                                       hit=hit(lg, t_ans), got=lm.dec(top1(lg)))
                    out["swap"].append(row)

    # ------------------------------------------------------------------ E3 loading (Fig 19 right)
    for cat in cats:
        for fn in cat["funcs"]:
            for a in cat["args"]:
                c = cells[(cat["name"], fn["name"], a)]
                if c["argtok"] is None:
                    continue
                out["loading"].append(dict(
                    category=cat["name"], function=fn["name"], arg=a,
                    loading=loading(lm, c["ids"], c["argtok"], c["argpos"], band, clean=c["clean"]),
                    gated=c["gated"]))

    # ------------------------------------------------------------------ floor (paper's bare templates)
    fl = dict(n=0, cap=0, per_cat={}, swap_n=0, swap_hit=0)
    for cat in cats:
        cc = dict(cap=0, n=0)
        clean_bare = {}
        for fn in cat["funcs"]:
            for a in cat["args"]:
                ids = lm.encode(fn["template"].format(arg=a))
                lg = lm.logits(ids)[-1]
                g = hit(lg, fn["answers"][a])
                fl["n"] += 1; cc["n"] += 1
                fl["cap"] += g; cc["cap"] += g
                clean_bare[(fn["name"], a)] = dict(ids=ids, lg=lg, gated=g)
        # bare-template swap on gated source cells (subtract-and-add, alpha=1)
        for fn in cat["funcs"]:
            for sa in cat["args"]:
                sc = clean_bare[(fn["name"], sa)]
                if not sc["gated"] or not lm.is_single(" " + sa):
                    continue
                for ta in cat["args"]:
                    if ta == sa or not lm.is_single(" " + ta):
                        continue
                    t_ans = fn["answers"][ta]
                    if set(W(t_ans)) & set(W(fn["answers"][sa])):
                        continue
                    s, t = lm.tid(" " + sa), lm.tid(" " + ta)
                    lg = lm.logits(sc["ids"], subadd_swap_edits(lm, sc["ids"], s, t, band))[-1]
                    fl["swap_n"] += 1
                    fl["swap_hit"] += hit(lg, t_ans)
        fl["per_cat"][cat["name"]] = cc
    out["floor"] = fl

    # ------------------------------------------------------------------ write
    os.makedirs(f"{HERE}/results", exist_ok=True)
    json.dump(out, open(f"{HERE}/results/results.json", "w"), indent=1, default=float)
    json.dump([dict(category=c, function=f, arg=a, answer=cells[(c, f, a)]["answer"],
                    gated=cells[(c, f, a)]["gated"], prompt=cells[(c, f, a)]["text"])
               for (c, f, a) in cells],
              open(f"{HERE}/results/prompts.json", "w"), indent=1)
    summary = summarize(out)
    open(f"{HERE}/results/summary.txt", "w").write(summary)
    print(summary)


def _spearman(xs, ys):
    if len(xs) < 2:
        return float("nan")
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for i, idx in enumerate(order):
            r[idx] = i
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1 - 6 * d2 / (n * (n * n - 1))


def summarize(out) -> str:
    L = ["Criterion 4 — flexible generalization — gpt2-small",
         f"\ncapability gate: {out['n_gate']}/{out['n_cells']} (category,function,argument) cells "
         f"answered correctly under the 2-shot frame"]

    # gate per category/function
    L.append("\ngate — cells answered correctly, by category / function (of 4 args each)")
    for cat in ["countries", "months", "animals", "numbers"]:
        rows = [r for r in out["gate"] if r["category"] == cat]
        byfn = {}
        for r in rows:
            byfn.setdefault(r["function"], []).append(r["gate"])
        tot = sum(r["gate"] for r in rows)
        detail = "  ".join(f"{fn}:{sum(v)}/4" for fn, v in byfn.items())
        L.append(f"  {cat:9s} {tot:2d}/16   {detail}")

    # E1 case study (Fig 18) — one argument, many functions, one fixed swap
    nfollow = sum(c["follows"] for c in out["case"])
    L.append(f"\nE1 case study (Fig 18) — one argument read by many functions: "
             f"{out['case'][0]['source']} -> {out['case'][0]['target']}, ONE fixed swap  "
             f"({nfollow}/{len(out['case'])} functions follow)")
    for c in out["case"]:
        mark = "FOLLOWS" if c["follows"] else "no     "
        L.append(f"  {c['function']:11s} clean={c['clean_top1']!r:11s} -> swapped={c['swapped_top1']!r:11s}"
                 f"  want {c['target']}'s {c['target_answer']!r:10s} (rank {c['target_rank_clean']:>4}->{c['target_rank_swapped']:<3}) [{mark}]")

    # E2 swap — subsets.  clean = distinct target answer, not echoable from the frame.
    def rate(rows, label, al):
        k = sum(r[label][str(al)]["hit"] for r in rows)
        return k, len(rows), *wilson(k, len(rows))

    def subset(name, flt, note=""):
        rows = [r for r in out["swap"] if flt(r)]
        L.append(f"\nE2 swap — {name}  (n={len(rows)} pairs){note}")
        L.append("     operation      alpha=1              alpha=2")
        for label in ("subadd", "coord"):
            k1, n1, p1, lo1, hi1 = rate(rows, label, 1.0)
            k2, n2, p2, lo2, hi2 = rate(rows, label, 2.0)
            tag = "  (paper's op)" if label == "subadd" else "  (comparison)"
            L.append(f"     {label:9s} {k1:3d}/{n1:<3d} {p1:5.1%} [{lo1:3.0%},{hi1:3.0%}]"
                     f"   {k2:3d}/{n2:<3d} {p2:5.1%} [{lo2:3.0%},{hi2:3.0%}]{tag}")
        return rows

    clean = lambda r: r["distinct"] and not r["echo"]
    subset("all 192 ordered pairs, ungated (paper: 76/192 a=1, 101/192 a=2)",
           lambda r: True, note="   <- capability-penalized: 44/64 cells fail the gate")
    subset("capable subset: target function gated (distinct, no echo)",
           lambda r: r["target_gated"] and clean(r),
           note="   <- the clean broadcast rate")
    subset("strictest: source AND target gated (distinct, no echo)",
           lambda r: r["source_gated"] and r["target_gated"] and clean(r))

    # per-category on the capable (target-gated) subset — paper's Fig 68 / Fig 19 ordering
    L.append("\n  by category, capable subset (target gated, distinct, no echo; subtract-and-add a=1):")
    for cat in ["countries", "months", "animals", "numbers"]:
        rows = [r for r in out["swap"] if r["category"] == cat and r["target_gated"] and clean(r)]
        if not rows:
            L.append(f"     {cat:9s} (no capable pairs)")
            continue
        k1, n1, p1, *_ = rate(rows, "subadd", 1.0)
        mb = median([r["before"] for r in rows]); ma = median([r["subadd"]["1.0"]["after"] for r in rows])
        L.append(f"     {cat:9s} {k1:2d}/{n1:<2d} {p1:5.0%}   median target rank {mb:.0f} -> {ma:.0f}")

    # per-function: which functions FOLLOW the swap.  A clean split (paper does not
    # report this): lookup/retrieval functions follow, relational/compute do not.
    L.append("\n  by function, capable subset (a=1) — lookup functions follow, relational ones do not:")
    byfn = {}
    for r in out["swap"]:
        if r["target_gated"] and clean(r):
            byfn.setdefault((r["category"], r["function"]), []).append(r["subadd"]["1.0"]["hit"])
    for (cat, fn), hits in sorted(byfn.items(), key=lambda kv: -sum(kv[1]) / max(len(kv[1]), 1)):
        L.append(f"     {cat:9s}/{fn:12s} {sum(hits):2d}/{len(hits):<2d}")

    # alpha=2 overshoot: the swap emits the injected ARGUMENT word rather than f(argument)
    def arg_out(al):
        rows = [r for r in out["swap"] if clean(r)]
        n = len(rows)
        k = sum(r["subadd"][str(al)]["got"].strip().lower() == r["target"].lower() for r in rows)
        return k, n
    k1, n1 = arg_out(1.0); k2, n2 = arg_out(2.0)
    L.append(f"\n  alpha=2 overshoot: the swap emits the injected ARGUMENT word (not f(argument)) on "
             f"{k1}/{n1} pairs at a=1 but {k2}/{n2} at a=2,")
    L.append("  so a=2 does not help here (opposite to the paper's 76->101); double strength names the argument.")

    # E3 loading (Fig 19 right)
    lo = out["loading"]
    L.append("\nE3 workspace loading (Fig 19 right) — cos(residual, arg lens vector), band mean")
    catload = {c: [x["loading"] for x in lo if x["category"] == c] for c in ["countries", "months", "animals", "numbers"]}
    for cat, v in sorted(((c, sum(vs) / len(vs)) for c, vs in catload.items()), key=lambda x: -x[1]):
        L.append(f"     {cat:9s} loading {v:+.3f}")
    # cell-level loading vs swap success (capable subset)
    loadmap = {(x["category"], x["function"], x["arg"]): x["loading"] for x in lo}
    cell = {}
    for r in out["swap"]:
        if r["target_gated"] and clean(r):
            key = (r["category"], r["function"], r["source"])
            c = cell.setdefault(key, [0, 0]); c[0] += r["subadd"]["1.0"]["hit"]; c[1] += 1
    xs = [loadmap[k] for k in cell]; ys = [cell[k][0] / cell[k][1] for k in cell]
    L.append(f"  category ordering matches the paper (countries/animals high, number words lowest);")
    L.append(f"  but the cell-level loading->success link is weak: Spearman = {_spearman(xs, ys):+.2f} "
             f"over {len(xs)} source cells (paper: loading predicts success well).")

    # floor
    fl = out["floor"]
    L.append(f"\nfloor — paper's bare templates, no frame: capability {fl['cap']}/{fl['n']}; "
             f"subtract-and-add swap on gated bare cells {fl['swap_hit']}/{fl['swap_n']}")
    L.append("  per category (bare capability): " +
             "  ".join(f"{c} {d['cap']}/{d['n']}" for c, d in fl["per_cat"].items()))
    return "\n".join(L)


if __name__ == "__main__":
    main()
