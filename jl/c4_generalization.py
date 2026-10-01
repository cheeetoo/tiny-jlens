"""Criterion 4 (flexible generalization) in gpt2-small — the paper's §3.4, in one run.
See PROTOCOL.md.

  gate    capability: for each (category, function, argument) cell, does the model
          answer correctly under the 2-shot frame?  (the capability floor)
  E1 case Fig 18: one argument (France) read by every country function under a single
          fixed swap France->China; which functions follow the swap
  E2 swap Fig 19 left + appendix grids: the 4x4 grids, 16 funcs x 12 ordered pairs =
          192 trials; the lens-coordinate swap (the operation the paper's Fig. 68 names) and
          the subtract-and-add swap (§3.4's wording), each at alpha=1 and alpha=2; success =
          target answer reaches top-1.  Scored on the pairs a swap could change (GPT-2 knows
          the target's answer, which is not already its top-1), with Claude's result on the
          same pairs from the paper's released grid (ref/paper-data/flex-gen-appendix.json);
          also at top-5, where the grid's ranks give Claude at alpha=1
  E3 load Fig 19 right: workspace loading (cos of residual with the arg lens vector)
          per argument and per category, and its relationship to the swap effect (the
          paper's measure: change in the target answer's log-prob minus change in the
          spontaneous answer's) and to swap success, with Claude's from the paper's
          released Fig. 19 data (ref/paper-data/flex-gen-systematic.json)
  floor   the paper's bare templates, verbatim, no frame: capability + coordinate swap

Writes results/c4_generalization/{results.json, prompts.json, summary.txt}.
Run:  python -m jl.c4_generalization          (the criterion)
      python -m jl.c4_generalization gating   (E2 swap rates by gating class)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from statistics import mean

import torch

import jl
from jl import coord_swap_edits, loading, swap_edits
from jl.stats import pearson, spearman, wilson

ALPHAS = [1.0, 2.0]
OPS = {"coord": "coordinate swap (paper's Fig. 68)", "subadd": "subtract-and-add swap"}
PAPER_GRID = jl.model.ROOT / "ref/paper-data/flex-gen-appendix.json"
PAPER_SUMMARY = jl.model.ROOT / "ref/paper-data/flex-gen-systematic.json"


def key(r):
    """(category, function, source argument, target argument) of a swap row."""
    return (r["category"], r["function"], r["source"], r["target"])


def claude_ranks() -> dict:
    """{key: the target answer's rank after the swap in Claude Sonnet 4.5 (alpha = 1)}, from the
    paper's released Fig. 68 data."""
    out = {}
    for c in json.load(open(PAPER_GRID))["cats"]:
        for f in c["funcs"]:
            for i, s in enumerate(c["args"]):
                for j, t in enumerate(c["args"]):
                    if i != j:
                        out[(c["name"], f["name"], s, t)] = f["cells"][i][j]["rk"]
    return out


def claude_grid() -> dict:
    """{key: did the swap put the target answer at top-1 in Claude Sonnet 4.5 (alpha = 1)}."""
    return {k: rk == 1 for k, rk in claude_ranks().items()}


# =============================================================================== prompt material
# Prompt material for criterion 4 (flexible generalization): one argument, many
# functions, one fixed swap.
#
# The paper's §3.4 constructs, for each *category* of argument (countries, months,
# animals, number words), a set of *functions* that each apply a different operation
# to the same argument ("the capital of France is", "most people in France speak",
# "France is on the continent of", ...).  It then swaps the J-lens vector for the
# argument (France -> China) identically across every function and asks whether each
# downstream function reads the swapped-in argument correctly.  Systematically:
# 4 categories x 4 functions x 4 arguments; within a category the four arguments give
# 12 ordered source->target swap pairs per function, hence 16 functions x 12 = 192
# swap trials (paper: 76/192 at alpha=1, 101/192 at alpha=2).
#
# We use the paper's OWN released material for this experiment verbatim ---
# `ref/jacobian-lens/data/experiments/flexible-generalization.json` (the categories,
# arguments, function templates, and answers).  `load_categories()` returns it.
#
# Deviation (capability floor).  GPT-2-small answers only 9/64 of these bare
# templates correctly (see PROTOCOL.md / the `floor` block below), so --- exactly
# as criterion 1 wrapped "Name a {cat}:" in a few-shot list and criterion 3 wrapped
# its two-hop query in a few-shot frame --- each function's query is preceded by a
# 2-shot frame that teaches the *function* with two DEMO arguments disjoint from the
# four test arguments.  The frame lifts capability to 20/64; the swap and loading
# analyses run on top of it.  `FRAMES[(category, function)]` is the demo prefix; the
# query is `FRAMES[...] + template.format(arg=test_arg)`.  Prepending the frame also
# means the test argument is never sentence-initial, so it always tokenizes with a
# leading space (the paper ignores the capitalization-token positions; here they do
# not arise).
#
# The demo arguments and their answers are chosen disjoint from the test arguments;
# any residual echo (a target answer appearing in the frame) is dropped at scoring
# time by the echo guard below.  The paper's bare templates are run verbatim,
# with no frame, as the capability `floor` below.
REF_FILE = jl.REF_DATA / "flexible-generalization.json"


def load_categories() -> list[dict]:
    """The paper's §3.4 data, verbatim: 4 categories, each with `args` (4) and
    `funcs` (4), each func a dict(name, template, answers={arg: answer})."""
    return json.load(open(REF_FILE))["categories"]


# 2-shot demo prefixes, one per (category, function).  Demo arguments are disjoint
# from the four test arguments of each category; each shot uses the function's own
# template so the frame teaches exactly the function under test.  A trailing space
# joins the frame to the query, keeping the test argument non-initial (leading-space
# token).  These are the ONLY authored text in this criterion; everything else
# (arguments, templates, answers) is the paper's.
FRAMES: dict[tuple[str, str], str] = {
    # countries — demos: Japan, Italy, Australia, Brazil, Sweden (not France/Canada/China/Egypt)
    ("countries", "capital"):   "The capital of Japan is the city of Tokyo. The capital of Italy is the city of Rome. ",
    ("countries", "language"):  "Most people in Japan speak Japanese. Most people in Italy speak Italian. ",
    ("countries", "continent"): "Australia is a country on the continent of Oceania. Brazil is a country on the continent of South. ",
    ("countries", "currency"):  "The single-word name for the currency now used in Japan is the Yen. The single-word name for the currency now used in Sweden is the Krona. ",
    # months — demos: January, June, December, November, March (not February/April/July/October)
    ("months", "season"):       "In the northern hemisphere, January is in the season of winter. In the northern hemisphere, June is in the season of summer. ",
    ("months", "number"):       "Counting from January, March is month number three. Counting from January, June is month number six. ",
    ("months", "holiday"):      "The biggest holiday in December is Christmas. The biggest holiday in November is Thanksgiving. ",
    ("months", "next_month"):   "The month right after January is February. The month right after June is July. ",
    # animals — demos: camel, frog, ant, crab, dog, snake, wolf, fish (not lion/eagle/shark/spider)
    ("animals", "habitat"):     "The natural habitat of a camel is the desert. The natural habitat of a frog is the pond. ",
    ("animals", "legs"):        "How many legs does an ant have? In one word: six. How many legs does a crab have? In one word: ten. ",
    ("animals", "class"):       "Biologically, a dog is a type of mammal. Biologically, a snake is a type of reptile. ",
    ("animals", "group"):       "A group of wolves is called a pack. A group of fish is called a school. ",
    # numbers — demos: one, four, two, six, twelve, zero, eight (not three/five/seven/nine)
    ("numbers", "double"):      "Two times one equals two. Two times four equals eight. ",
    ("numbers", "square"):      "Two squared equals four. Six squared equals thirty-six. ",
    ("numbers", "successor"):   "The number that comes right after one is two. The number that comes right after twelve is thirteen. ",
    ("numbers", "first_letter"):"The word 'zero' begins with the letter z. The word 'eight' begins with the letter e. ",
}

# Fig 18 case study: one argument (France) read by every country function, under a
# single fixed swap France -> China.
CASE_STUDY = dict(category="countries", source="France", target="China")


# =============================================================================== the run


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
    cats = load_categories()

    def top1(lg):
        return int(lg.argmax())

    def hit(lg, word):
        return top1(lg) in W(word)

    def rank_out(lg, word):
        v = W(word)
        return int(jl.ranks_of(lg, v).min()) if v else 10**9

    def lp(lg, toks):
        return float(lg.log_softmax(-1)[toks].max())

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
                text = FRAMES[(cat["name"], fn["name"])] + fn["template"].format(arg=a)
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
    cs = CASE_STUDY
    cat = next(c for c in cats if c["name"] == cs["category"])
    for fn in cat["funcs"]:
        src = cells[(cat["name"], fn["name"], cs["source"])]
        tgt_answer = fn["answers"][cs["target"]]
        s = lm.tid(" " + cs["source"])
        t = lm.tid(" " + cs["target"])
        row = dict(function=fn["name"], template=fn["template"],
                   source=cs["source"], target=cs["target"],
                   source_answer=fn["answers"][cs["source"]], target_answer=tgt_answer,
                   source_gated=src["gated"],
                   target_gated=cells[(cat["name"], fn["name"], cs["target"])]["gated"],
                   clean_top1=lm.dec(top1(src["lg"])),
                   target_rank_clean=rank_out(src["lg"], tgt_answer))
        for label, mk in (("coord", coord_swap_edits), ("subadd", swap_edits)):
            sw = lm.logits(src["ids"], mk(lm, src["ids"], s, t, band, alpha=1.0, clean=src["clean"]))[-1]
            row[label] = dict(swapped_top1=lm.dec(top1(sw)), follows=hit(sw, tgt_answer),
                              target_rank_swapped=rank_out(sw, tgt_answer))
        out["case"].append(row)

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
                    spon = [top1(src["lg"])]  # the spontaneous answer: GPT-2's own, before the swap
                    row = dict(category=cat["name"], function=fn["name"],
                               source=sa, target=ta, source_i=si, target_i=ti,
                               source_answer=src["answer"], target_answer=t_ans,
                               source_gated=src["gated"], target_gated=tgt["gated"],
                               distinct=distinct, echo=echo, before=before, subadd={}, coord={})
                    for label, mk in (("subadd", swap_edits), ("coord", coord_swap_edits)):
                        for al in ALPHAS:
                            lg = lm.logits(src["ids"], mk(lm, src["ids"], s, t, band,
                                                          alpha=al, clean=src["clean"]))[-1]
                            # the paper's Fig. 19 swap effect: dlp(target) - dlp(spontaneous)
                            effect = ((lp(lg, W(t_ans)) - lp(src["lg"], W(t_ans)))
                                      - (lp(lg, spon) - lp(src["lg"], spon)))
                            row[label][str(al)] = dict(after=rank_out(lg, t_ans),
                                                       hit=hit(lg, t_ans), got=lm.dec(top1(lg)),
                                                       effect=effect)
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
        # bare-template swap on gated source cells (coordinate swap, alpha=1)
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
                    lg = lm.logits(sc["ids"], coord_swap_edits(lm, sc["ids"], s, t, band))[-1]
                    fl["swap_n"] += 1
                    fl["swap_hit"] += hit(lg, t_ans)
        fl["per_cat"][cat["name"]] = cc
    out["floor"] = fl

    # ------------------------------------------------------------------ write
    R = jl.results_dir("c4_generalization")
    json.dump(out, open(R / "results.json", "w"), indent=1, default=float)
    json.dump([dict(category=c, function=f, arg=a, answer=cells[(c, f, a)]["answer"],
                    gated=cells[(c, f, a)]["gated"], prompt=cells[(c, f, a)]["text"])
               for (c, f, a) in cells],
              open(R / "prompts.json", "w"), indent=1)
    summary = summarize(out)
    open(R / "summary.txt", "w").write(summary)
    print(summary)


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
    for op in OPS:
        nfollow = sum(c[op]["follows"] for c in out["case"])
        L.append(f"\nE1 case study (Fig 18), {OPS[op]}: {out['case'][0]['source']} -> {out['case'][0]['target']}, "
                 f"one fixed swap ({nfollow}/{len(out['case'])} functions follow)")
        for c in out["case"]:
            s = c[op]
            L.append(f"  {c['function']:11s} clean={c['clean_top1']!r:11s} ({'right' if c['source_gated'] else 'WRONG'}) "
                     f"-> swapped={s['swapped_top1']!r:11s} want {c['target']}'s {c['target_answer']!r:10s} "
                     f"(rank {c['target_rank_clean']:>4}->{s['target_rank_swapped']:<3}) [{'FOLLOWS' if s['follows'] else 'no'}]")

    # E2 swap.  A swap can only succeed if GPT-2 knows the target argument's answer (target cell
    # gated), and only counts as moving the answer if that answer is not already GPT-2's top-1
    # before the swap (before > 1: e.g. GPT-2 answers "seven" to every month-number prompt, so a
    # swap *to* July would otherwise score as a success with nothing changed).  `clean` also drops
    # pairs whose two answers coincide or whose target answer appears in the worked examples.
    claude = claude_grid()

    def clean(r):
        return r["distinct"] and not r["echo"]

    def testable(r):
        return clean(r) and r["target_gated"] and r["before"] > 1

    def rate(rows, op, al="1.0"):
        return sum(r[op][al]["hit"] for r in rows)

    def subset(name, flt):
        rows = [r for r in out["swap"] if flt(r)]
        n = len(rows)
        cl = sum(claude[key(r)] for r in rows)
        L.append(f"\nE2 swap — {name}  (n={n} pairs)")
        for op in OPS:
            k1, k2 = rate(rows, op), rate(rows, op, "2.0")
            p, lo, hi = wilson(k1, n)
            L.append(f"     GPT-2, {OPS[op]:36s} a=1 {k1:3d}/{n:<3d} {p:5.1%} [{lo:3.0%},{hi:3.0%}]   a=2 {k2:3d}/{n}")
        L.append(f"     Claude Sonnet 4.5 on the same pairs (a=1, paper's Fig. 68 grid)  {cl:3d}/{n:<3d} {cl / max(n, 1):5.1%}")
        return rows

    subset("all 192 ordered pairs, ungated (paper: 76/192 at a=1, 101/192 at a=2)", lambda r: True)
    rows = subset("testable: target answer known to GPT-2 and not already its top-1 (the fair set)", testable)
    subset("stricter: GPT-2 also answers the source prompt correctly", lambda r: clean(r) and r["source_gated"] and r["target_gated"])
    L.append("  (without the before > 1 rule, target-gated pairs give coordinate "
             f"{rate([r for r in out['swap'] if clean(r) and r['target_gated']], 'coord')}/"
             f"{len([r for r in out['swap'] if clean(r) and r['target_gated']])}, of which "
             f"{sum(r['coord']['1.0']['hit'] for r in out['swap'] if clean(r) and r['target_gated'] and r['before'] == 1)} "
             "had the target answer at top-1 before any swap)")

    L.append("\n  testable pairs by category and function (a=1): GPT-2 coordinate | subtract-and-add | Claude, same pairs")
    for grp in (lambda r: r["category"], lambda r: (r["category"], r["function"])):
        by = {}
        for r in rows:
            by.setdefault(grp(r), []).append(r)
        for k, rs in sorted(by.items(), key=str):
            label = k if isinstance(k, str) else f"{k[0]}/{k[1]}"
            L.append(f"     {label:24s} n={len(rs):2d}   {rate(rs, 'coord'):2d} | {rate(rs, 'subadd'):2d} | "
                     f"{sum(claude[key(r)] for r in rs):2d}")
        L.append("")
    L.append("  Claude, all 12 swaps per function (Fig. 68): " + ", ".join(
        f"{fn} {sum(claude[k] for k in claude if k[1] == fn)}" for fn in dict.fromkeys(k[1] for k in claude)))

    # alpha=2 overshoot: the swap emits the injected ARGUMENT word rather than f(argument)
    for op in OPS:
        rs = [r for r in out["swap"] if clean(r)]
        k1 = sum(r[op]["1.0"]["got"].strip().lower() == r["target"].lower() for r in rs)
        k2 = sum(r[op]["2.0"]["got"].strip().lower() == r["target"].lower() for r in rs)
        L.append(f"  {OPS[op]}: the swap makes GPT-2 output the swapped-in argument itself on {k1}/{len(rs)} "
                 f"clean pairs at a=1 and {k2}/{len(rs)} at a=2")
    # ...while the swap still pushes the target answer further (the paper's Fig. 19 swap effect)
    L.append("  mean swap effect (dlp target - dlp spontaneous), coordinate swap, all 192 pairs: a=1 "
             f"{mean(r['coord']['1.0']['effect'] for r in out['swap']):+.2f}  a=2 "
             f"{mean(r['coord']['2.0']['effect'] for r in out['swap']):+.2f}   by category (a=1 / a=2): " +
             ", ".join(f"{c} {mean(r['coord']['1.0']['effect'] for r in out['swap'] if r['category'] == c):+.1f} / "
                       f"{mean(r['coord']['2.0']['effect'] for r in out['swap'] if r['category'] == c):+.1f}"
                       for c in ["countries", "months", "animals", "numbers"]))

    # E2 at top-5: the grid's ranks give Claude at alpha=1 (its alpha=2 ranks aren't released)
    K = 5
    crank = claude_ranks()
    pre = sum(r["before"] <= K for r in out["swap"])
    L.append(f"\nE2 swap at top-{K} — the target answer in the output top {K} (coordinate swap)")
    L.append(f"     all 192 pairs            GPT-2 a=1 {sum(r['coord']['1.0']['after'] <= K for r in out['swap']):3d}  "
             f"a=2 {sum(r['coord']['2.0']['after'] <= K for r in out['swap']):3d}   Claude a=1 "
             f"{sum(crank[key(r)] <= K for r in out['swap']):3d}   (GPT-2 already top-{K} before the swap: {pre})")
    rows5 = [r for r in out["swap"] if clean(r) and r["target_gated"] and r["before"] > K]
    L.append(f"     testable at top-{K} (n={len(rows5)}; as above, with the target outside the top {K} before)")
    for label, rs in [("all", rows5)] + sorted(
            ((f"{c}/{f}", [r for r in rows5 if (r["category"], r["function"]) == (c, f)])
             for c, f in dict.fromkeys((r["category"], r["function"]) for r in rows5)), key=str):
        L.append(f"       {label:22s} n={len(rs):2d}   GPT-2 a=1 {sum(r['coord']['1.0']['after'] <= K for r in rs):2d}  "
                 f"a=2 {sum(r['coord']['2.0']['after'] <= K for r in rs):2d}   Claude a=1 "
                 f"{sum(crank[key(r)] <= K for r in rs):2d}")

    # E3 loading (Fig 19 right)
    lo = out["loading"]
    L.append("\nE3 workspace loading (Fig 19 right) — cos(residual, arg lens vector), band mean")
    catload = {c: [x["loading"] for x in lo if x["category"] == c] for c in ["countries", "months", "animals", "numbers"]}
    for cat, v in sorted(((c, sum(vs) / len(vs)) for c, vs in catload.items()), key=lambda x: -x[1]):
        L.append(f"     {cat:9s} loading {v:+.3f}")
    # source-cell loading vs its swaps' success rate (testable pairs, coordinate swap)
    loadmap = {(x["category"], x["function"], x["arg"]): x["loading"] for x in lo}
    cell = {}
    for r in rows:
        c = cell.setdefault((r["category"], r["function"], r["source"]), [0, 0])
        c[0] += r["coord"]["1.0"]["hit"]
        c[1] += 1
    xs = [loadmap[k] for k in cell]
    ys = [cell[k][0] / cell[k][1] for k in cell]
    L.append(f"  source-cell loading vs its swaps' success (testable pairs, coordinate swap): "
             f"Spearman {spearman(xs, ys):+.2f} over {len(xs)} cells")
    # The paper's own measure: per function, mean loading vs the mean swap effect at alpha=1 over
    # all 12 pairs.  Its x is cos at the argument + cos at the readout; ours is their mean, which
    # doesn't change a correlation.
    paper = json.load(open(PAPER_SUMMARY))
    eff = defaultdict(list)
    for r in out["swap"]:
        eff[(r["category"], r["function"], r["source"])].append(r["coord"]["1.0"]["effect"])
    fns = list(dict.fromkeys(k[:2] for k in eff))
    g = {f: (mean(loadmap[k] for k in eff if k[:2] == f),
             mean(e for k in eff if k[:2] == f for e in eff[k])) for f in fns}
    c = {(p["cat"], p["func"]): (p["csum"], p["dlpc"]) for p in paper["pts"]}

    def corr(d, keep=lambda f: True):
        fs = [f for f in d if keep(f)]
        xs, ys = [d[f][0] for f in fs], [d[f][1] for f in fs]
        return f"Pearson {pearson(xs, ys):+.2f}  Spearman {spearman(xs, ys):+.2f}  (n={len(fs)})"

    L.append("  loading vs swap effect (dlp target - dlp spontaneous, a=1; the paper's Fig. 19 right), per function:")
    L.append(f"     GPT-2, coordinate swap        {corr(g)}")
    L.append(f"     Claude (paper's data)         {corr(c)}   (paper: r = {paper['r']})")
    L.append(f"     without the country functions: GPT-2 {corr(g, lambda f: f[0] != 'countries')}   "
             f"Claude {corr(c, lambda f: f[0] != 'countries')}")
    cells = {k: (loadmap[k], mean(v)) for k, v in eff.items()}
    L.append(f"     GPT-2 per source cell         {corr(cells)}")
    for f in fns:
        L.append(f"       {f[0] + '/' + f[1]:24s} loading {g[f][0]:.3f}  effect {g[f][1]:+6.2f}   "
                 f"Claude: csum {c[f][0]:.3f}  effect {c[f][1]:+6.2f}")

    # floor
    fl = out["floor"]
    L.append(f"\nfloor — paper's bare templates, no frame: capability {fl['cap']}/{fl['n']}; "
             f"coordinate swap on gated bare cells {fl['swap_hit']}/{fl['swap_n']}")
    L.append("  per category (bare capability): " +
             "  ".join(f"{c} {d['cap']}/{d['n']}" for c, d in fl["per_cat"].items()))
    return "\n".join(L)


# =============================================================================== gating classes
def gating_breakdown():
    """E2 swap rates under each gating class, split by function type, from an existing
    results.json.  Shows that the strict-gating 18% is composition (the successor functions),
    not gating."""
    r = json.load(open(jl.RESULTS / "c4_generalization/results.json"))
    sw = [x for x in r["swap"] if x["distinct"] and not x["echo"]]
    succ = {("months", "next_month"), ("numbers", "successor")}

    def rate(rows):
        h = sum(x["subadd"]["1.0"]["hit"] for x in rows)
        return f"{h}/{len(rows)} = {100 * h / max(len(rows), 1):.0f}%"

    def both(x):
        return x["source_gated"] and x["target_gated"]

    for label, sel in [("ungated", lambda x: True), ("target-gated", lambda x: x["target_gated"]),
                       ("both-gated", both)]:
        rows = [x for x in sw if sel(x)]
        print(f"{label:13s} all: {rate(rows):18s}"
              f" non-successor fns: {rate([x for x in rows if (x['category'], x['function']) not in succ]):18s}"
              f" successor fns: {rate([x for x in rows if (x['category'], x['function']) in succ])}")
    print("\nper function, hits/pairs by gate class [both | target-only | source-only | neither]")
    d = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for x in sw:
        k = (x["category"], x["function"])
        cls = ("both" if both(x) else "tgt" if x["target_gated"]
               else "src" if x["source_gated"] else "none")
        d[k][cls][0] += x["subadd"]["1.0"]["hit"]
        d[k][cls][1] += 1
    for k in sorted(d):
        print(f"  {k[0]:9s}/{k[1]:12s}",
              "  ".join(f"{c}:{d[k][c][0]}/{d[k][c][1]}" for c in ["both", "tgt", "src", "none"] if d[k][c][1]))


if __name__ == "__main__":
    gating_breakdown() if "gating" in sys.argv[1:] else main()
