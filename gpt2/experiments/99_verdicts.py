"""Compute the CONFIRMED.md bar checks mechanically from result JSONs.

Run:  python experiments/99_verdicts.py [model] [suffix]
      (suffix "" for exploration files, "_confirm" for the confirmatory)
"""

import json
import math
import sys

sys.path.insert(0, "/tiny-jlens/gpt2")

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
SUF = sys.argv[2] if len(sys.argv) > 2 else "_confirm"
R = "/tiny-jlens/gpt2/results"


def load(name):
    try:
        return json.load(open(f"{R}/{name}"))
    except FileNotFoundError:
        return None


def binom_p(k, n):  # two-sided sign test vs 0.5
    if n == 0:
        return float("nan")
    from math import comb
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


checks = []


def bar(name, value, passed, note=""):
    checks.append((name, value, passed, note))


# ---- C1a
d = load(f"c1_corr_{MODEL}{SUF}.json")
if d:
    import torch
    vals = torch.tensor([(a + b) / 2 for a, b in zip(d[sorted(d, key=int)[-2]],
                                                     d[sorted(d, key=int)[-1]])])
    torch.manual_seed(0)
    boots = torch.stack([vals[torch.randint(0, len(vals), (len(vals),))].mean()
                         for _ in range(10000)])
    lo = boots.quantile(0.025).item()
    bar("C1a corr(last-2-layer mean)", f"{vals.mean():.3f} CI[{lo:.2f},{boots.quantile(0.975):.2f}] n={len(vals)}",
        lo > 0)

# ---- C1b
d = load(f"c1_swap_{MODEL}{SUF}.json")
if d is not None:
    sel = [r for r in d if r["centered"] and r["alpha"] == 2.0]
    if sel:
        rate = sum(r["top5"] for r in sel) / len(sel)
        bar("C1b swap-to-report top5 (cen a2)", f"{rate:.0%} n={len(sel)}", rate >= 0.40)
    else:
        bar("C1b swap-to-report", "no evaluable swaps", None, "capability")

# ---- C1d
d = load(f"c1_inject_{MODEL}{SUF}.json")
if d:
    s25 = [r for r in d if r["strength"] == 0.25]
    rep = sum(r["report_rank"] < 5 for r in s25) / max(1, len(s25))
    blurt = sum(r["control_rank"] < 5 for r in s25) / max(1, len(s25))
    bar("C1d inject s=.25 top5 vs blurt", f"{rep:.0%} vs {blurt:.0%}",
        rep >= 0.40 and rep >= 2 * blurt)

# ---- C2
d = load(f"c2_{MODEL}{SUF}.json")
if d:
    n = len(d)
    fb = sum(r["think"]["best"] < r["base"]["best"] for r in d)
    fi = sum(r["think"]["best"] < r["dont"]["best"] for r in d)
    ib = sum(r["dont"]["best"] < r["base"]["best"] for r in d)
    nb = [r for r in d if r["think"]["blurt_rank"] >= 10]
    fb_nb = sum(r["think"]["best"] < r["base"]["best"] for r in nb)
    bar("C2.1 focus<base", f"{fb}/{n} p={binom_p(fb, n):.1e}",
        fb / n >= 0.80 and binom_p(fb, n) < 0.001)
    bar("C2.2 focus<ignore", f"{fi}/{n} p={binom_p(fi, n):.1e}",
        fi / n >= 0.65 and binom_p(fi, n) < 0.01)
    bar("C2.3 ignore<base (white-bear)", f"{ib}/{n}", ib / n > 0.5)
    bar("C2.4 ordering among non-blurt", f"{fb_nb}/{len(nb)}",
        len(nb) > 0 and fb_nb / len(nb) >= 0.80)

# ---- C3a
d = load(f"c3_readout_{MODEL}{SUF}.json")
if d:
    uns = [r for r in d if r["unspoken"]]
    hit = sum(min(r["ranks"]["intermediate"].values()) < 10 for r in uns)
    bar("C3a unspoken readout top10", f"{hit}/{len(uns)}",
        len(uns) > 0 and hit / len(uns) >= 0.50)
    shortcut = [r for r in uns if r["family"] == "city_language"]
    rest = [r for r in uns if r["family"] != "city_language"]
    if rest:
        h2 = sum(min(r["ranks"]["intermediate"].values()) < 10 for r in rest)
        bar("C3a excl. shortcut family (descriptive)", f"{h2}/{len(rest)}", None)

# ---- C3b
d = load(f"c3_swap_{MODEL}{SUF}.json")
if d:
    sel = [r for r in d if r["window"] == "late" and r["form"] == "coord"
           and r["alpha"] == 1.0 and (r["centered"] or SUF == "")]
    if SUF == "":
        sel = [r for r in sel if not r["centered"]]
    if sel:
        rate = sum(r["hit"] for r in sel) / len(sel)
        bar("C3b swap top1 (late, a1)", f"{rate:.0%} n={len(sel)}", rate >= 0.30)

# ---- C3c
d = load(f"c3_crossfn_{MODEL}{SUF}.json")
if d is not None:
    if len(d) >= 8:
        both = sum(r["both"] for r in d)
        bar("C3c crossfn both-flip", f"{both}/{len(d)}", both / len(d) >= 0.25)
    else:
        bar("C3c crossfn", f"n={len(d)} (<8)", None, "capability")

# ---- C3d
d = load(f"c3_probe_{MODEL}{SUF}.json")
if d:
    n = len(d)
    rates = {c: sum(r["conds"][c]["hit"] for r in d) / n
             for c in ("full", "J", "nonJ", "nonJ_clamped")}
    bar("C3d J>=2x nonJ", f"J {rates['J']:.0%} nonJ {rates['nonJ']:.0%} n={n}",
        rates["J"] >= 2 * rates["nonJ"] and rates["nonJ_clamped"] <= rates["nonJ"] / 2,
        f"clamp {rates['nonJ_clamped']:.0%}")

# ---- C4
d = load(f"c4_{MODEL}{SUF}.json")
if d:
    ok = [r for r in d if r["tgt_pre"] >= 10]
    if ok:
        rate = sum(r["hit"] for r in ok) / len(ok)
        cats = {}
        for r in ok:
            cats.setdefault(r["category"], []).append(r["hit"])
        best = max(sum(v) / len(v) for v in cats.values())
        bar("C4 overall top1", f"{rate:.0%} n={len(ok)}; {len(cats)} cats, best {best:.0%}",
            rate >= 0.25 and best >= 0.50)

# ---- C5a
d = load(f"c5a_{MODEL}{SUF}.json")
if d:
    rep = [r for r in d if r["tasks"]["report"]["clean_ok"]]
    ctry = [r for r in d if r["tasks"]["country"]["clean_ok"]]
    flex_n = sum(r["tasks"]["report"]["swap_flipped"] for r in rep) + \
        sum(r["tasks"]["country"]["swap_flipped"] for r in ctry)
    flex_d = len(rep) + len(ctry)
    cont = [r for r in d if r["tasks"]["continuation"]["clean_lang"] == r["lang"]]
    redirect = sum(r["tasks"]["continuation"]["swap_lang"] == r["alt"] for r in cont)
    bar("C5a flexible follows", f"{flex_n}/{flex_d}", flex_d > 0 and flex_n / flex_d >= 0.60)
    bar("C5a automatic redirects", f"{redirect}/{len(cont)}",
        len(cont) > 0 and redirect / len(cont) <= 0.15)

# ---- C5b
d = load(f"c5b_{MODEL}{SUF}_k1.json")
if d:
    c, a = d["clean"], d["ablate"]
    ret = {t: (a[t] / c[t] if c[t] else float("nan")) for t in c}
    shallow = ["cont_lang", "copy", "wikitext"]
    flex = ["twohop", "country", "report"]
    sh_ok = [t for t in shallow if not math.isnan(ret[t]) and ret[t] >= 0.80]
    fl_drop = [t for t in flex if not math.isnan(ret[t]) and ret[t] <= 0.60]
    ctrl_ok = all(
        (d[m][t] / c[t] if c[t] else 1) >= 0.8 or math.isnan(d[m][t] / c[t] if c[t] else 1)
        for m in ("randproj", "noise") for t in flex if c[t])
    bar("C5b k=1 shallow>=80%", f"{ {t: round(ret[t],2) for t in shallow} }",
        len(sh_ok) == 3)
    bar("C5b k=1 >=2 flexible drop>=40%", f"{ {t: round(ret[t],2) for t in flex} }",
        len(fl_drop) >= 2)
    bar("C5b controls leave flexible >=80%",
        f"{ {m: {t: round(d[m][t]/c[t],2) for t in flex if c[t]} for m in ('randproj','noise')} }",
        ctrl_ok)

print(f"\n=== bar checks: {MODEL}{SUF} ===")
for name, value, passed, note in checks:
    mark = {True: "PASS", False: "MISS", None: "n/a "}[passed]
    print(f"  [{mark}] {name:38s} {value}  {note}")
