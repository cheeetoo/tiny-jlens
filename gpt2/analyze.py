"""Unified summary of the C1-C5 battery with uncertainty.

Reads results/*.json (regenerated on the expanded pools) and prints, per
experiment: n, effect estimate, 95% Wilson CI on proportions, exact sign
tests for paired designs, and cluster bootstrap CIs where trials share a
category/country/word/passage.

Run:  python analyze.py [model]
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "/tiny-jlens/gpt2")

from stats import cluster_boot, fmt_prop, sign_test, wilson

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
R = "/tiny-jlens/gpt2/results"


def load(name):
    try:
        with open(f"{R}/{name}_{MODEL}.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def med(xs):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def cb(rows, key, val):
    """cluster bootstrap of mean(val) clustered by key -> 'xx% [lo, hi]'"""
    by = {}
    for r in rows:
        by.setdefault(key(r), []).append(1.0 if val(r) else 0.0)
    m, lo, hi = cluster_boot(by)
    return f"{100*m:.0f}% [{100*lo:.0f}, {100*hi:.0f}] over {len(by)} clusters"


print(f"===== {MODEL}: expanded-pool battery with uncertainty =====\n")

# ---------------- C1 ----------------
d = load("c1_corr")
if d:
    layers = sorted(d[0]["per_layer"], key=int)
    n_fmt = len({r["fmt"] for r in d})
    print(f"C1a report correlation ({len(d)} (format, category) cells over "
          f"{n_fmt} formats; mean Spearman, cluster-boot 95% CI over categories):")
    for l in layers[-4:]:
        by = {}
        for r in d:
            by.setdefault(r["cat"], []).append(r["per_layer"][l])
        m, lo, hi = cluster_boot(by)
        per_fmt = "  ".join(
            f"f{f}:{sum(r['per_layer'][l] for r in d if r['fmt'] == f) / max(1, sum(1 for r in d if r['fmt'] == f)):+.2f}"
            for f in sorted({r["fmt"] for r in d}))
        print(f"  L{int(l):2d}  {m:+.3f} [{lo:+.3f}, {hi:+.3f}]   ({per_fmt})")

d = load("c1_swap")
if d:
    for r in d:
        r.setdefault("fmt", 0)
    fmts = sorted({r["fmt"] for r in d})
    print(f"\nC1b report swap over {len(fmts)} few-shot formats "
          f"(success = target in graded top-5):")
    for centered in (False, True):
        for alpha in (1.0, 2.0):
            sel = [r for r in d if r["centered"] == centered and r["alpha"] == alpha]
            if not sel:
                continue
            k = sum(r["top5"] for r in sel)
            per_fmt = " ".join(
                f"f{f}:{sum(r['top5'] for r in sel if r['fmt'] == f)}"
                f"/{sum(1 for r in sel if r['fmt'] == f)}" for f in fmts)
            print(f"  {'centered' if centered else 'raw     '} a={alpha}: "
                  f"top5 {fmt_prop(k, len(sel))}   "
                  f"(cluster {cb(sel, lambda r: r['cat'], lambda r: r['top5'])}); "
                  f"top1 {fmt_prop(sum(r['top1'] for r in sel), len(sel))}   [{per_fmt}]")

d = load("c1_inject")
if d and "kind" in d[0]:
    rep_frames = sorted({r["frame"] for r in d if r["kind"] == "report"})
    ctl_frames = sorted({r["frame"] for r in d if r["kind"] == "control"})
    print(f"\nC1d inject ({len(rep_frames)} report frames, {len(ctl_frames)} "
          f"noun-expecting controls; success = injected concept in top-5):")
    for s in sorted({r["strength"] for r in d}):
        rep = [r for r in d if r["strength"] == s and r["kind"] == "report"]
        ctl = [r for r in d if r["strength"] == s and r["kind"] == "control"]
        kr, kc = sum(r["rank"] < 5 for r in rep), sum(r["rank"] < 5 for r in ctl)
        pf = " ".join(f"f{f}:{sum(r['rank'] < 5 for r in rep if r['frame'] == f)}"
                      f"/{sum(1 for r in rep if r['frame'] == f)}" for f in rep_frames)
        print(f"  strength {s:4.2f}: report {fmt_prop(kr, len(rep))}   "
              f"control {fmt_prop(kc, len(ctl))}   [{pf}]")

d = load("c1c_addendum")
if d:
    n = len(d["full"])
    print(f"\nC1c report privilege (matched-norm probe-component swaps, n={n}):")
    for cond in ("full", "J", "nonJ", "nonJ_clamped"):
        print(f"  {cond:14s} {fmt_prop(sum(d[cond]), n)}")

# ---------------- C2 ----------------
d = load("c2")
if d and "cond" in d[0]:
    base_by = {(r["word"], r["sentence"]): r for r in d if r["cond"] == "base"}
    n_pairs = len(base_by)
    n_ph = {c: len({r["phrasing"] for r in d if r["cond"] == c})
            for c in ("think", "dont")}
    print(f"\nC2 directed modulation (n={n_pairs} word x sentence trials x "
          f"{n_ph['think']}/{n_ph['dont']} think/dont phrasings; best lens rank "
          f"over transcription span; base median "
          f"{med(r['best'] for r in base_by.values())}):")
    for cond in ("think", "dont"):
        sel = [r for r in d if r["cond"] == cond]
        wins = sum(r["best"] < base_by[(r["word"], r["sentence"])]["best"] for r in sel)
        ne = sum(r["best"] != base_by[(r["word"], r["sentence"])]["best"] for r in sel)
        pf = " ".join(
            f"p{p}:{med(r['best'] for r in sel if r['phrasing'] == p)}"
            for p in sorted({r["phrasing"] for r in sel}))
        print(f"  {cond}: pooled median {med(r['best'] for r in sel)}, "
              f"beats base {wins}/{ne} "
              f"(cluster {cb(sel, lambda r: r['word'], lambda r: r['best'] < base_by[(r['word'], r['sentence'])]['best'])})   "
              f"[per-phrasing medians {pf}]")
    think = [r for r in d if r["cond"] == "think"]
    dont_by = {(r["word"], r["sentence"], r["phrasing"]): r
               for r in d if r["cond"] == "dont"}
    paired = [(r, dont_by[(r["word"], r["sentence"], r["phrasing"])])
              for r in think if (r["word"], r["sentence"], r["phrasing"]) in dont_by]
    w = sum(a["best"] < b["best"] for a, b in paired)
    ne = sum(a["best"] != b["best"] for a, b in paired)
    print(f"  think < dont (same phrasing index): {w}/{ne}, "
          f"p = {sign_test(w, ne):.2e}")

d = load("c2_privilege")
if d:
    n = len(d)
    print(f"\nC2 privilege, word form (n={n}; negative-by-design: the "
          f"instruction MENTIONS the word, contaminating all channels):")
    real_dl = sum(r["real"][0] - r["base"][0] for r in d) / n
    real_dp = sum(r["real"][1] - r["base"][1] for r in d) / n
    for cond in ("think", "dont", "real"):
        dl = sum(r[cond][0] - r["base"][0] for r in d) / n
        dp = sum(r[cond][1] - r["base"][1] for r in d) / n
        wl = sum(r[cond][0] > r["base"][0] for r in d)
        wp = sum(r[cond][1] > r["base"][1] for r in d)
        print(f"  {cond:6s} lens delta {dl:+.2f} ({100*dl/real_dl:+.0f}% of real; "
              f"up {wl}/{n})   J-orth probe delta {dp:+.3f} "
              f"({100*dp/real_dp:+.0f}% of real; up {wp}/{n})")

d = load("c2_imagine")
if d and isinstance(d[0]["neutral"][0], list):
    n = len(d)
    n_claim, n_neu = len(d[0]["claim"]), len(d[0]["neutral"])
    base_v = lambda r, i: sum(m[i] for m in r["neutral"]) / len(r["neutral"])
    print(f"\nC2 privilege, property form (n={n} sentences x {n_claim} claim / "
          f"{n_neu} neutral headers; deltas vs mean-neutral, as % of the "
          f"real-French-stimulus delta):")
    real_d = [sum(sum(m[i] for m in r["real"]) / len(r["real"]) - base_v(r, i)
                  for r in d) / n for i in range(3)]
    for cond in ("claim", "real"):
        for vi in range(len(d[0][cond])):
            parts = []
            for i, name in [(0, "lens'French'"), (1, "J-orth probe")]:
                dv = sum(r[cond][vi][i] - base_v(r, i) for r in d) / n
                w = sum(r[cond][vi][i] > base_v(r, i) for r in d)
                parts.append(f"{name} {100*dv/real_d[i]:+.0f}% (up {w}/{n})")
            print(f"  {cond}[{vi}] " + "   ".join(parts))

d = load("c2_imagine_tense")
if d and isinstance(d[0]["neutral"][0], list):
    n = len(d)
    base_v = lambda r, i: sum(m[i] for m in r["neutral"]) / len(r["neutral"])
    real_d = [sum(sum(m[i] for m in r["real"]) / len(r["real"]) - base_v(r, i)
                  for r in d) / n for i in range(2)]
    print(f"\nC2 privilege, property form, PAST TENSE category (n={n}; deltas "
          f"vs mean-neutral, as % of the real-past-tense delta; real lens "
          f"delta raw {real_d[0]:+.2f}):")
    for cond in ("claim", "real"):
        for vi in range(len(d[0][cond])):
            parts = []
            for i, name in [(0, "lens'past'"), (1, "J-orth probe")]:
                dv = sum(r[cond][vi][i] - base_v(r, i) for r in d) / n
                w = sum(r[cond][vi][i] > base_v(r, i) for r in d)
                parts.append(f"{name} {100*dv/real_d[i]:+.0f}% (up {w}/{n})")
            print(f"  {cond}[{vi}] " + "   ".join(parts))

d = load("c2_demand")
if d:
    n = len(d)
    n_var = len({r.get("variant", 0) for r in d})
    wins = sum(r["demand"] < r["nodemand"] for r in d)
    ne = sum(r["demand"] != r["nodemand"] for r in d)
    pv = " ".join(
        f"v{v}:{sum(r['demand'] < r['nodemand'] for r in d if r.get('variant', 0) == v)}"
        f"/{sum(r['demand'] != r['nodemand'] for r in d if r.get('variant', 0) == v)}"
        for v in sorted({r.get("variant", 0) for r in d}))
    print(f"\nC2 demand-loading (n={n} over {n_var} shot variants): median rank "
          f"demand {med(r['demand'] for r in d)} vs no-demand "
          f"{med(r['nodemand'] for r in d)}; demand<nodemand {wins}/{ne}, "
          f"p={sign_test(wins, ne):.2e}   "
          f"(cluster {cb(d, lambda r: r['word'], lambda r: r['demand'] < r['nodemand'])})   [{pv}]")

# ---------------- C3 ----------------
d = load("c3_readout")
if d:
    n = len(d)
    best = [min(r["ranks"]["intermediate"].values()) for r in d]
    k = sum(b < 10 for b in best)
    uns = [r for r in d if r["unspoken"]]
    ku = sum(min(r["ranks"]["intermediate"].values()) < 10 for r in uns)
    kn = sum(min(r["ranks"]["null"].values()) < 10 for r in d if r["ranks"]["null"])
    nn = sum(1 for r in d if r["ranks"]["null"])
    print(f"\nC3a intermediate readout: in lens top-10 at some layer "
          f"{fmt_prop(k, n)}   unspoken-only {fmt_prop(ku, len(uns))}   "
          f"null control {fmt_prop(kn, nn)}")

d = load("c3_swap")
if d:
    items = len({r["prompt"] for r in d})
    print(f"\nC3b intermediate swap (n={items} items; success = counterfactual "
          f"answer graded top-1):")
    for window in ("late", "mid+", "all"):
        for form, centered, alpha in [("coord", False, 1.0), ("coord", True, 1.0),
                                      ("coord", False, 2.0)]:
            sel = [r for r in d if r["window"] == window and r["form"] == form
                   and r["centered"] == centered and r["alpha"] == alpha]
            if sel:
                k = sum(r["hit"] for r in sel)
                print(f"  {window:5s} {form} {'cen' if centered else 'raw'} a={alpha}: "
                      f"{fmt_prop(k, len(sel))}   "
                      f"(cluster {cb(sel, lambda r: r['intermediate'], lambda r: r['hit'])})")
    sel = [r for r in d if r["window"] == "late" and r["form"] == "proj"
           and not r["centered"] and r["alpha"] == 1.0]
    if sel:
        print(f"  late proj raw a=1 (gauge check, expect ~0): "
              f"{fmt_prop(sum(r['hit'] for r in sel), len(sel))}")

d = load("c3_probe")
if d:
    n = len(d)
    nested = "full" not in d[0]["conds"]  # variant schema: conds[variant][cond]
    variants = (sorted(d[0]["conds"]) if nested else [None])
    print(f"\nC3d probe privilege (n={n}; naming = country-name cues k=16 "
          f"[legacy, favors J], implying = paper-faithful attribute cues k=25; "
          f"matched = norm-matched perturbations, natural = raw component "
          f"magnitudes):")
    for v in variants:
        parts = []
        for cond in ("full", "J", "nonJ", "nonJ_clamped"):
            c = d[0]["conds"]
            k = sum((r["conds"][v][cond] if nested else r["conds"][cond])["hit"]
                    for r in d)
            parts.append(f"{cond} {fmt_prop(k, n)}")
        print(f"  {v or 'matched':18s} " + "   ".join(parts))

d = load("c3_crossfn")
if d:
    n = len(d)
    print(f"\nC3c cross-function (one swap must flip BOTH questions, n={n}): "
          f"both {fmt_prop(sum(r['both'] for r in d), n)}   "
          f"any {fmt_prop(sum(r['any'] for r in d), n)}")

# ---------------- C4 ----------------
d = load("c4")
if d:
    ok = [r for r in d if r["tgt_pre"] >= 10]
    k = sum(r["hit"] for r in ok)
    print(f"\nC4 flexible generalization (target not already top-10): "
          f"{fmt_prop(k, len(ok))}   "
          f"(cluster {cb(ok, lambda r: (r['A'], r['B']), lambda r: r['hit'])})")
    funcs = sorted({(r["category"], r["func"]) for r in ok})
    for cat, fn in funcs:
        sel = [r for r in ok if r["category"] == cat and r["func"] == fn]
        print(f"  {cat:10s} {fn:12s} {fmt_prop(sum(r['hit'] for r in sel), len(sel))}")
    base_fn = lambda f: f.rstrip("0123456789")
    pair_stats, pair_bases = {}, {}
    for r in ok:
        key = (r["category"], r["A"], r["B"])
        pair_stats.setdefault(key, []).append(r["hit"])
        pair_bases.setdefault(key, set()).add(base_fn(r["func"]))
    multi = {p: v for p, v in pair_stats.items() if len(pair_bases[p]) >= 2}
    both = sum(all(v) for v in multi.values())
    print(f"  broadcast: same swap redirects ALL functions, over pairs spanning "
          f">=2 distinct base functions: {fmt_prop(both, len(multi))}")
    hit_load = [r["loading"] for r in ok if r["hit"]]
    miss_load = [r["loading"] for r in ok if not r["hit"]]
    if hit_load and miss_load:
        print(f"  workspace loading: hits {sum(hit_load)/len(hit_load):.3f} "
              f"vs misses {sum(miss_load)/len(miss_load):.3f}")

# ---------------- C5 ----------------
d = load("c5a")
if d:
    cont = [r for r in d if r["tasks"]["continuation"]["clean_lang"] == r["lang"]]
    tasks = [t for t in ("report", "country", "report2", "country2")
             if t in d[0]["tasks"]]
    print(f"\nC5a same-latent dissociation (capability-passing trials; "
          f"*2 = frame-diversity variant formats):")
    for task in tasks:
        sel = [r for r in d if r["tasks"][task]["clean_ok"]]
        k = sum(r["tasks"][task]["swap_flipped"] for r in sel)
        by_lang = {}
        for r in sel:
            by_lang.setdefault(r["lang"], []).append(r["tasks"][task]["swap_flipped"])
        langs = ", ".join(f"{l} {sum(v)}/{len(v)}" for l, v in sorted(by_lang.items()))
        # sensitivity framing: did the swap change the answer at all?
        changed = sum(r["tasks"][task]["swap_flipped"]
                      or r["tasks"][task]["swap_got"].strip().lower()
                      not in (r["lang"].lower(),
                              {"French": "france", "German": "germany",
                               "Spanish": "spain", "Italian": "italy"}[r["lang"]])
                      for r in sel)
        print(f"  {task:9s} flips {fmt_prop(k, len(sel))}   "
              f"changed {changed}/{len(sel)}   by lang: {langs}")
    changed = [r for r in cont if r["tasks"]["continuation"]["swap_lang"] != r["lang"]]
    print(f"  continuation changed:   {fmt_prop(len(changed), len(cont))} "
          f"(automatic task; prediction: ~0)")
    pres = [r["tasks"][t]["presence_rank"] for r in d
            for t in ("report", "country", "continuation")]
    print(f"  presence rank (median over all conditions): {med(pres)}")

print("\nC5b whole-J-space ablation (score = correct/n; [95% Wilson]):")
C5B_SHALLOW = ["wikitext", "copy", "cont_lang", "seq_formulaic", "idiom",
               "collocation", "func_cloze", "punct", "agreement"]
C5B_FLEX = ["twohop", "country", "report"]
for suffix in [f"k{kk}" for kk in (1, 2, 3, 5, 10)] + ["k3_L9-10"]:
    try:
        with open(f"{R}/c5b_{MODEL}_{suffix}.json") as f:
            d = json.load(f)
    except FileNotFoundError:
        continue
    def sc(label, name):
        v = d[label][name]
        if isinstance(v, list):
            return v[0] / v[1] if v[1] else float("nan"), v[0], v[1]
        return v, None, None
    for tag, names in (("S", C5B_SHALLOW), ("F", C5B_FLEX)):
        parts = []
        for name in names:
            if name not in d["clean"]:
                continue
            c, _, _ = sc("clean", name)
            a, ka, na = sc("ablate", name)
            r_, _, _ = sc("randproj", name)
            if ka is not None and na and c:
                lo, hi = wilson(ka, na)
                parts.append(f"{name} {a/c:.2f} [{lo/c:.2f},{hi/c:.2f}] (ctl {r_/c:.2f})")
            elif c:
                parts.append(f"{name} {a/c:.2f} (ctl {r_/c:.2f})")
        print(f"  {suffix:>8s} {tag}: " + "  ".join(parts))
    # answer-token class split (the content-channel readout): excess damage
    # vs the random-projection control, pooled over wikitext+copy+suite
    pooled = {}
    for skey in ("wiki_strata", "copy_strata", "suite_strata"):
        s = d.get(skey, {})
        if "ablate" not in s:
            continue
        for c in ("content", "function", "punct"):
            ka, na = s["ablate"].get(c, (0, 0))
            kr, nr = s.get("randproj", {}).get(c, (0, 0))
            if na and nr:
                agg = pooled.setdefault(c, [0, 0, 0, 0])
                agg[0] += ka; agg[1] += na; agg[2] += kr; agg[3] += nr
    if pooled:
        line = "  " + " " * 8 + " class excess damage (ctl - ablate): "
        line += "  ".join(f"{c} {p[2]/p[3] - p[0]/p[1]:+.2f}"
                          for c, p in pooled.items())
        print(line)
print("  (S = shallow/automatic, F = flexible; entries are retention vs "
      "clean, ctl = random-projection control; k3_L9-10 = the light-dose "
      "narrow-band corner)")
