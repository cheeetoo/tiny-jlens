"""One figure for criterion 2.  Run after run.py.

A  the condition ladder (2a): median best band rank by condition; bar = condition median,
   dots = per-phrasing medians (lower = more present in the workspace).  The story: mention
   primes, focus adds on top, "ignore" does nothing, "don't think" backfires.
B  per-layer localization (2a): median best rank across L6..L10 per condition; band 7-9 shaded.
   The effect strengthens toward the motor layer L10 -- the caveat, made visible.
C  privileging (2d): the "imagine French" claim's effect as a share of a real French sentence's,
   in the lens channel vs the J-orthogonalized-probe channel -- the dissociation.
D  capability floors (2b, 2c): can the model do the task at all, and does the target enter the
   workspace?  Math fails the task; the paired-question task succeeds but its label never enters.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
out = json.load(open(f"{HERE}/results/results.json"))
BAND = out["band"]
LAYERS = out["layers_reported"]
CONDS = ["baseline", "mention", "focus", "dismissal", "negated"]
COLOR = {"baseline": "#999999", "mention": "#4C72B0", "focus": "#2E7D32",
         "dismissal": "#DD8452", "negated": "#C44E52"}
C = out["concept"]


def cond_pair_medians(cond):
    """per (word,carrier): median best band rank over that condition's phrasings."""
    d = defaultdict(list)
    for r in C:
        if r["cond"] == cond:
            d[(r["word"], r["carrier"])].append(r["best"])
    return {k: median(v) for k, v in d.items()}


fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

# ---------------------------------------------------------------- A: condition ladder
# bar = fraction of (word,carrier) pairs whose median-over-phrasings best band rank <= 25
# (the word reaches the workspace proper); dots = per-phrasing hit@25.
ax = axes[0, 0]
for i, cond in enumerate(CONDS):
    pm = cond_pair_medians(cond)
    hit25 = sum(v <= 25 for v in pm.values()) / len(pm)
    ax.bar(i, hit25, color=COLOR[cond], alpha=0.8, zorder=1, width=0.68)
    ax.text(i, hit25 + 0.006, f"{hit25:.0%}", ha="center", fontsize=9, color="#333")
    if cond != "baseline":
        phs = sorted({r["phrasing"] for r in C if r["cond"] == cond})
        for p in phs:
            vals = [r["best"] for r in C if r["cond"] == cond and r["phrasing"] == p]
            ax.scatter(i, sum(v <= 25 for v in vals) / len(vals), s=16, color="k", alpha=0.5, zorder=3)
ax.set_xticks(range(len(CONDS)))
ax.set_xticklabels([c + f"\n({['—', 'control', 'focus', 'suppress', 'suppress'][i]})"
                    for i, c in enumerate(CONDS)], fontsize=8)
ax.set_ylim(0, 0.26)
ax.set_ylabel("word reaches the workspace band (top-25)")
ax.set_title("A  instructed modulation, five conditions (band 7–9)", loc="left", fontsize=10)
# paired-contrast callouts (the significance story)
ax.text(1, 0.235, "mention > baseline\n86%  (priming)", fontsize=7, color="#4C72B0", ha="center")
ax.text(2, 0.235, "focus > mention\n76%  (instructed)", fontsize=7, color="#2E7D32", ha="center")
ax.text(3, 0.235, "ignore ≈ mention\n50%  (no control)", fontsize=7, color="#DD8452", ha="center")
ax.text(4, 0.235, "don't-think > mention\n84%  (white bear)", fontsize=7, color="#C44E52", ha="center")

# ---------------------------------------------------------------- B: per-layer localization
ax = axes[0, 1]
ax.axvspan(BAND[0] - 0.35, BAND[-1] + 0.35, color="#4C72B0", alpha=0.08, zorder=0)
ax.text(sum(BAND) / len(BAND), 1500, "workspace\nband", ha="center", fontsize=7, color="#4C72B0")
for cond in CONDS:
    sub = [r for r in C if r["cond"] == cond]
    ys = [median([r["per_layer"][str(L)] for r in sub]) for L in LAYERS]
    ax.plot(LAYERS, ys, marker="o", ms=4, color=COLOR[cond], label=cond, lw=1.6)
ax.text(10, 60, "motor layer\n(lens ≈ output)", ha="center", fontsize=7, color="#555")
ax.set_yscale("log")
ax.set_ylim(2000, 30)
ax.set_xticks(LAYERS)
ax.set_xlabel("layer")
ax.set_ylabel("median best rank of the word  (↑ = more present)")
ax.set_title("B  the effect strengthens toward the motor layer L10", loc="left", fontsize=10)
ax.legend(fontsize=7, loc="lower left", ncol=2)

# ---------------------------------------------------------------- C: privileging dissociation
ax = axes[1, 0]
IM = out["imagine"]


def cond_delta(cond, ch):
    ds = []
    for r in IM:
        base = sum(m[ch] for m in r["neutral"]) / len(r["neutral"])
        ds += [m[ch] - base for m in r[cond]]
    return sum(ds) / len(ds)


channels = [("lens", "lens ‘French’\n(J-space readout)"), ("orth", "J-orthogonalized\nFrench probe")]
x = np.arange(len(channels))
claim = [cond_delta("claim", ch) / cond_delta("real", ch) for ch, _ in channels]
ax.bar(x - 0.2, [1.0, 1.0], width=0.38, color="#B0B0B0", alpha=0.8, label="real French sentence")
ax.bar(x + 0.2, claim, width=0.38, color="#4C72B0", alpha=0.85, label="“imagine French” claim")
for xi, c in zip(x, claim):
    ax.text(xi + 0.2, c + 0.03, f"{c:.0%}", ha="center", fontsize=9, color="#4C72B0")
ax.set_xticks(x)
ax.set_xticklabels([lab for _, lab in channels], fontsize=8)
ax.set_ylabel("effect as a share of a real French sentence")
ax.set_ylim(0, 1.2)
ax.legend(fontsize=8, loc="upper right")
ax.set_title("C  the instruction writes to the J-space, not the representation", loc="left", fontsize=10)

# ---------------------------------------------------------------- D: capability floors
ax = axes[1, 1]
MC = out["math_capability"]
math_solve = sum(m["correct"] for m in MC) / len(MC)
ml = out["math_lens"]
math_ws = sum(r["best"] <= 25 for r in ml) / len(ml) if ml else 0.0
DC = out["demand_capability"]
demand_task = sum(d["q1_hit"] for d in DC) / len(DC)
DD = out["demand"]
demand_ws = sum(d["hit10"] for d in DD if d["q"] == "q2") / max(1, sum(d["n"] for d in DD if d["q"] == "q2"))
labels = ["2b math:\nsolve the task", "2b math:\nanswer in workspace",
          "2c demand:\ndo the task (Q1)", "2c demand:\nname property in workspace"]
vals = [math_solve, math_ws, demand_task, demand_ws]
cols = ["#C44E52", "#C44E52", "#2E7D32", "#C44E52"]
ax.barh(range(len(labels)), vals, color=cols, alpha=0.8)
for i, v in enumerate(vals):
    ax.text(min(v + 0.02, 0.9), i, f"{v:.0%}", va="center", fontsize=9)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlim(0, 1.05)
ax.set_xlabel("rate")
ax.set_title("D  capability floors: math unsolvable; property never named", loc="left", fontsize=10)

fig.suptitle("Criterion 2 — directed modulation — gpt2-small", y=1.00, fontsize=13)
fig.tight_layout()
os.makedirs(f"{HERE}/figures", exist_ok=True)
fig.savefig(f"{HERE}/figures/criterion2.png", dpi=160, bbox_inches="tight")
print("wrote figures/criterion2.png")
