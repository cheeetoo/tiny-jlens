"""One figure for criterion 1.  Run after run.py.

A  Spearman(lens, output) over the candidates, per layer: bar = mean over categories, dots = categories.
B  the swap: where targets START (output rank before) vs where they END (after) -- the swap
   drives ~every target to rank 1, so instead of a before/after scatter (which would collapse
   onto y=1) we show the cumulative curve: fraction of targets with output rank <= k.
C  privileging: swapping along each component; bar = top-5 rate, dots = categories.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
out = json.load(open(f"{HERE}/results/results.json"))


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return p, max(0.0, c - h), min(1.0, c + h)


fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

# ---- A: Spearman at three layers, bar = mean, dots = categories
ax = axes[0]
by_layer = defaultdict(list)
for x in out["corr"]:
    by_layer[x["layer"]].append(x["rho"])
show = [4, 9, 10]
rng = np.random.default_rng(0)
for i, L in enumerate(show):
    ys = by_layer[L]
    ax.bar(i, np.mean(ys), color="tab:blue", alpha=0.55, zorder=1)
    ax.scatter(i + rng.uniform(-0.22, 0.22, len(ys)), ys, s=14, color="k", alpha=0.55, zorder=3)
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(range(len(show))); ax.set_xticklabels([f"L{L}" for L in show])
ax.set_xlabel("layer"); ax.set_ylabel("Spearman ρ, lens vs output logits of 10 candidates")
ax.set_title("A  the lens ranks the candidates as the output does", loc="left", fontsize=10)

# ---- B: paired output rank, no-swap vs after-swap, one line per candidate (paper Fig. 6)
ax = axes[1]
sub = [x for x in out["swap"] if x["gate"] and x["before"] >= 11]
rngB = np.random.default_rng(1)
jitter = rngB.uniform(-0.07, 0.07, len(sub))
for x, dx in zip(sub, jitter):
    ax.plot([0 + dx, 1 + dx], [x["before"], x["after"]], color="gray", lw=0.6, alpha=0.4, zorder=1)
ax.scatter(jitter, [x["before"] for x in sub], s=16, color="gray", zorder=3, label="no swap")
ax.scatter(1 + jitter, [x["after"] for x in sub], s=16, color="tab:blue", zorder=3, label="after swap")
ax.axhline(5, color="k", ls="--", lw=0.8); ax.text(0.02, 4, "top-5", fontsize=8)
ax.set_yscale("log"); ax.set_ylim(max(x["before"] for x in sub) * 1.3, 0.7)
ax.set_xlim(-0.3, 1.3); ax.set_xticks([0, 1]); ax.set_xticklabels(["no swap", "after swap"])
ax.set_ylabel(f"target candidate's output rank  (n={len(sub)})")
ax.legend(fontsize=8, loc="center right")
top5 = np.mean([x["after"] <= 5 for x in sub])
ax.set_title(f"B  the swap redirects the report ({top5:.0%} reach the top-5)", loc="left", fontsize=10)

# ---- C: privileging, bar = top-5 rate, dots = categories
ax = axes[2]
conds = [("lens", "J-lens\nvectors", "tab:blue"), ("jpart", "J-space\ncomponent", "tab:blue"),
         ("nonj", "non-J-space\ncomponent", "tab:orange"), ("nonj_clamp", "non-J, J coords\nclamped to clean", "tab:orange")]
sub_all = [x for x in out["privilege"] if x["gate"] and x["before"] >= 11]
cats = sorted({x["cat"] for x in sub_all})
for i, (cond, lab, color) in enumerate(conds):
    s = [x for x in sub_all if x["cond"] == cond]
    p, lo, hi = wilson(sum(x["after"] <= 5 for x in s), len(s))
    ax.bar(i, p, color=color, alpha=0.6, zorder=1)
    ax.errorbar(i, p, yerr=[[max(0.0, p - lo)], [max(0.0, hi - p)]], color="k", capsize=4, zorder=2)
    for j, cat in enumerate(cats):
        cs = [x for x in s if x["cat"] == cat]
        ax.scatter(i + (j - len(cats) / 2) * 0.05, np.mean([x["after"] <= 5 for x in cs]), s=12, color="k", alpha=0.55, zorder=3)
    ax.text(i, hi + 0.03, f"{p:.0%}", ha="center", fontsize=9)
ax.set_xticks(range(len(conds))); ax.set_xticklabels([c[1] for c in conds], fontsize=8)
ax.set_ylim(0, 1.12); ax.set_ylabel("target reaches top-5 (bar = mean, dots = categories)")
ax.set_title(f"C  only the J-space component carries the effect (n={len(sub_all) // len(conds)})", loc="left", fontsize=10)

fig.suptitle("Criterion 1 — verbal report — gpt2-small", y=1.03)
fig.tight_layout()
os.makedirs(f"{HERE}/figures", exist_ok=True)
fig.savefig(f"{HERE}/figures/criterion1.png", dpi=160, bbox_inches="tight")
print("wrote figures/criterion1.png")
