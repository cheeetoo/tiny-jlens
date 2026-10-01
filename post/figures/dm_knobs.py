"""Directed modulation in our two models, with the choices the paper's score fixes left open:
which layers are read, and how high a tracked token has to rank.

dm_knobs.png  One row per model (category family, the paper's prompt).
  (a) Each layer on its own: the share of trials with a tracked token on top of the lens at some
      token of the copied sentence.  "out" is the model's own next-token prediction.  The shaded
      layers are the band our other results use; the bracket marks the paper's band (38% to 92%
      of depth) in this model's layers.
  (b) The same with a looser score: a tracked token in the lens top 25.
  (c) The band as a whole (the shaded layers), as the rank a tracked token has to reach goes from
      1 (the paper's hit) to 100.

Run from the repo root:  python post/figures/dm_knobs.py
"""
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dm_data as D  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "dm"
OUT.mkdir(exist_ok=True)
MAROON, SLATE, ORANGE, TEAL, GOLD, GRAY = "#800000", "#33658a", "#d45d00", "#3a7d6a", "#c99000", "#8a8580"
BG = "#fbfbf9"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.prop_cycle": cycler(color=[MAROON, SLATE, ORANGE, TEAL, GOLD, GRAY]),
})
COND_COLOR = {"focus": MAROON, "mention": ORANGE, "negated-think": GRAY, "dismissal": SLATE, "baseline": "black"}
KS = [1, 2, 5, 10, 25, 50, 100]
FAMILY = sys.argv[1] if len(sys.argv) > 1 else "topic"


def cond_rate(g, hit, cond):
    """The mean over phrasings of the hit rate, as the paper averages."""
    m = (g.family == FAMILY) & (g.cond == cond)
    return np.mean([hit[m & (g.phrasing == p)].mean() for p in dict.fromkeys(g.phrasing[m])])


models = ["gpt2", "qwen"]
fig, axes = plt.subplots(len(models), 3, figsize=(14, 3.7 * len(models)))
for i, model in enumerate(models):
    g = D.Grid(model)
    n = len(g.layers)                                   # lens layers, then the output
    per_layer = g.ranks.min(axis=2)                     # [trial, layer]
    lo, hi = D.BANDS[model]["ours"]
    plo, phi = D.BANDS[model]["paper"]
    for j, k in enumerate((1, 25)):
        ax = axes[i, j]
        ax.axvspan(lo - 0.5, hi + 0.5, color=GRAY, alpha=0.15, lw=0)
        for cond in ["baseline"] + D.CONDS:
            ax.plot(range(n), [cond_rate(g, per_layer[:, L] <= k, cond) for L in range(n)], "o-", ms=3,
                    color=COND_COLOR[cond], lw=1 if cond == "baseline" else 1.5,
                    ls="--" if cond == "baseline" else "-", label=D.COND_LABEL[cond])
        top = ax.get_ylim()[1]
        ax.plot([plo, phi], [top, top], color="black", lw=2, solid_capstyle="butt", clip_on=False)
        step = 1 if n <= 13 else 2
        shown = [L for L in range(0, n - 1, step) if L < n - step]          # keep the last label clear of "out"
        ax.set_xticks(shown + [n - 1], [str(L) for L in shown] + ["out"])
        ax.set_xlabel("Layer")
        ax.set_ylabel("Fraction of trials")
        ax.set_title(f"({'ab'[j]}) A tracked token {'on top of the lens' if k == 1 else 'in the lens top 25'}, "
                     f"layer by layer\n{D.MODEL_LABEL[model]}", fontsize=10)
    axes[i, 0].legend(frameon=False, fontsize=8)
    ax = axes[i, 2]
    best = g.best("ours")
    for cond in ["baseline"] + D.CONDS:
        ax.plot(KS, [cond_rate(g, best <= k, cond) for k in KS], "o-", ms=3, color=COND_COLOR[cond],
                lw=1 if cond == "baseline" else 1.5, ls="--" if cond == "baseline" else "-")
    ax.set_xscale("log")
    ax.set_xticks(KS, [str(k) for k in KS])
    ax.set_xlabel("Rank a tracked token has to reach")
    ax.set_ylabel("Fraction of trials")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"(c) Anywhere in layers {lo} to {hi}, by rank threshold\n{D.MODEL_LABEL[model]}", fontsize=10)
fig.tight_layout()
name = "dm_knobs.png" if FAMILY == "topic" else f"dm_knobs_{FAMILY}.png"
fig.savefig(OUT / name, dpi=200)
print("saved", OUT / name)
