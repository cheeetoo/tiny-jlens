"""Directed modulation in Gemma 3, base against instruction-tuned (post/DM.md).

dm/dm_gemma.png   One panel per model size and prompt frame.  In each, the base model and the
                  instruction-tuned one on the same tokens: bars for the mean over each condition's
                  phrasings, a dot per phrasing.  Categories, the paper's score (a member at J-lens
                  rank 1 at any layer of the paper's band and any token of the copied sentence).
                  Pass k=25 for a member in the top 25 instead (dm/dm_gemma_top25.png).

Run from the repo root:  python post/figures/dm_gemma_fig.py [k]
"""
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dm_data as D  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "dm"
MAROON, SLATE, ORANGE, TEAL, GOLD, GRAY = "#800000", "#33658a", "#d45d00", "#3a7d6a", "#c99000", "#8a8580"
BG = "#fbfbf9"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.prop_cycle": cycler(color=[MAROON, SLATE, ORANGE, TEAL, GOLD, GRAY]),
})
CONDS = ["baseline"] + D.CONDS
FRAMES = {"human": "the paper's prompt as plain text", "paper": "the paper's prompt as a chat"}
k = int(sys.argv[1]) if len(sys.argv) > 1 else 1

fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharey=True)
for i, size in enumerate(("270m", "1b")):
    for j, frame in enumerate(FRAMES):
        ax = axes[i, j]
        for s, (m, color, label) in enumerate(((f"gemma-{size}", SLATE, "base"), (f"gemma-{size}-it", MAROON, "instruction-tuned"))):
            if not (D.RESULTS / f"control/{m}/modulation_grid_{frame}.json").exists():
                ax.text(0.25 + 0.5 * s, 0.5, f"{label}:\nnot run", transform=ax.transAxes, ha="center", color=color, fontsize=8)
                continue
            r = D.Grid(m, frame).rates("topic", "paper", k=k)
            for x, cond in enumerate(CONDS):
                mean, dots = r[cond]
                xc = x + (s - 0.5) * 0.38
                ax.bar(xc, mean, width=0.36, color=color, alpha=0.3, label=label if x == 0 else None)
                ys = sorted(dots.values())
                ax.plot(xc + np.linspace(-0.12, 0.12, len(ys)), ys, "o", ms=2.5, color=color)
        ax.set_xticks(range(len(CONDS)), [D.COND_LABEL[c].replace(" about", "\nabout").replace("no ", "no\n") for c in CONDS],
                      fontsize=8)
        ax.set_xlim(-0.6, len(CONDS) - 0.4)
        ax.set_ylim(0, 1.03)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
        ax.set_title(f"Gemma-3-{size}, {FRAMES[frame]}", fontsize=10)
axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
what = "on top of the J-lens" if k == 1 else f"in the J-lens top {k}"
fig.supylabel(f"Trials with a category member {what}", fontsize=10)
fig.tight_layout()
name = "dm_gemma.png" if k == 1 else f"dm_gemma_top{k}.png"
fig.savefig(OUT / name, dpi=200)
print("saved", OUT / name)
