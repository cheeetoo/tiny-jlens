"""Directed modulation in every model, at the paper's score: the share of trials on which a tracked token
(a member of the category, or the answer to the sum) is on top of the J-lens at some band layer and some
token of the copied sentence, by instruction (mean over phrasings, as in the paper's Figs. 10 and 65).

(a) holding a category in mind; (b) mental arithmetic.  GPT-2 small and the four Gemma 3 models on the
paper's prompt (plain text for the base models, the chat format for the instruction-tuned ones), each
read over its band (dm_data.BANDS["ours"]); Claude is the paper's released data.

Run from the repo root:  python post/figures/dm_models.py   -> dm_models.png
"""
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dm_data as D  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent
MAROON, SLATE, ORANGE, TEAL, GOLD, GRAY = "#800000", "#33658a", "#d45d00", "#3a7d6a", "#c99000", "#8a8580"
BG = "#fbfbf9"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.prop_cycle": cycler(color=[MAROON, SLATE, ORANGE, TEAL, GOLD, GRAY]),
})
CONDS = [("baseline", "no instruction", GRAY), ("focus", "think about", MAROON), ("mention", "mention", SLATE),
         ("negated-think", "don't think about", ORANGE), ("dismissal", "ignore", TEAL)]
# (label, model, frame of the category run, frame of the math run)
OURS = [("GPT-2\nsmall", "gpt2", "human", "human"),
        ("Gemma-3\n270m", "gemma-270m", "human", "human"),
        ("Gemma-3\n270m-it", "gemma-270m-it", "paper", "paper"),
        ("Gemma-3\n1b", "gemma-1b", "human", "human_math"),
        ("Gemma-3\n1b-it", "gemma-1b-it", "paper", "paper_math")]
FAMILIES = [("topic", "(a) Holding a category in mind"), ("math", "(b) Mental arithmetic")]


def ours(model, frame, fam):
    r = D.Grid(model, frame).rates(fam, "ours", k=1)
    return {c: r[c][0] for c, _, _ in CONDS}


claude, names = D.claude()
fig, axes = plt.subplots(2, 1, figsize=(10, 7.4), sharex=True)
for ax, (fam, title) in zip(axes, FAMILIES):
    rates = [ours(m, f_topic if fam == "topic" else f_math, fam) for _, m, f_topic, f_math in OURS]
    rates += [{c: claude[fam][i][c][0] for c, _, _ in CONDS} for i in range(len(names))]
    x, w = np.arange(len(rates)), 0.16
    for i, (c, label, color) in enumerate(CONDS):
        ax.bar(x + (i - 2) * w, [100 * r[c] for r in rates], w, color=color, label=label)
    ax.axvline(len(OURS) - 0.5, color=GRAY, lw=1)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Trials with a hit (%)")
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, *_ in OURS] + [f"Claude\n{n.replace('Claude ', '')}" for n in names])
    for r, xi in zip(rates, x):                                  # the "think about" rate, written above its bar
        v = 100 * r["focus"]
        ax.text(xi - w, v + 1.5, f"{v:.0f}" if v >= 1 else f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color=MAROON)
axes[1].legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.01, 0.97))
fig.tight_layout()
fig.savefig(OUT / "dm_models.png", dpi=200)
print("saved", OUT / "dm_models.png")
for (label, m, ft, fm) in OURS:
    for fam, f in (("topic", ft), ("math", fm)):
        r = ours(m, f, fam)
        print(label.replace("\n", " "), fam, " / ".join(f"{100 * r[c]:.1f}" for c, _, _ in CONDS))
