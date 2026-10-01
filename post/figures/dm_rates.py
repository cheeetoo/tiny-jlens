"""Directed modulation, drawn like the paper's Fig. 10 and Fig. 65, with our two models added.

dm/dm_lines.png          Fig. 10: the hit rate under each kind of instruction, by model.  The
                         paper's figure has "think about" and "ignore"; this adds the bare mention
                         and "don't think about" (its Fig. 65 data) and, in gray, the rate with no
                         instruction.
fig6_modulation.png      Figure 6 of the post (Appendix C.2), the paper's Fig. 65: one dot per
                         instruction phrasing, bars for the mean over phrasings, one row per model
                         and one column per task family.  The dashed line is the rate with no
                         instruction.

A hit is the paper's: a tracked token at J-lens rank 1 at any layer of the band and any token of
the copied sentence.  Claude: the paper's released data (ref/paper-data/modulation-prompts.json).
Ours: the grid runs, on the paper's prompt, read over each model's band (dm_data.BANDS["ours"]).

Run from the repo root:  python post/figures/dm_rates.py
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
COND_COLOR = {"focus": MAROON, "mention": ORANGE, "negated-think": GRAY, "dismissal": SLATE}

claude, claude_models = D.claude()
ours = {m: D.Grid(m) for m in ("gpt2", "qwen")}
models = [D.MODEL_LABEL["gpt2"], D.MODEL_LABEL["qwen"]] + claude_models
# rates[family] = one {cond: (mean, {phrasing: rate})} per model, in the order of `models`
rates = {f: [ours["gpt2"].rates(f), ours["qwen"].rates(f)] + claude[f] for f in D.FAMILIES}

# ---------------------------------------------------------------- Fig. 10
fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
for ax, cond in zip(axes, D.CONDS):
    for fam, color in (("topic", MAROON), ("math", SLATE)):
        ax.plot(range(len(models)), [r[cond][0] for r in rates[fam]], "o-", color=color, label=D.FAMILY_LABEL[fam])
    ax.plot(range(len(models)), [max(rates[f][i]["baseline"][0] for f in D.FAMILIES) for i in range(len(models))],
            "o-", ms=3, color=GRAY, lw=1, label="No instruction (the higher of the two)")
    ax.axvline(1.5, color=GRAY, lw=1, ls=":")
    ax.set_xticks(range(len(models)), [m.replace(" ", "\n", 1).replace("-0.8B", "\n0.8B") for m in models], fontsize=8)
    ax.set_title(f'"{D.COND_LABEL[cond].capitalize()} X"' if cond != "mention" else "X is only mentioned", fontsize=10)
    ax.set_ylim(-0.03, 1.03)
axes[0].set_ylabel("Fraction of trials with a tracked\ntoken on top of the J-lens")
axes[-1].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "dm_lines.png", dpi=200)
print("saved", OUT / "dm_lines.png")

# ---------------------------------------------------------------- Fig. 65
fig, axes = plt.subplots(len(models), 2, figsize=(6.4, 10), sharex=True, sharey=True)
for i, model in enumerate(models):
    for j, fam in enumerate(D.FAMILIES):
        ax = axes[i, j]
        r = rates[fam][i]
        for x, cond in enumerate(D.CONDS):
            mean, dots = r[cond]
            ax.bar(x, mean, width=0.7, color=COND_COLOR[cond], alpha=0.3)
            ys = sorted(dots.values())
            ax.plot(x + np.linspace(-0.22, 0.22, len(ys)), ys, "o", ms=3, color=COND_COLOR[cond])
        ax.axhline(r["baseline"][0], color="black", lw=1, ls="--")
        ax.set_xticks(range(len(D.CONDS)), [D.COND_LABEL[c].replace(" about", "\nabout") for c in D.CONDS], fontsize=7.5)
        ax.set_xlim(-0.6, len(D.CONDS) - 0.4)
        ax.set_ylim(0, 1.03)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
        if i == 0:
            ax.set_title(D.FAMILY_LABEL[fam], fontsize=10)
        if j == 0:
            ax.set_ylabel(model, fontsize=10)
fig.supylabel("Trials with a tracked token on top of the J-lens", fontsize=10)
fig.tight_layout()
fig.savefig(OUT.parent / "fig6_modulation.png", dpi=200)
print("saved", OUT.parent / "fig6_modulation.png")
