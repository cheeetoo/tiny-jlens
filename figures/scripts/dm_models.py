"""Directed modulation in every model, at the paper's score: the share of trials on which a tracked token
(a member of the category, or the answer to the sum) is on top of the J-lens at some band layer and some
token of the copied sentence, by instruction.  In the style of the paper's Fig. 65: a bar is the mean over
phrasings, a dot one phrasing, and the dotted line the rate with no instruction.

dm_section.png (the post's directed-modulation section): categories, GPT-2 small, Gemma 3 1B and 1B-it, and
Claude Haiku 4.5.  dm_models.png (Appendix E): categories and mental arithmetic, every model.  GPT-2 small
and the four Gemma 3 models are on the paper's prompt (plain text for the base models, the chat format for
the instruction-tuned ones), each read over its band (dm_data.BANDS["ours"]); Claude is the paper's
released data.

Run from the repo root:  python figures/scripts/dm_models.py   -> dm_section.png, dm_models.png
"""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dm_data as D  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1]
BG, GRID, INK, MUTE, AXIS = "#fbfbf9", "#e7e4dd", "#222222", "#6b6760", "#b9b4ab"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "font.size": 11})
CONDS = [("focus", "think about", "#b22222"), ("mention", "mention", "#d9822b"),
         ("negated-think", "don't think about", "#7a7670"), ("dismissal", "ignore", "#3d85c6")]
NONE = (0, (1, 1.5))                                                # the no-instruction line's dash pattern
# label: (model, frame of the category run, frame of the math run)
OURS = {"GPT-2 small": ("gpt2", "human", "human"),
        "Gemma 3 270M": ("gemma-270m", "human", "human"), "Gemma 3 270M-it": ("gemma-270m-it", "paper", "paper"),
        "Gemma 3 1B": ("gemma-1b", "human", "human_math"), "Gemma 3 1B-it": ("gemma-1b-it", "paper", "paper_math")}
FAMILIES = {"topic": "Category instance (top-1)", "math": "Math expression (top-1)"}
claude, cnames = D.claude()


def data(model, fam):
    """{cond: (mean over phrasings, [rate per phrasing])}, with the no-instruction rate as "baseline"."""
    if model.startswith("Claude"):
        r = claude[fam][cnames.index(model.replace("Claude ", ""))]
    else:
        m, f_topic, f_math = OURS[model]
        r = D.Grid(m, f_topic if fam == "topic" else f_math).rates(fam, "ours", k=1)
    return {c: (r[c][0], list(r[c][1].values())) for c in ["baseline"] + [c for c, _, _ in CONDS]}


def panel(ax, models, fam, w=0.2, numbers=True, dot=22):
    for gi, model in enumerate(models):
        d = data(model, fam)
        base = d["baseline"][0]
        ax.plot([gi - 2 * w, gi + 2 * w], [base, base], ls=NONE, color=INK, lw=1, zorder=4)
        for i, (c, _, color) in enumerate(CONDS):
            x = gi + (i - 1.5) * w
            mean, per = d[c]
            ax.bar(x, mean, w * 0.86, color=color, alpha=0.4, lw=0)
            per = sorted(per, reverse=True)                         # phrasings high to low, across the bar
            xs = [x + w * 0.3 * (2 * j / (len(per) - 1) - 1) for j in range(len(per))]
            ax.scatter(xs, per, s=dot, color=color, edgecolor="white", linewidth=0.7, zorder=3)
            if numbers:
                v = 100 * mean
                ax.text(x, max(per + [mean]) + 0.025, f"{v:.0f}" if v >= 1 or v == 0 else f"{v:.1f}",
                        ha="center", va="bottom", fontsize=8.5, color=MUTE)
    ax.set_xticks(range(len(models)), models)
    ax.set_xlim(-0.55, len(models) - 0.45)
    ax.set_ylim(0, 1.06)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel(FAMILIES[fam], color=INK)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTE, length=0)
    ax.tick_params(axis="x", labelsize=11, labelcolor=INK)


def legend(fig):
    h = [Line2D([], [], marker="s", ls="", color=c, markersize=9, label=l) for _, l, c in CONDS]
    h.append(Line2D([], [], ls=NONE, color=INK, lw=1.2, label="no instruction"))
    fig.legend(handles=h, loc="lower center", ncol=len(h), frameon=False, bbox_to_anchor=(0.5, 0),
               handlelength=1.4, columnspacing=1.6, fontsize=10.5, handletextpad=0.5)


fig, ax = plt.subplots(figsize=(9.6, 4.4))
panel(ax, ["GPT-2 small", "Gemma 3 1B", "Gemma 3 1B-it", "Claude Haiku 4.5"], "topic")
legend(fig)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(OUT / "dm_section.png", dpi=220)
print("saved", OUT / "dm_section.png")

ALL = list(OURS) + [f"Claude {n}" for n in cnames]
fig, axes = plt.subplots(2, 1, figsize=(13, 7.6), sharex=True)
for ax, fam in zip(axes, FAMILIES):
    panel(ax, ALL, fam, numbers=False, dot=16)
    ax.axvline(len(OURS) - 0.5, color=AXIS, lw=1)
axes[1].set_xticklabels([m.replace("GPT-2 ", "GPT-2\n").replace("Gemma 3 ", "Gemma 3\n").replace(" 4.5", "\n4.5")
                         for m in ALL])
legend(fig)
fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=2)
fig.savefig(OUT / "dm_models.png", dpi=200)
print("saved", OUT / "dm_models.png")

for model in OURS:
    for fam in FAMILIES:
        d = data(model, fam)
        print(model, fam, " / ".join(f"{100 * d[c][0]:.1f}" for c in ["baseline"] + [c for c, _, _ in CONDS]))
