"""Figure 3: J-space ablation at the paper's three strengths, and random directions at medium strength,
Claude Sonnet 4.5 (the paper's Fig. 22, ref/paper-data/ablation-strength.json) against GPT-2 small
(results/c5_selectivity, 1 direction).  GPT-2's random bar is the mean over five draws.

Run from the repository root:  PYTHONPATH=. python figures/scripts/selectivity.py
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "figures/selectivity.png"
CONDS = [("light", "#f7c190"), ("medium", "#ee7a40"), ("heavy", "#b3501f"), ("random (medium)", "#bbbbbb")]
BG, GRID, INK, MUTE = "#fbfbf9", "#e7e4dd", "#222222", "#6b6760"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "font.size": 11})

paper = {r["name"]: r for r in json.load(open(ROOT / "ref/paper-data/ablation-strength.json"))["rows"]}
bat = json.load(open(ROOT / "results/c5_selectivity/results.json"))["battery"]
gpt2 = {"mh_acc": next(r for r in bat["acc"] if r["task"] == "two_hop"), "pt_acc": bat["pretrain"]}


def values(key):
    g = gpt2[key]
    claude = [paper[s][key] for s in ("light", "medium", "heavy", "random")]
    ours = [g[f"{s}_J_k1"] for s in ("light", "medium", "heavy")] + [sum(g["medium_R_k1"]) / len(g["medium_R_k1"])]
    return {"Claude Sonnet 4.5": claude, "GPT-2 small": ours}


fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1), sharey=True)
w, gap = 0.2, 1.15
for ax, (key, title) in zip(axes, [("mh_acc", "Two-hop accuracy"), ("pt_acc", "Next-token top-1 match")]):
    for gi, (model, vals) in enumerate(values(key).items()):
        for i, ((_, color), v) in enumerate(zip(CONDS, vals)):
            x = gi * gap + (i - 1.5) * w
            ax.bar(x, v, w * 0.9, color=color)
            ax.text(x, v + 0.015, f"{v * 100:.0f}", ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_xticks([0, gap], list(values(key)))
    ax.set_xlim(-0.6, gap + 0.6)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12)
    ax.set_ylim(0, 1.06)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#b9b4ab")
    ax.tick_params(colors=MUTE, length=0)
    ax.tick_params(axis="x", labelsize=11.5, labelcolor=INK)
fig.legend(handles=[Patch(color=c, label=s) for s, c in CONDS], title="Ablation strength", loc="lower center",
           ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=1.1, handleheight=1.1, columnspacing=1.6,
           title_fontsize=11)
fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=3)
fig.savefig(OUT, dpi=220)
print("wrote", OUT)
