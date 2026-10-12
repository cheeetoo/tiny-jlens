"""Flexible generalization, GPT-2 small next to Claude Sonnet 4.5 on the paper's 192 swaps.

flexgen_swaps.png: the testable swaps (GPT-2 knows the target argument's answer and doesn't already
give it; the two answers differ and the target's isn't in the worked examples), one square per
swap, colored by which model the coordinate swap works for at alpha = 1 (target answer at top-1).
Claude's outcomes are the paper's released per-swap grid (its Fig. 68).

flexgen_loading.png: the paper's Fig. 19 (right) for both models: per function, workspace loading
against the swap effect at alpha = 1 (change in the target answer's log-prob minus change in the
spontaneous answer's), over all 12 swaps.  Claude's from the paper's released data; loading is the
mean of the cosine at the argument and at the readout in both, as in the paper's figure.

Run from the repository root:  PYTHONPATH=. python post/figures/flexgen.py
Reads results/c4_generalization/results.json (python -m jl.c4_generalization writes it).
"""
import json
import pathlib
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.patches import Patch, Rectangle

from jl.c4_generalization import PAPER_SUMMARY, claude_grid, key
from jl.stats import pearson

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/c4_generalization/results.json"

MAROON, SLATE, ORANGE, TEAL, GOLD, GRAY = "#800000", "#33658a", "#d45d00", "#3a7d6a", "#c99000", "#8a8580"
BG = "#fbfbf9"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                     "axes.prop_cycle": cycler(color=[MAROON, SLATE, ORANGE, TEAL, GOLD, GRAY])})

CATS = ["countries", "months", "animals", "numbers"]
CAT_COLOR = dict(zip(CATS, [MAROON, SLATE, ORANGE, TEAL]))
NAMES = {"capital": "capital", "language": "language", "continent": "continent", "currency": "currency",
         "season": "season", "number": "month number", "holiday": "holiday", "next_month": "month after",
         "habitat": "habitat", "legs": "legs", "class": "class", "group": "group name",
         "double": "double", "square": "square", "successor": "number after", "first_letter": "first letter"}

out = json.load(open(RESULTS))
claude = claude_grid()


def testable(r):
    return r["distinct"] and not r["echo"] and r["target_gated"] and r["before"] > 1


def swaps_figure(path):
    rows = [r for r in out["swap"] if testable(r)]
    fns = list(dict.fromkeys((r["category"], r["function"]) for r in rows))
    fns.sort(key=lambda f: CATS.index(f[0]))
    kinds = [("both", MAROON, 1.0), ("Claude only", SLATE, 1.0), ("GPT-2 only", ORANGE, 1.0),
             ("neither", GRAY, 0.25)]

    def kind(r):
        g, c = r["coord"]["1.0"]["hit"], claude[key(r)]
        return "both" if g and c else "Claude only" if c else "GPT-2 only" if g else "neither"

    fig, ax = plt.subplots(figsize=(6.8, 2.9))
    for y, f in enumerate(fns):
        rs = sorted((r for r in rows if (r["category"], r["function"]) == f),
                    key=lambda r: [k[0] for k in kinds].index(kind(r)))
        for x, r in enumerate(rs):
            _, color, alpha = next(k for k in kinds if k[0] == kind(r))
            ax.add_patch(Rectangle((x + 0.08, -y - 0.42), 0.84, 0.84, color=color, alpha=alpha, lw=0))
        g = sum(r["coord"]["1.0"]["hit"] for r in rs)
        c = sum(claude[key(r)] for r in rs)
        ax.text(12.4, -y, f"GPT-2 {g}, Claude {c} of {len(rs)}", va="center", fontsize=8)
    ax.set_yticks([-y for y in range(len(fns))], [f"{f[0]}: {NAMES[f[1]]}" for f in fns])
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 17.2)
    ax.set_ylim(-len(fns) + 0.4, 0.6)
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)
    n_g = sum(r["coord"]["1.0"]["hit"] for r in rows)
    n_c = sum(claude[key(r)] for r in rows)
    ax.set_title(f"The {len(rows)} swaps GPT-2 can be scored on: GPT-2 {n_g}, Claude {n_c}", fontsize=10)
    ax.legend(handles=[Patch(color=c, alpha=a, lw=0, label=k) for k, c, a in kinds], frameon=False,
              fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout()
    fig.savefig(path, dpi=200)


# Label offsets (points) and alignment where the default (right of the dot) collides.
NUDGE_GPT2 = {"first_letter": (0, 7, "center"), "number": (0, -8, "center"), "double": (-4, 7, "right"),
              "holiday": (4, 6, "left"), "square": (4, -6, "left"), "season": (4, 4, "left")}
NUDGE_CLAUDE = {"continent": (-4, 2, "right"), "group": (4, 3, "left"), "class": (4, -4, "left"),
                "holiday": (4, 5, "left"), "season": (4, -2, "left"), "square": (-4, 2, "right"),
                "first_letter": (0, 7, "center"), "double": (4, -5, "left")}


def loading_figure(path):
    paper = {(p["cat"], p["func"]): p for p in json.load(open(PAPER_SUMMARY))["pts"]}
    fns = list(paper)
    load = {f: mean(x["loading"] for x in out["loading"] if (x["category"], x["function"]) == f) for f in fns}
    eff = {f: mean(r["coord"]["1.0"]["effect"] for r in out["swap"] if (r["category"], r["function"]) == f)
           for f in fns}
    # csum is the cosine at the argument plus the cosine at the readout; the paper's figure plots
    # their mean (x_cos in its data.json), as does GPT-2's loading.
    panels = [("(a) GPT-2 small", load, eff, NUDGE_GPT2),
              ("(b) Claude Sonnet 4.5 (paper's data)", {f: paper[f]["csum"] / 2 for f in fns},
               {f: paper[f]["dlpc"] for f in fns}, NUDGE_CLAUDE)]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), sharey=True)
    for ax, (title, x, y, nudge) in zip(axes, panels):
        r = pearson([x[f] for f in fns], [y[f] for f in fns])
        for f in fns:
            dx, dy, ha = nudge.get(f[1], (4, 2, "left"))
            ax.scatter(x[f], y[f], s=22, color=CAT_COLOR[f[0]], zorder=3)
            ax.annotate(NAMES[f[1]], (x[f], y[f]), xytext=(dx, dy), textcoords="offset points", fontsize=6.5,
                        color=CAT_COLOR[f[0]], ha=ha, va="center")
        ax.axhline(0, color=GRAY, lw=1)
        ax.set_title(f"{title}\nr = {r:.2f}", fontsize=10)
        ax.set_xlabel("Workspace loading (mean cosine)")
    axes[0].set_ylabel("Swap effect at α = 1 (nats)")
    axes[0].legend(handles=[Patch(color=CAT_COLOR[c], label=c) for c in CATS], frameon=False, fontsize=8,
                   loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)


if __name__ == "__main__":
    swaps_figure(ROOT / "post/figures/flexgen_swaps.png")
    loading_figure(ROOT / "post/figures/flexgen_loading.png")
