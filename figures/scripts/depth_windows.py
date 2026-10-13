"""The paper's Fig. 15 (right), Claude Sonnet 4.5 next to GPT-2 small: swap the intermediate (the
country) or the answer over a window of layers [lo, u]; y = log p(expected answer) - log p(original
answer) after the swap, mean and s.e. over the trials where the swap over the whole band works.
Windows start at the band's first layer (the paper's design) and end at an upper layer from the
band's start to the last layer, so that layers after the band are shown.  One random same-family
target per question, as the paper.  The data also holds windows starting two layers earlier
(layer 5), used only by the scratch version of the figure.

Run from the repository root:  PYTHONPATH=. python figures/scripts/depth_windows.py [--run]
--run recomputes results/c3_reasoning/depth_windows.json (a few minutes on CPU); otherwise it is read.
"""
import json
import pathlib
import random
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler

import jl

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "results/c3_reasoning/depth_windows.json"
CLAUDE = ROOT / "ref/paper-data/multihop-swap-success.json"
BAND_LO, BAND_HI, TOP, EARLY = min(jl.BAND), max(jl.BAND), 11, min(jl.BAND) - 2
STARTS = {BAND_LO: range(BAND_LO, TOP + 1), EARLY: range(EARLY, TOP + 1)}


def run():
    import torch
    from jl import coord_swap_edits
    torch.set_grad_enabled(False)
    lm = jl.Lensed()
    items = json.load(open(ROOT / "results/c3_reasoning/prompts.json"))
    for it in items:
        it["ids"] = lm.encode(it["prompt"].rstrip())
        it["clean"] = lm.residuals(it["ids"], list(range(12)))
        it["lg"] = lm.logits(it["ids"])[-1]
        it["s"], it["a"] = lm.tid(" " + it["intermediate"]), lm.tid(" " + it["answer"])
    rank = lambda lg, t: int(jl.ranks_of(lg, [t])[0])
    rng, rows = random.Random(0), []
    for it in items:
        ps = [p for p in items if p["family"] == it["family"] and p["intermediate"] != it["intermediate"]
              and p["answer"] != it["answer"] and rank(it["lg"], p["a"]) > 10]
        if not ps:
            continue
        p = rng.choice(ps)
        lp0 = it["lg"].log_softmax(-1)
        row = dict(item=it["intermediate"], family=it["family"], target=p["intermediate"],
                   clean=float(lp0[p["a"]] - lp0[it["a"]]), w={})
        for lo, ups in STARTS.items():
            for u in ups:
                for kind, (a, b) in (("interm", (it["s"], p["s"])), ("answer", (it["a"], p["a"]))):
                    lg = lm.logits(it["ids"], coord_swap_edits(lm, it["ids"], a, b, list(range(lo, u + 1)),
                                                               clean=it["clean"]))[-1]
                    lp = lg.log_softmax(-1)
                    row["w"][f"{kind}:{lo}:{u}"] = [float(lp[p["a"]] - lp[it["a"]]), int(lg.argmax()) == p["a"]]
        rows.append(row)
        print(it["intermediate"], "->", p["intermediate"], flush=True)
    json.dump(dict(band=jl.BAND, rows=rows), open(DATA, "w"))


if "--run" in sys.argv:
    run()

MAROON, SLATE, GRAY = "#800000", "#33658a", "#8a8580"
BG = "#fbfbf9"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                     "axes.prop_cycle": cycler(color=[MAROON, SLATE])})

claude = json.load(open(CLAUDE))["onset"]
rows = json.load(open(DATA))["rows"]
ok = [r for r in rows if r["w"][f"interm:{BAND_LO}:{BAND_HI}"][1]]
NL = 11  # GPT-2 layers 0..11 -> 0..100% of depth


def curve(kind, lo=BAND_LO, last=TOP):
    ups = [u for u in STARTS[lo] if u <= last]
    v = [[r["w"][f"{kind}:{lo}:{u}"][0] for r in ok] for u in ups]
    return ups, [st.mean(x) for x in v], [st.stdev(x) / len(x) ** 0.5 for x in v]


def draw(ax, x, m, s, color, label=None, **kw):
    ax.plot(x, m, "o-", ms=3, color=color, label=label, **kw)
    ax.fill_between(x, [a - b for a, b in zip(m, s)], [a + b for a, b in zip(m, s)], color=color, alpha=0.2, lw=0)


def figure(path, layer_numbers=False):
    """The repo version stops at layer 10: the swap at layer 11 (the output layer, where the lens is
    the identity) writes the answer straight into the logits, which swamps the y axis."""
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), sharey=True, gridspec_kw=dict(width_ratios=[3, 2]))
    ax = axes[0]
    x = [l * 100 / 24 for l in claude["l"]]
    ax.axvspan(min(x), max(x), color=GRAY, alpha=0.15, lw=0)
    for key, color, name in (("intermediate", MAROON, "intermediate"), ("target", SLATE, "answer")):
        draw(ax, x, claude["mean"][key], claude["se"][key], color, f"Swap the {name}")
    ax.set_title("(a) Claude Sonnet 4.5 (paper's data)", fontsize=10)
    ax.set_xlabel("Swap window upper layer (percent of depth)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    unit = lambda u: u
    ax.axvspan(unit(BAND_LO), unit(BAND_HI), color=GRAY, alpha=0.15, lw=0)
    for kind, color, name in (("interm", MAROON, "intermediate"), ("answer", SLATE, "answer")):
        ups, m, s = curve(kind, last=TOP if layer_numbers else TOP - 1)
        draw(ax, [unit(u) for u in ups], m, s, color, f"Swap the {name}")
        if layer_numbers:
            ups, m, s = curve(kind, EARLY, last=TOP)
            ax.plot([unit(u) for u in ups], m, "o--", ms=3, color=color, alpha=0.5)
    if layer_numbers:
        ax.plot([], [], "k-", label=f"window starts at layer {BAND_LO} (band start)")
        ax.plot([], [], "k--", alpha=0.5, label=f"window starts at layer {EARLY}")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("(b) GPT-2 small", fontsize=10)
    ax.set_xlabel("Swap window upper layer")
    ax.set_xticks(range(BAND_LO if not layer_numbers else EARLY, (TOP if layer_numbers else TOP - 1) + 1))
    for a in axes:
        a.axhline(0, color=GRAY, lw=1)
    axes[0].set_ylabel("Log-prob, expected \u2212 original")
    fig.tight_layout()
    fig.savefig(path, dpi=200)


if __name__ == "__main__":
    figure(ROOT / "figures/depth_windows.png")
    if "--scratch" in sys.argv:   # actual layer numbers and the layer-5 windows, kept out of the repo
        figure(ROOT / "scratch/post_review/depth_windows_layers.png", layer_numbers=True)
    print(f"trials: {len(ok)} of {len(rows)}")
