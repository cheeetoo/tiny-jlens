"""Figure 5 (appendix): the paper's Fig 28 layer statistics, Claude against gpt2-small.

Reads post/fig1/data/paper_layer_lines.json (Claude Sonnet 4.5: the data behind the paper's Fig 28,
from transformer-circuits.pub/2026/workspace/data/layer-lines/main.json) and results/band/fig28.json
(`python -m jl.band fig28`).
Run from the repo root:  python post/figures/band_stats.py
"""
import json
import pathlib

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parent
MAROON, ORANGE, GOLD, GRAY = "#800000", "#d45d00", "#c99000", "#8a8580"
BG = "#fbfbf9"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG})
ramp = LinearSegmentedColormap.from_list("ramp", [MAROON, ORANGE, GOLD])

paper = json.load(open(ROOT / "post/fig1/data/paper_layer_lines.json"))
ours = json.load(open(ROOT / "results/band/fig28.json"))
n = paper["n_layers"] - 1  # the paper labels its 25 layers 0-100
lo, hi = (100 * b / n for b in paper["band"])

PANELS = [  # key in fig28.json, legend title, y label, title
    ("topk", "k", "Fraction of positions", "(a) Model's top-1 token in lens top k"),
    ("kurtosis", "percentile", "Excess kurtosis", "(b) Kurtosis of the lens readout"),
    ("autocorr", "Δ", "Log-prob gain over shuffled (nats)", "(c) Persistence of the lens's top-1 token"),
    ("effdim", "variance", "Fraction of dimensions", "(d) Dimensions needed for share of variance"),
]

fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey="col")
for j, (key, legend, ylabel, title) in enumerate(PANELS):
    claude = [(s["label"], [100 * x / n for x in s["x"]], s["y"]) for s in paper["panels"][j]["series"]]
    gpt2 = [(k, list(range(12)), y) for k, y in ours[key].items()]
    for i, (model, series, band) in enumerate([("Claude Sonnet 4.5 (paper's data)", claude, (lo, hi)),
                                               ("GPT-2 small", gpt2, (6.5, 9.5))]):
        ax = axes[i, j]
        ax.axvspan(*band, color=GRAY, alpha=0.15, lw=0)
        if key in ("kurtosis", "autocorr"):
            ax.axhline(0, color=GRAY, lw=1)
        for si, (label, x, y) in enumerate(series):
            ax.plot(x, y, "o-", ms=3, color=ramp(si / (len(series) - 1)), label=label)
        ax.set_title(f"{title}\n{model}", fontsize=10)
        ax.set_xlabel("Layer (percent of depth)" if i == 0 else "Layer")
        ax.set_ylabel(ylabel)
        if i == 1:
            ax.set_xticks(range(12))
        ax.legend(title=legend, frameon=False, fontsize=8, title_fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "fig5_band_stats.png", dpi=200)
print("saved", OUT / "fig5_band_stats.png")
