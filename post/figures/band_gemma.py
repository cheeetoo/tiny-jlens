"""How we chose each Gemma 3 model's band: the layer statistics of the post's Figure 8 (the paper's Fig 28)
and the CKA between the layers' J-lens dictionaries (the paper's Fig 27), one row per model.  Shaded (and
outlined on the CKA): the band, 6-11 at 270m and 11-16 at 1b, the middle CKA block, ending where
persistence peaks and the dictionary's dimensionality jumps (dm_data.BANDS["ours"]).

The dictionary statistics, (d) and (e), use the J-lens vectors with the final norm's gain folded into the
unembedding (the `_gain` keys of jl.band): without it, Gemma's dictionary is dominated by one direction
that the final norm suppresses.

Reads results/band/<model>/{fig28,cka}.json (`python -m jl.band --model <model> cka fig28`).
Run from the repo root:  python post/figures/band_gemma.py   -> band_gemma.png
"""
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dm_data as D  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parent
MAROON, SLATE, ORANGE, TEAL, GOLD, GRAY = "#800000", "#33658a", "#d45d00", "#3a7d6a", "#c99000", "#8a8580"
BG, CREAM = "#fbfbf9", "#f2e2b3"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.prop_cycle": cycler(color=[MAROON, SLATE, ORANGE, TEAL, GOLD, GRAY]),
})
RAMP = LinearSegmentedColormap.from_list("slr_ramp", [MAROON, ORANGE, GOLD])
HEAT = LinearSegmentedColormap.from_list("slr_heat", [MAROON, ORANGE, GOLD, CREAM])

MODELS = ["gemma-270m", "gemma-270m-it", "gemma-1b", "gemma-1b-it"]
PANELS = [  # key in fig28.json, legend title, y label, title
    ("topk", "k", "Fraction of positions", "(a) Model's top-1 token in lens top k"),
    ("kurtosis", "percentile", "Excess kurtosis", "(b) Kurtosis of the lens readout"),
    ("autocorr", "Δ", "Log-prob gain over shuffled (nats)", "(c) Persistence of the lens's top-1 token"),
    ("effdim_gain", "variance", "Fraction of dimensions", "(d) Dimensions needed for share of variance"),
]

fig, axes = plt.subplots(len(MODELS), 5, figsize=(22.5, 4 * len(MODELS)),
                         gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 0.9]))
for i, m in enumerate(MODELS):
    f28 = json.load(open(ROOT / f"results/band/{m}/fig28.json"))
    cka = np.array(json.load(open(ROOT / f"results/band/{m}/cka.json"))["linear_gain"])
    lo, hi = D.BANDS[m]["ours"]
    name = D.MODEL_LABEL[m]
    for j, (key, legend, ylabel, title) in enumerate(PANELS):
        ax = axes[i, j]
        series = list(f28[key].items())
        n = len(series[0][1])
        ax.axvspan(lo - 0.5, hi + 0.5, color=GRAY, alpha=0.15, lw=0)
        if key in ("kurtosis", "autocorr"):
            ax.axhline(0, color=GRAY, lw=1)
        for si, (label, y) in enumerate(series):
            ax.plot(range(n), y, "o-", ms=3, color=RAMP(si / (len(series) - 1)), label=label)
        ax.set_xticks(range(0, n, 2))
        ax.set_title(f"{title}\n{name}", fontsize=10)
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        if i == 0:
            ax.legend(title=legend, frameon=False, fontsize=8, title_fontsize=8, ncol=2)
    ax = axes[i, 4]
    im = ax.imshow(cka, cmap=HEAT, vmin=0, vmax=1, origin="lower")
    ax.add_patch(Rectangle((lo - 0.5, lo - 0.5), hi - lo + 1, hi - lo + 1, fill=False, ec=SLATE, lw=1.5))
    ax.set_title(f"(e) CKA between layers' J-lens vectors\n{name}", fontsize=10)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
# the same axis range down each column of line plots
for j in range(4):
    lo_y = min(axes[i, j].get_ylim()[0] for i in range(len(MODELS)))
    hi_y = max(axes[i, j].get_ylim()[1] for i in range(len(MODELS)))
    for i in range(len(MODELS)):
        axes[i, j].set_ylim(lo_y, hi_y)
fig.tight_layout()
fig.savefig(OUT / "band_gemma.png", dpi=150)
print("saved", OUT / "band_gemma.png")
