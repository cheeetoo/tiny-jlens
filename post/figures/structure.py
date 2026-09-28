"""Figure 4: the structural signatures in gpt2-small.

Reads results/band/{band,cka,mlp_gain,fig28}.json and results/followups/{ignition,lists}.json.
Run from the repo root:  python post/figures/structure.py
"""
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = pathlib.Path(__file__).resolve().parents[2]
R = ROOT / "results"
OUT = pathlib.Path(__file__).resolve().parent
MAROON, SLATE, ORANGE, TEAL, GOLD, GRAY = "#800000", "#33658a", "#d45d00", "#3a7d6a", "#c99000", "#8a8580"
BG = "#fbfbf9"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG})

band = json.load(open(R / "band/band.json"))
cka = json.load(open(R / "band/cka.json"))
mlp = json.load(open(R / "band/mlp_gain.json"))
f28 = json.load(open(R / "band/fig28.json"))
ign = json.load(open(R / "followups/ignition.json"))
lists = json.load(open(R / "followups/lists.json"))
L12 = list(range(12))


def shade(ax):
    ax.axvspan(6.5, 9.5, color=GRAY, alpha=0.15, lw=0)


fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))

ax = axes[0, 0]
st = band["stats"]
ax.plot(L12, [st[str(L)]["top1"] for L in L12], "o-", color=MAROON, label="top-1")
ax.plot(L12, [st[str(L)]["top10"] for L in L12], "o-", color=SLATE, label="top-10")
shade(ax)
ax.set_xlabel("Layer")
ax.set_ylabel("Fraction of positions")
ax.set_title("(a) J-lens agrees with the model's next token")
ax.legend(frameon=False)

ax = axes[0, 1]
ramp = LinearSegmentedColormap.from_list("ramp", [MAROON, ORANGE, GOLD])
offs = list(f28["autocorr"])
for i, d in enumerate(offs):
    ax.plot(L12, f28["autocorr"][d], "o-", ms=4, color=ramp(i / (len(offs) - 1)), label=d)
shade(ax)
ax.axhline(0, color=GRAY, lw=1)
ax.set_xlabel("Layer")
ax.set_ylabel("Log-prob of top-1 token from Δ positions\nearlier, minus shuffled baseline (nats)")
ax.set_title("(b) J-lens content persists, but only briefly")
ax.legend(title="Δ", frameon=False, fontsize=9, ncol=2)

ax = axes[0, 2]
M = np.array(cka["mean_cca_r50"])
im = ax.imshow(M, cmap="Greys_r", vmin=0.4, vmax=1.0, origin="upper")
ax.set_xticks(L12)
ax.set_yticks(L12)
ax.set_xlabel("Layer")
ax.set_ylabel("Layer")
ax.set_title("(c) Similarity of layers' J-lens vectors\n(mean CCA, top 50 dims)")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax = axes[1, 0]
pl = ign["per_layer"]
ax.plot([r["layer"] for r in pl], [r["width_full"] for r in pl], "o-", color=MAROON, label="full residual")
ax.plot([r["layer"] for r in pl], [r["width_J"] for r in pl], "o-", color=SLATE, label="J-lens span")
ax.axhline(0.8, color=GRAY, ls="--", lw=1)
ax.text(0.2, 0.82, "a linear blend would give 0.8", color=GRAY, fontsize=9, va="bottom")
shade(ax)
ax.set_ylim(0, 1)
ax.set_xlabel("Layer")
ax.set_ylabel("Mixture weight needed to go\nfrom 10% to 90% of the way")
ax.set_title("(d) Ambiguous input: no sharp switch")
ax.legend(frameon=False, loc="lower left")

ax = axes[1, 1]
Ls = list(range(11))
ax.plot(Ls, [mlp[str(L)]["centered"] for L in Ls], "o-", color=SLATE, label="J-lens vectors (centered)")
ax.plot(Ls, [mlp[str(L)]["raw"] for L in Ls], "o-", color=MAROON, label="J-lens vectors (raw)")
ax.plot(Ls, [mlp[str(L)]["neuron"] for L in Ls], "o-", color=GRAY, label="MLP neuron output directions")
ax.axhline(1, color=GRAY, lw=1)
shade(ax)
ax.set_xlabel("Layer of the direction (MLP of the next block)")
ax.set_ylabel("MLP output norm / random direction")
ax.set_title("(e) MLP gain (paper: ~10× in Claude's band)")
ax.legend(frameon=False, fontsize=9)

ax = axes[1, 2]
n = len(lists["related"][0]["read"])
x = np.arange(1, n + 1)
rel = np.mean([r["read"] for r in lists["related"]], 0)
rel_all = np.mean([r["all"] for r in lists["related"]], 0)
unr = np.mean([r["read"] for r in lists["unrelated"]], 0)
ax.plot(x, rel, color=MAROON, label="one category: words read so far")
ax.plot(x, rel_all, color=MAROON, ls="--", label="one category: all 80 list words")
ax.plot(x, unr, color=SLATE, label="unrelated words: read so far")
ax.set_xlabel("Words read")
ax.set_ylabel("List words in the J-lens top 25")
ax.set_title("(f) How many list words the J-space holds")
ax.set_ylim(0, 21)
ax.legend(frameon=False, fontsize=9, loc="upper left")

for a in axes.flat:
    if a is not axes[0, 2]:
        a.grid(False)
fig.tight_layout()
fig.savefig(OUT / "fig4_structure.png", dpi=200)
print("saved", OUT / "fig4_structure.png")
