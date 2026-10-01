"""Figure 3: the injected-thought test, drawn like the right panel of the paper's Fig. 7.

The median (line) and quartiles (band) of the injected word's reciprocal rank at the open quote,
where the report goes, and at every other position of the reply, pooled over positions and
concepts, against the per-layer steering strength. (a) Claude Sonnet 4.5, the paper's released
data; (b) GPT-2 small, the one-line frame with the paper's Fig. 7 reply ('... about the word "'),
the centered J-lens vector of the word's bare token, strengths up to 0.5 (the table in the post's
Appendix C.1 goes to 1.0).

Reads ref/paper-data/verbal-introspection.json and
results/control/gpt2/introspect_researcher_centered_surface_word.json.  Run from the repo root:
    python post/figures/introspect.py
"""
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parent
MAROON, SLATE, ORANGE, TEAL, GOLD, GRAY = "#800000", "#33658a", "#d45d00", "#3a7d6a", "#c99000", "#8a8580"
BG = "#fbfbf9"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.prop_cycle": cycler(color=[MAROON, SLATE, ORANGE, TEAL, GOLD, GRAY]),
})
FLOOR, GPT2_MAX = 1e-4, 0.5
QS = (25, 50, 75)

c = json.load(open(ROOT / "ref/paper-data/verbal-introspection.json"))["curve"]
claude = {"s": np.array(c["s"]),
          "report": np.array([c["slot"][f"q{q}"] for q in QS]),
          "other": np.array([c["other"][f"q{q}"] for q in QS])}

r = json.load(open(ROOT / "results/control/gpt2/introspect_researcher_centered_surface_word.json"))
S = [s for s in r["strengths"] if s <= GPT2_MAX]
rows = {s: [x for x in r["rows"] if x["strength"] == s] for s in S}
gpt2 = {"s": np.array(S),
        "report": np.array([np.percentile([1 / x["report"] for x in rows[s]], QS) for s in S]).T,
        "other": np.array([np.percentile([1 / v for x in rows[s] for v in x["other"]], QS) for s in S]).T}

SERIES = [("report", MAROON, "At the open quote (the report)"),
          ("other", GRAY, "Every other position of the reply")]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
for ax, d, title in [(axes[0], claude, "(a) Claude Sonnet 4.5"),
                     (axes[1], gpt2, "(b) GPT-2 small")]:
    for key, col, label in SERIES:
        lo, med, hi = np.maximum(d[key], FLOOR)
        ax.fill_between(d["s"], lo, hi, color=col, alpha=0.2, lw=0)
        ax.plot(d["s"], med, "o-", ms=3, color=col, label=label)
    ax.set_yscale("log")
    ax.set_ylim(FLOOR, 1.6)
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1], ["0.0001", "0.001", "0.01", "0.1", "1"])
    ax.set_xlabel("Steering strength (per layer)")
    ax.set_title(title)
axes[0].set_ylabel("Median reciprocal rank of the injected word")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "fig3_introspect.png", dpi=200)
print("saved", OUT / "fig3_introspect.png")
