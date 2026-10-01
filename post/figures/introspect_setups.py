"""Figure 5 (appendix): the injected-thought test in every GPT-2 setup we report, centered vector.

Columns: the question (one line, or the paper's full prompt as a User/Assistant transcript). Rows:
the reply ending (the paper's Fig. 7 reply '... about the word "', or its default '... about "')
and which token's vector is injected (the bare token 'dog', as in the paper's concept list, or the
word-initial ' dog'). Each panel is drawn like Figure 3(b): the median and quartiles of the reciprocal
rank of the word's bare token (the paper's score) at the report and at every other position of the
reply, pooled. The top-left panel is the main setup.

Reads results/control/gpt2/introspect_{researcher,transcript}_centered_{surface,space}{,_word}.json.
Run from the repo root:
    python post/figures/introspect_setups.py
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
FLOOR = 1e-4
QS = (25, 50, 75)
COLS = [("researcher", "One-line question"), ("transcript", "The paper's prompt, as a transcript")]
ROWS = [("_word", "surface", 'Reply … about the word "\nvector of dog'),
        ("_word", "space", 'Reply … about the word "\nvector of " dog"'),
        ("", "surface", 'Reply … about "\nvector of dog'),
        ("", "space", 'Reply … about "\nvector of " dog"')]

fig, axes = plt.subplots(len(ROWS), len(COLS), figsize=(8.4, 11), sharex=True, sharey=True)
for i, (suffix, token, row_label) in enumerate(ROWS):
    for j, (frame, col_label) in enumerate(COLS):
        ax = axes[i, j]
        r = json.load(open(ROOT / f"results/control/gpt2/introspect_{frame}_centered_{token}{suffix}.json"))
        assert "report_forms" in r["rows"][0], "expects the bare-token scoring"
        S = r["strengths"]
        rows = {s: [x for x in r["rows"] if x["strength"] == s] for s in S}
        series = {"report": [[1 / x["report"] for x in rows[s]] for s in S],
                  "other": [[1 / v for x in rows[s] for v in x["other"]] for s in S]}
        for key, col, label in (("report", MAROON, "At the open quote (the report)"),
                                ("other", GRAY, "Every other position of the reply")):
            lo, med, hi = np.maximum(np.array([np.percentile(v, QS) for v in series[key]]).T, FLOOR)
            ax.fill_between(S, lo, hi, color=col, alpha=0.2, lw=0)
            ax.plot(S, med, "o-", ms=2.5, color=col, label=label)
        ax.set_yscale("log")
        ax.set_ylim(FLOOR, 1.6)
        ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1], ["0.0001", "0.001", "0.01", "0.1", "1"])
        if i == 0:
            ax.set_title(col_label, fontsize=10)
        if j == 0:
            ax.set_ylabel(row_label, fontsize=9.5)
axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
fig.supxlabel("Steering strength (per layer)", fontsize=10)
fig.supylabel("Median reciprocal rank of the injected word", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "fig5_introspect_setups.png", dpi=200)
print("saved", OUT / "fig5_introspect_setups.png")
