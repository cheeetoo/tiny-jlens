"""Figure 4: the injected-thought test.  Fraction of concepts in the model's top-10 predictions at the
report position vs. anywhere else in the model's line, by injection strength.

Reads results/followups/introspect.json (GPT-2 small) and, if present,
results/qwen_control/introspect.json (Qwen3-1.7B).  Run from the repo root:
    python post/figures/introspect.py
"""
import json
import pathlib

import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parent
MAROON, SLATE, GRAY = "#800000", "#33658a", "#8a8580"
BG = "#fbfbf9"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG})

panels = [("GPT-2 small", ROOT / "results/followups/introspect.json", "report_rank", "other_best_rank")]
q = ROOT / "results/qwen_control/introspect.json"
if q.exists():
    panels.append(("Qwen3-1.7B (instruction-tuned)", q, "report", "other"))

fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.8), squeeze=False)
for ax, (name, path, rk, ok) in zip(axes[0], panels):
    rows = json.load(open(path))["rows"]
    S = sorted({r["strength"] for r in rows})
    rep = [sum(r[rk] <= 10 for r in rows if r["strength"] == s) / sum(r["strength"] == s for r in rows) for s in S]
    oth = [sum(r[ok] <= 10 for r in rows if r["strength"] == s) / sum(r["strength"] == s for r in rows) for s in S]
    ax.plot(S, rep, "o-", color=MAROON, label='at the report ("about" and the open quote)')
    ax.plot(S, oth, "o-", color=SLATE, label="anywhere earlier in the model's line")
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xticks(S, [f"{s:g}" for s in S], fontsize=8)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Injection strength (× mean residual norm)")
    ax.set_ylabel("Concepts in the top-10 predictions")
    ax.set_title(name)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig4_introspect.png", dpi=200)
print("saved", OUT / "fig4_introspect.png")
