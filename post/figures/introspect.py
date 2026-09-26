"""Figure 3: the injected-thought test, in top predictions.

For each injection strength, the number of concepts for which the injected word is the model's top
next-token prediction (a) at the report ("about" and the open quote) and (b) at the very start of
the reply, before it says anything about a thought.  GPT-2 small (two-speaker transcript) and
Qwen3-1.7B (the paper's chat prompt).

Reads results/introspect_probs/{gpt2,qwen}.json.  Run from the repo root:
    python post/figures/introspect.py
"""
import json
import pathlib

import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parent
MAROON, SLATE = "#800000", "#33658a"
BG = "#fbfbf9"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG})


def start_positions(model, toks):
    """Positions whose next token begins the reply, before any mention of a thought."""
    if model == "gpt2":
        return [toks.index("Model"), toks.index(":"), toks.index(" Yes")]
    return [toks.index("</think>") + 1, toks.index("Yes")]


panels = [("gpt2", "GPT-2 small"), ("qwen", "Qwen3-1.7B (instruction-tuned)")]
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
for ax, (model, title) in zip(axes, panels):
    forced = json.load(open(ROOT / f"results/introspect_probs/{model}.json"))["forced"]
    toks, rows = forced["tokens"], forced["rows"]
    start = start_positions(model, toks)
    S = sorted({r["strength"] for r in rows})
    n = sum(r["strength"] == S[0] for r in rows)
    rep = [sum(any(r["top1"][-2:]) for r in rows if r["strength"] == s) for s in S]
    beg = [sum(any(r["top1"][i] for i in start) for r in rows if r["strength"] == s) for s in S]
    ax.plot(S, rep, "o-", color=MAROON, label="at the report")
    ax.plot(S, beg, "o-", color=SLATE, label="at the start of the reply")
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xticks(S, [f"{s:g}" for s in S], fontsize=8)
    ax.set_ylim(-1, n + 1)
    ax.set_xlabel("Injection strength (× mean residual norm)")
    ax.set_ylabel(f"Concepts where the word is the\ntop prediction (of {n})")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig3_introspect.png", dpi=200)
print("saved", OUT / "fig3_introspect.png")
