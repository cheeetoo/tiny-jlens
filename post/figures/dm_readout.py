"""Directed modulation, drawn like the paper's Fig. 9: what the J-lens shows while the model copies
a sentence under an instruction to think about something else.

One figure per example of the paper's Fig. 9 (citrus fruits; evaluating 3^2 - 2).  One panel per
model: Claude Sonnet 4.5 (the data behind the paper's figure, which has six of its lens layers),
Qwen3.5-0.8B, and GPT-2 small, all on the paper's prompt.  Columns are the tokens of the copied
sentence, rows are layers (the output at the top), and each cell shows the lens's top token there.
A cell is dark when a tracked word is the top token (the paper's "hit") and light when one is in
the top 5.  The gray bar marks the band the hit rate is read over.

Reads ref/paper-data/modulation-readout.json and
results/control/{gpt2,qwen}/modulation_readout_{human,paper}.json
(`python -m jl.control --model gpt2 modulation_readout`, and the same with `--model qwen`).
Run from the repo root:  python post/figures/dm_readout.py
"""
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.colors import to_rgb

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parent / "dm"
OUT.mkdir(exist_ok=True)
MAROON, SLATE, ORANGE, TEAL, GOLD, GRAY = "#800000", "#33658a", "#d45d00", "#3a7d6a", "#c99000", "#8a8580"
BG, CREAM = "#fbfbf9", "#f2e2b3"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.prop_cycle": cycler(color=[MAROON, SLATE, ORANGE, TEAL, GOLD, GRAY]),
})
DM = json.load(open(ROOT / "ref/jacobian-lens/data/experiments/directed-modulation.json"))
TRACKED = {"topic": set(DM["topic_categories"][0]["members"]), "math": {"7", "seven"}}
CLAUDE_BAND = (38, 92)          # percent of depth, the paper's workspace band


def shown(tok):
    """A token as cell text: spaces marked, scripts the default font lacks replaced."""
    t = tok.replace("⍽", " ").replace("↑", "").replace("\n", "\\n")
    if any(ord(c) > 0x2E7F or 0x1200 <= ord(c) <= 0x139F for c in t) or "\\x" in t:
        return "(other script)"
    t = t.strip() or repr(tok)[1:-1]
    return t if len(t) <= 11 else t[:10] + "…"


def claude_panel(i):
    """rows (top = output) of (label, in band, [(top token, best tracked rank or None)])."""
    p = json.load(open(ROOT / "ref/paper-data/modulation-readout.json"))["panels"][i]
    fam = ("topic", "math")[i]
    start = max(j for j, t in enumerate(p["tokens"]) if t == "Assistant") + 2
    cols = list(range(start, len(p["tokens"])))
    rows = []
    for li, layer in enumerate(x["layer"] for x in p["positions"][0]):
        pct = round(100 * layer / 24)
        cells = []
        for c in cols:
            topk = [shown(t["t"]).lower() for t in p["positions"][c][li]["topk"]]
            rank = next((r + 1 for r, t in enumerate(topk) if t in TRACKED[fam]), None)
            cells.append((shown(p["positions"][c][li]["top1"]), rank))
        rows.append(("output" if pct == 100 else f"L{pct}", CLAUDE_BAND[0] <= pct <= CLAUDE_BAND[1], cells))
    return [shown(p["tokens"][c]) for c in cols], rows


def our_panel(model, frame, i, first_layer, band):
    p = json.load(open(ROOT / f"results/control/{model}/modulation_readout_{frame}.json"))["panels"][i]
    cols = p["copy_positions"]
    rows = []
    for layer in reversed(p["layers"]):
        L = layer["layer"]
        if L != "output" and L < first_layer:
            continue
        cells = [(shown(layer["topk"][c][0][0]), layer["tracked_rank"][c]) for c in cols]
        rows.append(("output" if L == "output" else f"L{L}", L != "output" and band[0] <= L <= band[1], cells))
    return [shown(p["tokens"][c]) for c in cols], rows


def draw(ax, cols, rows, title):
    fill = np.ones((len(rows), len(cols), 3)) * np.array(to_rgb(BG))
    for r, (_, _, cells) in enumerate(rows):
        for c, (tok, rank) in enumerate(cells):
            if rank == 1:
                fill[r, c] = to_rgb(MAROON)
            elif rank is not None and rank <= 5:
                fill[r, c] = to_rgb(CREAM)
            ax.text(c, r, tok, ha="center", va="center", fontsize=6.5,
                    color="white" if rank == 1 else "black")
    ax.imshow(fill, aspect="auto")
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_xticks(range(len(cols)), cols, fontsize=8)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7.5)
    ax.set_xticks(np.arange(len(cols) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(rows) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=GRAY, lw=0.3, alpha=0.5)
    ax.tick_params(which="both", length=0)
    band = [r for r, row in enumerate(rows) if row[1]]
    if band:
        x = len(cols) - 0.5 + 0.012 * len(cols)
        ax.plot([x, x], [min(band) - 0.45, max(band) + 0.45], color=GRAY, lw=4,
                clip_on=False, solid_capstyle="butt")
    ax.set_title(title, fontsize=10, loc="left", pad=22)


for i, (name, instr) in enumerate([("topic", "Concentrate on citrus fruits while you write the sentence."),
                                   ("math", "Try to focus on evaluating 3^2 - 2 while you write the sentence.")]):
    panels = [(*claude_panel(i), "(a) Claude Sonnet 4.5 (the paper's data; six of its lens layers, in percent of depth)")]
    for label, model, frame, first, band in [("(b) Qwen3.5-0.8B (layers 8 to 22 of 0 to 23)", "qwen", "paper", 8, (15, 22)),
                                             ("(c) GPT-2 small (layers 0 to 10 of 0 to 11)", "gpt2", "human", 0, (7, 9))]:
        if (ROOT / f"results/control/{model}/modulation_readout_{frame}.json").exists():
            panels.append((*our_panel(model, frame, i, first, band), label))
    heights = [len(rows) + 2.5 for _, rows, _ in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(10.5, 0.27 * sum(heights) + 0.6),
                             gridspec_kw={"height_ratios": heights})
    for ax, (cols, rows, title) in zip(np.atleast_1d(axes), panels):
        draw(ax, cols, rows, title)
    fig.suptitle(f'Write "The old painting hung crookedly on the wall." {instr} Don\'t write anything else.',
                 fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0.01, 0, 1, 0.985))
    fig.savefig(OUT / f"dm_readout_{name}.png", dpi=200)
    print("saved", OUT / f"dm_readout_{name}.png")
