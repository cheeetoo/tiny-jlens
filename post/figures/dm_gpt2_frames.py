"""Figure 7 of the post (Appendix C.2): directed modulation in GPT-2 small, drawn like the paper's
Fig. 65, one panel per prompt frame.

Each dot is one instruction phrasing, each bar the mean over a group's phrasings, and the dashed
line the rate with no instruction.  The score is the paper's: a member of the category on top of
the J-lens at any of layers 7 to 9 and any token of the copied sentence (on a zoomed axis).
Categories only (the math problems were run only on the paper's prompt, and give no hits).  22
categories, 20 sentences, every phrasing.

The frames (post Appendix B.2), with the share of the sentence's tokens (after the first) that
GPT-2 predicts itself when the reply isn't forced, from `modulation_copy`:
  human       the paper's prompt as plain text:
              \\n\\nHuman: Write "{sentence}" {instruction} Don't write anything else.\\n\\nAssistant: {sentence}
  copy        Write "{sentence}" {instruction} "{sentence}
  transcript  User: Write the following sentence: "{sentence}" {instruction}\\nAssistant: {sentence}
  exercise    Exercise 4. {instruction} Copy the sentence below.\\n{sentence}\\nAnswer: {sentence}
  teacher     The teacher wrote a sentence on the board for the class to copy. She said, "{instruction}"
              \\nSentence: {sentence}\\nThe student wrote: {sentence}
  narrative   Her teacher gave her a sentence to copy. She said, "{instruction}" She wrote it out: {sentence}
              (the sentence isn't shown before it is written, so this isn't a copying task)

Run from the repo root:  python post/figures/dm_gpt2_frames.py
"""
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dm_data as D  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "dm"
OUT.mkdir(exist_ok=True)
MAROON, SLATE, ORANGE, TEAL, GOLD, GRAY = "#800000", "#33658a", "#d45d00", "#3a7d6a", "#c99000", "#8a8580"
BG = "#fbfbf9"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.prop_cycle": cycler(color=[MAROON, SLATE, ORANGE, TEAL, GOLD, GRAY]),
})
COND_COLOR = {"focus": MAROON, "mention": ORANGE, "negated-think": GRAY, "dismissal": SLATE}
FRAMES = [("human", "The paper's prompt"), ("copy", "Sentence and copy"), ("transcript", "Chat transcript"),
          ("exercise", "Exercise"), ("teacher", "Teacher"), ("narrative", "Story")]
copy_rows = json.load(open(D.RESULTS / "control/gpt2/modulation_copy.json"))["rows"]


def copies(frame):
    """The share of the sentence's tokens after the first that GPT-2 predicts itself."""
    s = [r for r in copy_rows if r["frame"] == frame]
    return sum(r["rest"] for r in s) / sum(r["n_rest"] for r in s)


fig, axes = plt.subplots(3, 2, figsize=(6.4, 7.4), sharex=True, sharey=True)
for ax, (frame, label) in zip(axes.flat, FRAMES):
    r = D.Grid("gpt2", frame).rates("topic")
    for x, cond in enumerate(D.CONDS):
        mean, dots = r[cond]
        ax.bar(x, mean, width=0.7, color=COND_COLOR[cond], alpha=0.3)
        ys = sorted(dots.values())
        ax.plot(x + np.linspace(-0.22, 0.22, len(ys)), ys, "o", ms=3, color=COND_COLOR[cond])
    ax.axhline(r["baseline"][0], color="black", lw=1, ls="--")
    ax.set_xticks(range(len(D.CONDS)), [D.COND_LABEL[c].replace(" about", "\nabout") for c in D.CONDS], fontsize=7.5)
    ax.set_xlim(-0.6, len(D.CONDS) - 0.4)
    ax.set_ylim(0, 0.07)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.set_title(f"{label}\n(GPT-2 predicts {100 * copies(frame):.0f}% of the sentence itself)", fontsize=9)
fig.supylabel("Trials with a category member on top of the J-lens", fontsize=10)
fig.tight_layout()
fig.savefig(OUT.parent / "fig7_modulation_frames.png", dpi=200)
print("saved", OUT.parent / "fig7_modulation_frames.png")
