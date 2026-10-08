"""Directed modulation with a named word (post/DM_LINDSEY.md).

dm/dm_lindsey_cos.png   Lindsey (2025)'s own measure on his task, in the style of his Fig. (and of our
                        earlier post's Fig. 1): the cosine of the residual with the word's concept
                        vector, averaged over the reply's tokens, at each layer, under "Think about
                        {word}" and "Don't think about {word}" (mean over the 50 words, ±1 SEM over
                        words), against the same cosine with 100 unrelated words' vectors (gray band,
                        95% CI over those words) and the prompt with no word (dashed).  Centered
                        (the residual minus its mean over the no-word replies).  One panel per model
                        and frame.
dm/dm_lindsey_lens.png  The J-lens on the same trials: the share in which a form of the word reaches
                        the lens top 5 at any token of the reply over the paper's band (its Fig. 46
                        score; GPT-2: layers 7-9), on every token (solid) and without the reply's last
                        token (hatched), where a small model that has copied the sentence is about to add
                        the word.
dm/dm_word.png          The workspace paper's own protocol with a named word in place of a category,
                        next to the category results: the paper's score (rank 1), mean over each
                        condition's phrasings, on every token of the reply (top row) and without its last
                        token (bottom row), where a small model recalls the word that followed the
                        sentence in the prompt.  dm/dm_word_top5.png: the same with the top 5.

Run from the repo root:  python post/figures/dm_lindsey_fig.py
"""
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dm_data as D  # noqa: E402
import dm_lindsey as M  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "dm"
MAROON, SLATE, GRAY, INK, MUTED = "#800000", "#33658a", "#8a8580", "#2b2b2b", "#6b6b6b"
BG = "#fbfbf9"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#c8c4bc",
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
PANELS = [("gpt2", "human"), ("gemma-270m", "human"), ("gemma-270m-it", "human"), ("gemma-1b", "human"),
          ("gemma-1b-it", "human"), (None, None), ("gemma-270m", "chat"), ("gemma-270m-it", "chat"),
          ("gemma-1b", "chat"), ("gemma-1b-it", "chat")]
pct = plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%")


def load(m, f):
    try:
        return M.Lindsey(m, f)
    except FileNotFoundError:
        return None


def cos_figure():
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.2), sharex=True)
    for ax, (m, f) in zip(axes.flat, PANELS):
        r = load(m, f) if m else None
        if r is None:
            ax.axis("off")
            continue
        c = r.cos(centered=True)
        n = c.shape[1]
        x = np.arange(n) / (n - 1)
        base = r.z["base_c"][r.conds.index("think")]                    # [layer, control word]
        lo = base.mean(1) - 1.96 * base.std(1) / np.sqrt(base.shape[1])
        hi = base.mean(1) + 1.96 * base.std(1) / np.sqrt(base.shape[1])
        ax.fill_between(x, lo, hi, color=GRAY, alpha=0.3, lw=0, label="100 unrelated words (95% CI)")
        ax.plot(x, c[r.cond == "none"].mean(0), "--", color=GRAY, lw=1.2, label="no word in the prompt")
        for cond, color, label in (("think", MAROON, '"Think about {word}"'), ("dont_think", SLATE, '"Don\'t think about {word}"')):
            per_word = np.array([c[(r.cond == cond) & (r.word == w)].mean(0) for w in range(len(r.words))])
            mean, err = per_word.mean(0), per_word.std(0) / np.sqrt(len(per_word))
            ax.fill_between(x, mean - err, mean + err, color=color, alpha=0.25, lw=0)
            ax.plot(x, mean, color=color, lw=1.8, marker="o", ms=2.5, label=label)
        ax.axhline(0, color="#c8c4bc", lw=0.6)
        ax.set_title(f"{D.MODEL_LABEL[m]}, {M.FRAME_LABEL[f]}", fontsize=9)
        ax.set_xticks([0, 0.5, 1], ["first", "", "last"])
    for ax in axes[1]:
        ax.set_xlabel("layer (relative depth)")
    axes[0, 0].set_ylabel("cosine with the word's\nconcept vector (centered)")
    axes[1, 1].set_ylabel("cosine with the word's\nconcept vector (centered)")
    handles, labels = next(a for a in axes.flat if a.has_data()).get_legend_handles_labels()
    axes[1, 0].legend(handles, labels, frameon=False, fontsize=8, loc="center")
    fig.tight_layout()
    fig.savefig(OUT / "dm_lindsey_cos.png", dpi=200)
    print("saved", OUT / "dm_lindsey_cos.png")


def lens_figure():
    conds = [("none", "no\nword", GRAY), ("think", "think", MAROON), ("dont_think", "don't\nthink", SLATE),
             ("rewarded", "re-\nwarded", MAROON), ("punished", "pun-\nished", SLATE)]
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.0), sharey=True)
    for ax, (m, f) in zip(axes.flat, PANELS):
        r = load(m, f) if m else None
        if r is None:
            ax.axis("off")
            continue
        b, h = r.best("ours"), r.best("ours", skip_last=True)
        for i, (cond, label, color) in enumerate(conds):
            mask = r.mask(cond)
            ax.bar(i, (b[mask] <= 5).mean(), width=0.7, color=color, alpha=0.35, lw=0)
            ax.bar(i, (h[mask] <= 5).mean(), width=0.7, color="none", edgecolor=color, hatch="////", lw=0)
            ax.text(i, (b[mask] <= 5).mean() + 0.01, f"{100 * (b[mask] <= 5).mean():.0f}%", ha="center", fontsize=7, color=MUTED)
        ax.set_xticks(range(len(conds)), [c[1] for c in conds], fontsize=7.5)
        ax.yaxis.set_major_formatter(pct)
        ax.set_title(f"{D.MODEL_LABEL[m]}, {M.FRAME_LABEL[f]}" + (", layers 7-9" if m == "gpt2" else ""), fontsize=9)
    axes[0, 0].set_ylabel("trials with the word in\nthe J-lens top 5")
    axes[1, 1].set_ylabel("trials with the word in\nthe J-lens top 5")
    from matplotlib.patches import Patch
    axes[1, 0].legend(handles=[Patch(color=GRAY, alpha=0.35, label="at any token of the reply"),
                               Patch(facecolor="none", edgecolor=GRAY, hatch="////", label="without the reply's last token")],
                      frameon=False, fontsize=8, loc="center")
    fig.tight_layout()
    fig.savefig(OUT / "dm_lindsey_lens.png", dpi=200)
    print("saved", OUT / "dm_lindsey_lens.png")


def word_figure(k=1):
    conds = ["baseline"] + D.CONDS
    panels = [(m, f) for m, f in [("gpt2", "human"), ("gemma-270m", "human"), ("gemma-270m-it", "paper"), ("gemma-1b", "human"),
                                  ("gemma-1b-it", "paper")]]
    fig, axes = plt.subplots(2, 5, figsize=(13, 6.0), sharey=True)
    for ax, (m, f), kw in [(axes[i, j], panels[j], dict(skip_last=bool(i))) for i in range(2) for j in range(5)]:
        for s, (fam, frame, color, label) in enumerate((("topic", f, SLATE, "a category (track its members)"),
                                                       ("word", f + "_word", MAROON, "a named word (track the word)"))):
            if not (D.RESULTS / f"control/{m}/modulation_grid_{frame}.json").exists():
                continue
            rates = D.Grid(m, frame).rates(fam, "ours", k=k, **kw)
            for x, cond in enumerate(conds):
                mean, dots = rates[cond]
                xc = x + (s - 0.5) * 0.38
                ax.bar(xc, mean, width=0.36, color=color, alpha=0.3, lw=0, label=label if x == 0 else None)
                ys = sorted(dots.values())
                ax.plot(xc + np.linspace(-0.12, 0.12, len(ys)), ys, "o", ms=2, color=color)
        ax.set_xticks(range(len(conds)), ["none", "think\nabout", "mention", "don't\nthink", "ignore"], fontsize=7.5)
        ax.yaxis.set_major_formatter(pct)
        ax.set_title(f"{D.MODEL_LABEL[m]}, {M.FRAME_LABEL[f]}" + (", layers 7-9" if m == "gpt2" else "")
                     + ("\nwithout the reply's last token" if kw["skip_last"] else "\nevery token of the reply"), fontsize=8.5)
    what = "on top of the J-lens" if k == 1 else f"in the J-lens top {k}"
    for ax in axes[:, 0]:
        ax.set_ylabel(f"trials with the target\n{what}")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=SLATE, alpha=0.3, label="a category (its members tracked), as in DM_GEMMA"),
                        Patch(color=MAROON, alpha=0.3, label="a named word (the word tracked)")],
               loc="upper center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    name = "dm_word.png" if k == 1 else f"dm_word_top{k}.png"
    fig.savefig(OUT / name, dpi=200)
    print("saved", OUT / name)


if __name__ == "__main__":
    cos_figure()
    lens_figure()
    word_figure(1)
    word_figure(5)
