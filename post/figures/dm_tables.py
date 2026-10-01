"""The directed-modulation numbers in post/DM_SETUP.md, as Markdown tables, from the grid runs.

Run from the repo root:  python post/figures/dm_tables.py
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dm_data as D  # noqa: E402

sys.path.insert(0, str(D.ROOT))
from jl.stats import sign_test  # noqa: E402

ALL = ["baseline"] + D.CONDS
GPT2_FRAMES = ["human", "copy", "transcript", "exercise", "teacher", "narrative"]


def pct(x, digits=1):
    return "–" if x != x else f"{100 * x:.{digits}f}%"


def cond_mask(g, fam, cond):
    return (g.family == fam) & (g.cond == cond)


def mean_over_phrasings(g, fam, cond, values):
    m = cond_mask(g, fam, cond)
    return float(np.mean([values[m & (g.phrasing == p)].mean() for p in dict.fromkeys(g.phrasing[m])]))


def condition_table(g, fam, band="ours"):
    best = g.best(band)
    print(f"| condition | hit (top 1) | top 5 | top 25 | median best rank | trials |\n|---|---|---|---|---|---|")
    for c in ALL:
        m = cond_mask(g, fam, c)
        print(f"| {D.COND_LABEL[c]} | " + " | ".join(pct(mean_over_phrasings(g, fam, c, best <= k)) for k in (1, 5, 25))
              + f" | {np.median(best[m]):.0f} | {m.sum():,} |")


def paired(g, fam, band="ours"):
    """Per (target, sentence): the median best rank over a condition's phrasings; the share of pairs
    in which the first condition ranks the target higher (sign test)."""
    best = g.best(band)
    med = {}
    for t, b in zip(g.trials, best):
        if t["family"] == fam:
            med.setdefault((t["x"], t["carrier"], t["cond"]), []).append(b)
    med = {k: float(np.median(v)) for k, v in med.items()}
    keys = sorted({(x, c) for x, c, _ in med})
    out = []
    for a, b in (("mention", "baseline"), ("focus", "mention"), ("dismissal", "mention"),
                 ("negated-think", "mention"), ("negated-think", "focus")):
        frac, _, p = sign_test([(med[(x, c, a)], med[(x, c, b)]) for x, c in keys])
        out.append(f"{frac:.0%} (p = {p:.0e})")
    return out


def main():
    grids = {m: D.Grid(m) for m in ("gpt2", "qwen")}
    claude, claude_models = D.claude()

    for m, g in grids.items():
        for fam in D.FAMILIES:
            if not (g.family == fam).any():
                continue
            lo, hi = D.BANDS[m]["ours"]
            print(f"\n### {D.MODEL_LABEL[m]}, {D.FAMILY_LABEL[fam].lower()}, frame `{g.frame}`, layers {lo} to {hi}\n")
            condition_table(g, fam)

    print("\n### Hit rate (top 1) by which layers are read, categories\n")
    print("| model | layers | " + " | ".join(D.COND_LABEL[c] for c in ALL) + " |\n|---|---|" + "---|" * len(ALL))
    for m, g in grids.items():
        bands = [(f"{lo} to {hi} ({name})", (lo, hi)) for name, (lo, hi) in
                 (("our band", D.BANDS[m]["ours"]), ("the paper's band in depth", D.BANDS[m]["paper"]),
                  ("every lens layer", D.BANDS[m]["all"]))]
        for label, band in bands:
            best = g.best(band)
            print(f"| {D.MODEL_LABEL[m]} | {label} | "
                  + " | ".join(pct(mean_over_phrasings(g, "topic", c, best <= 1)) for c in ALL) + " |")

    print("\n### Hit rate (top 1) by which positions count, categories, our band\n")
    print("| model | positions | " + " | ".join(D.COND_LABEL[c] for c in ALL) + " |\n|---|---|" + "---|" * len(ALL))
    for m, g in grids.items():
        for label, kw in (("every token of the sentence", {}), ("without its first token", dict(skip_first=True)),
                          ("only where no tracked token is in the model's own top 10", dict(held=10)),
                          ("both", dict(skip_first=True, held=10))):
            best = g.best("ours", **kw)
            print(f"| {D.MODEL_LABEL[m]} | {label} | "
                  + " | ".join(pct(mean_over_phrasings(g, "topic", c, best <= 1)) for c in ALL) + " |")
    print("\nShare of hit cells (layer, position) by position in the sentence, focus trials, categories, our band:\n")
    for m, g in grids.items():
        lo, hi = D.BANDS[m]["ours"]
        r = g.ranks[cond_mask(g, "topic", "focus")][:, lo:hi + 1, :]
        cells = (r == 1).sum(axis=(0, 1))
        print(f"- {D.MODEL_LABEL[m]}: " + ", ".join(f"token {i + 1}: {c / max(cells.sum(), 1):.0%}" for i, c in enumerate(cells) if c)
              + f" ({cells.sum()} cells)")

    print("\n### Hit rate by phrasing (Claude: the paper's Fig. 65 data)\n")
    print("| condition | phrasing | " + " | ".join([D.MODEL_LABEL[m] for m in grids] + claude_models) + " |\n|---|---|"
          + "---|" * (len(grids) + len(claude_models)))
    for fam in D.FAMILIES:
        print(f"| **{D.FAMILY_LABEL[fam]}** | | " + " | " * (len(grids) + len(claude_models) - 1) + " |")
        rates = {m: g.rates(fam) for m, g in grids.items() if (g.family == fam).any()}
        for c in ALL:
            phrasings = list(claude[fam][0][c][1]) if c != "baseline" else [""]
            cells = [pct(rates[m][c][0]) if m in rates else "–" for m in grids] + [pct(r[c][0]) for r in claude[fam]]
            print(f"| {D.COND_LABEL[c]} | **mean** | " + " | ".join(f"**{x}**" for x in cells) + " |")
            for p in phrasings if c != "baseline" else []:
                cells = [pct(rates[m][c][1].get(p, float("nan"))) if m in rates else "–" for m in grids]
                cells += [pct(r[c][1].get(p, float("nan"))) for r in claude[fam]]
                print(f"| | `{p}` | " + " | ".join(cells) + " |")

    print("\n### GPT-2 by phrasing: a member in the lens top 25, categories, layers 7 to 9\n")
    g = grids["gpt2"]
    r = g.rates("topic", k=25)
    rows = sorted(((v, p, c) for c in D.CONDS for p, v in r[c][1].items()), reverse=True)
    print("| phrasing | condition | top 25 |\n|---|---|---|")
    for v, p, c in rows:
        print(f"| `{p}` | {D.COND_LABEL[c]} | {pct(v, 0)} |")
    print(f"| (no instruction) | | {pct(r['baseline'][0], 0)} |")

    print("\n### GPT-2 by prompt frame, categories, layers 7 to 9\n")
    print("| frame | " + " | ".join(f"{D.COND_LABEL[c]}: hit / top 25" for c in ALL) + " |\n|---|" + "---|" * len(ALL))
    rows = []
    for frame in GPT2_FRAMES:
        if not (D.RESULTS / f"control/gpt2/modulation_grid_{frame}.json").exists():
            continue
        g = D.Grid("gpt2", frame)
        best = g.best("ours")
        print(f"| `{frame}` | " + " | ".join(
            f"{pct(mean_over_phrasings(g, 'topic', c, best <= 1))} / {pct(mean_over_phrasings(g, 'topic', c, best <= 25), 0)}"
            for c in ALL) + " |")
        rows.append((frame, paired(g, "topic")))
    print("\n### Paired comparisons (share of category and sentence pairs in which the first ranks the category higher)\n")
    print("| model, frame | mention vs. none | think about vs. mention | ignore vs. mention | don't think vs. mention | don't think vs. think about |\n|---|---|---|---|---|---|")
    for frame, cells in rows:
        print(f"| GPT-2, `{frame}` | " + " | ".join(cells) + " |")
    print(f"| Qwen, `{grids['qwen'].frame}` | " + " | ".join(paired(grids["qwen"], "topic")) + " |")

    print("\n### Qwen by prompt, categories, layers 15 to 22 (hit rate)\n")
    print("| prompt | sentences | " + " | ".join(D.COND_LABEL[c] for c in ALL) + " |\n|---|---|" + "---|" * len(ALL))
    g = grids["qwen"]
    best = g.best("ours")
    print(f"| `paper`: the paper's | 20 | " + " | ".join(pct(mean_over_phrasings(g, "topic", c, best <= 1)) for c in ALL) + " |")
    first10 = np.array([t["carrier"] < 10 for t in g.trials])
    print("| `paper`, first 10 sentences | 10 | " + " | ".join(
        pct(float((best <= 1)[cond_mask(g, "topic", c) & first10].mean())) for c in ALL) + " |")
    for frame, label in (("after", "`after`: `Write the following sentence: \"…\" {instruction}`"),
                         ("before", "`before`: `{instruction} Write the following sentence: \"…\"`")):
        old = json.load(open(D.RESULTS / f"control/qwen/modulation_{frame}.json"))
        R = [r for r in old["rows"] if r["family"] == "topic"]
        print(f"| {label} | {len(old['carriers'])} | " + " | ".join(
            pct(np.mean([r["hit"] for r in R if r["cond"] == c])) for c in ALL) + " |")

    print("\n### Does the model copy the sentence when the reply isn't forced?\n")
    print("| model, frame | condition | first token right | the rest of the tokens right | whole sentence |\n|---|---|---|---|---|")
    for m in grids:
        rows = json.load(open(D.RESULTS / f"control/{m}/modulation_copy.json"))["rows"]
        for frame in dict.fromkeys(r["frame"] for r in rows):
            for c in ["baseline", "focus", "mention", "negated-think", "dismissal"]:
                s = [r for r in rows if r["frame"] == frame and r["cond"] == c]
                print(f"| {D.MODEL_LABEL[m]}, `{frame}` | {D.COND_LABEL[c]} | {pct(np.mean([r['first'] for r in s]), 0)} | "
                      f"{pct(sum(r['rest'] for r in s) / sum(r['n_rest'] for r in s), 0)} | "
                      f"{pct(np.mean([r['first'] and r['rest'] == r['n_rest'] for r in s]), 0)} |")

    g = grids["qwen"]
    if (g.family == "math").any():
        can = {r["expr"] for r in json.load(open(D.RESULTS / "control/qwen/dm_clauses.json"))["math"] if r["rank"] == 1}
        doable = np.array([t["x"] in can for t in g.trials])
        print(f"\n### Qwen, the {len(can)} math problems it answers when asked directly, layers 15 to 22\n")
        print("| condition | hits | trials | top 5 | top 25 | median best rank |\n|---|---|---|---|---|---|")
        best = g.best("ours")
        for c in ALL:
            m = cond_mask(g, "math", c) & doable
            print(f"| {D.COND_LABEL[c]} | {(best[m] <= 1).sum()} | {m.sum():,} | {pct((best[m] <= 5).mean())} | "
                  f"{pct((best[m] <= 25).mean())} | {np.median(best[m]):.0f} |")
        for label, band in (("the paper's band in depth (9 to 21)", "paper"), ("every lens layer", "all")):
            b = g.best(band)
            print(f"\nHits over {label}: " + ", ".join(
                f"{D.COND_LABEL[c]} {(b[cond_mask(g, 'math', c) & doable] <= 1).sum()}" for c in ALL) + ".")


if __name__ == "__main__":
    main()
