"""Directed modulation in Gemma 3, base against instruction-tuned, from the grid runs.

Each model is run on the paper's prompt in two frames: as a chat in Gemma's turn format (`paper`)
and as plain text (`human`).  Each model has its own Neuronpedia lens.  So for each frame, the two
models see the same tokens, and the difference between them is post-training (and its lens).

Run from the repo root:  python post/figures/dm_gemma.py [270m|1b]
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dm_data as D  # noqa: E402
from dm_tables import ALL, cond_mask, mean_over_phrasings, paired, pct  # noqa: E402

FRAMES = {"human": "plain text", "paper": "chat"}


def runs(size, tag=""):
    """[(label, Grid)] for the base and the instruction-tuned model, in each frame.  `tag` picks a
    partial run (`_math`: the math family at 1b, run on its own)."""
    out = []
    for frame in FRAMES:
        for m in (f"gemma-{size}", f"gemma-{size}-it"):
            if (D.RESULTS / f"control/{m}/modulation_grid_{frame}{tag}.json").exists():
                out.append((f"{D.MODEL_LABEL[m]}, {FRAMES[frame]}", D.Grid(m, frame + tag)))
    return out


def controls(size):
    """post/DM.md section 3: each model in its main frame, read as the paper reads it, with the other
    model's lens (LENS_FROM), and centered (LENS_CENTER=1), for whichever of these runs exist."""
    print("\n### Controls (categories, our band): the other model's lens, and the centered readout\n")
    print("| model, prompt | readout | " + " | ".join(D.COND_LABEL[c] for c in ALL) + " | think about vs. mention (pairs) "
          "| think about, where the model isn't about to say it | median best rank, think about |\n|---|---|"
          + "---|" * (len(ALL) + 3))
    for m, other in ((f"gemma-{size}", f"gemma-{size}-it"), (f"gemma-{size}-it", f"gemma-{size}")):
        frame = D.MAIN_FRAME[m]
        for label, model, fr in (("its own lens", m, frame), (f"{D.MODEL_LABEL[other]}'s lens", f"{m}_lens-{other}", frame),
                                 ("centered", m, f"{frame}_centered")):
            if not (D.RESULTS / f"control/{model}/modulation_grid_{fr}.json").exists():
                continue
            g = D.Grid(model, fr)
            b, h = g.best("ours"), g.best("ours", held=10)
            print(f"| {D.MODEL_LABEL[m]}, {FRAMES[frame]} | {label} | " + " | ".join(rate_row(g, "topic", b))
                  + f" | {paired(g, 'topic', 'ours')[1]} | {pct(mean_over_phrasings(g, 'topic', 'focus', h <= 1))} | "
                  f"{np.median(b[cond_mask(g, 'topic', 'focus')]):.0f} |")


def rate_row(g, fam, best, k=1):
    return [pct(mean_over_phrasings(g, fam, c, best <= k)) for c in ALL]


def coincident(g, fam, best, k=1):
    """[trial] True for trials whose (target, sentence) pair already scores with no instruction:
    the sentence itself brings up a member ("The sun is shining..." and weather), whatever the
    instruction says."""
    pairs = {(t["x"], t["carrier"]) for t, b in zip(g.trials, best) if t["family"] == fam and t["cond"] == "baseline" and b <= k}
    return np.array([(t["x"], t["carrier"]) in pairs for t in g.trials])


def rate_row_excl(g, fam, best, k=1):
    """rate_row over the pairs that don't score with no instruction."""
    keep, hit = ~coincident(g, fam, best, k), best <= k
    out = []
    for c in ALL:
        m = cond_mask(g, fam, c) & keep
        out.append(pct(float(np.mean([hit[m & (g.phrasing == p)].mean() for p in dict.fromkeys(g.phrasing[m])]))))
    return out


def main(size="270m"):
    R = runs(size)
    head = "| model, prompt | " + " | ".join(D.COND_LABEL[c] for c in ALL) + " |\n|---|" + "---|" * len(ALL)

    R_main = R
    for fam in D.FAMILIES:
        R = R_main
        if not any((g.family == fam).any() for _, g in R):      # the 1b math problems were run on their own
            R = runs(size, "_math")
            if not any((g.family == fam).any() for _, g in R):
                continue
        for k, what in ((1, "hit rate (a tracked token at lens rank 1)"), (5, "a tracked token in the lens top 5"),
                        (25, "a tracked token in the lens top 25")):
            print(f"\n### {D.FAMILY_LABEL[fam]}: {what}, our band\n\n{head}")
            for label, g in R:
                print(f"| {label} | " + " | ".join(rate_row(g, fam, g.best("ours"), k)) + " |")
        print(f"\n### {D.FAMILY_LABEL[fam]}: hit rate without the (target, sentence) pairs that hit with no "
              f"instruction, our band\n\n" + head.replace("| model, prompt |", "| model, prompt | pairs left out |")
              .replace("|---|", "|---|---|", 1))
        for label, g in R:
            best = g.best("ours")
            pairs = {(t["x"], t["carrier"]) for t, c in zip(g.trials, coincident(g, fam, best)) if c}
            print(f"| {label} | {len(pairs)}: " + "; ".join(f"{x} + sentence {c + 1}" for x, c in sorted(pairs))
                  + " | " + " | ".join(rate_row_excl(g, fam, best)) + " |")
        print(f"\n### {D.FAMILY_LABEL[fam]}: median best rank, our band\n\n{head}")
        for label, g in R:
            b = g.best("ours")
            print(f"| {label} | " + " | ".join(f"{np.median(b[cond_mask(g, fam, c)]):.0f}" for c in ALL) + " |")
        print(f"\n### {D.FAMILY_LABEL[fam]}: paired comparisons, our band\n")
        print("Share of (target, sentence) pairs in which the first condition ranks the target higher, on the "
              "median best rank over each condition's phrasings.\n")
        print("| model, prompt | mention vs. none | think about vs. mention | ignore vs. mention | "
              "don't think vs. mention | don't think vs. think about |\n|---|---|---|---|---|---|")
        for label, g in R:
            print(f"| {label} | " + " | ".join(paired(g, fam, "ours")) + " |")

    R = R_main
    controls(size)

    print("\n### Categories: hit rate by which layers are read\n")
    print("| model, prompt | layers | " + " | ".join(D.COND_LABEL[c] for c in ALL) + " |\n|---|---|" + "---|" * len(ALL))
    for label, g in R:
        for name in ("ours", "paper", "all"):
            if name in D.BANDS[g.model]:
                lo, hi = D.BANDS[g.model][name]
                print(f"| {label} | {lo} to {hi} ({name}) | " + " | ".join(rate_row(g, "topic", g.best(name))) + " |")
        print(f"| {label} | the model's output | " + " | ".join(rate_row(g, "topic", g.best("output"))) + " |")

    print("\n### Categories: hit rate at each layer alone, think about / mention / no instruction\n")
    for label, g in R:
        cells = []
        for L in range(len(g.layers) - 1):
            b = g.best((L, L))
            cells.append(f"{L}: " + "/".join(pct(mean_over_phrasings(g, "topic", c, b <= 1))
                                             for c in ("focus", "mention", "baseline")))
        print(f"- {label}: " + ", ".join(cells))

    print("\n### Categories: hit rate by which positions count, our band\n")
    print("| model, prompt | positions | " + " | ".join(D.COND_LABEL[c] for c in ALL) + " |\n|---|---|" + "---|" * len(ALL))
    for label, g in R:
        for name, kw in (("every token of the sentence", {}), ("without its first token", dict(skip_first=True)),
                         ("only where no tracked token is in the model's own top 10", dict(held=10))):
            print(f"| {label} | {name} | " + " | ".join(rate_row(g, "topic", g.best("ours", **kw))) + " |")

    print("\n### Categories: a tracked token in the lens top 25 by phrasing, our band\n")
    print("| phrasing | condition | " + " | ".join(label for label, _ in R) + " |\n|---|---|" + "---|" * len(R))
    rates = [g.rates("topic", "ours", k=25) for _, g in R]
    for c in ALL:
        for p in rates[0][c][1]:
            print(f"| `{p}` | {D.COND_LABEL[c]} | " + " | ".join(pct(r[c][1][p], 0) for r in rates) + " |")

    print("\n### Does the model copy the sentence when the reply isn't forced? (categories, first 5 sentences)\n")
    print("| model, prompt | condition | first token its top prediction | the rest | whole sentence | "
          "its own greedy reply is the sentence | begins with it |\n|---|---|---|---|---|---|---|")
    for m in (f"gemma-{size}", f"gemma-{size}-it"):
        f = D.RESULTS / f"control/{m}/modulation_copy.json"
        if not f.exists():
            continue
        d = json.load(open(f))
        for frame in FRAMES:
            for c in ALL:
                s = [r for r in d["rows"] if r["frame"] == frame and r["cond"] == c]
                rep = [r for r in d["replies"] if r["frame"] == frame and r["cond"] == c]
                if not s:
                    continue
                print(f"| {D.MODEL_LABEL[m]}, {FRAMES[frame]} | {D.COND_LABEL[c]} | {pct(np.mean([r['first'] for r in s]), 0)} | "
                      f"{pct(sum(r['rest'] for r in s) / sum(r['n_rest'] for r in s), 0)} | "
                      f"{pct(np.mean([r['first'] and r['rest'] == r['n_rest'] for r in s]), 0)} | "
                      f"{sum(r['exact'] for r in rep)}/{len(rep)} | {sum(r['starts'] for r in rep)}/{len(rep)} |")

    print("\n### Can the model do the sums?\n")
    for m in (f"gemma-{size}", f"gemma-{size}-it"):
        f = D.RESULTS / f"control/{m}/arithmetic.json"
        if f.exists():
            rows = json.load(open(f))["rows"]
            print(f"- {D.MODEL_LABEL[m]}, plain text: " + ", ".join(
                f"{fr} {sum(r['rank'] == 1 for r in rows if r['frame'] == fr)}/24"
                for fr in dict.fromkeys(r["frame"] for r in rows)))
        f = D.RESULTS / f"control/{m}/dm_clauses.json"
        if f.exists():
            math = json.load(open(f))["math"]
            print(f"- {D.MODEL_LABEL[m]}, chat (`What is {{expr}}? Answer with just the number.`): "
                  f"{sum(r['rank'] == 1 for r in math)}/24")


if __name__ == "__main__":
    main(*sys.argv[1:])
