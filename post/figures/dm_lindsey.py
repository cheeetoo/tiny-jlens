"""Directed modulation with a named word: the numbers in post/DM_LINDSEY.md, as Markdown tables.

Two runs, on GPT-2 small and Gemma 3 (270m, 1b; base and instruction-tuned):

  lindsey  Lindsey (2025)'s intentional-control experiment as that paper runs it (jl/intentional.py):
           its prompts, words and sentences, read with its concept vectors and with the J-lens.
  word     the workspace paper's directed-modulation protocol (jl.control.modulation_grid) with the
           category replaced by one of Lindsey's words, named in the instruction and tracked itself.

Run from the repo root:  python post/figures/dm_lindsey.py
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dm_data as D  # noqa: E402
from dm_tables import ALL, mean_over_phrasings, paired, pct  # noqa: E402

sys.path.insert(0, str(D.ROOT))
from jl.stats import sign_test  # noqa: E402

MODELS = ["gpt2", "gemma-270m", "gemma-270m-it", "gemma-1b", "gemma-1b-it"]
FRAMES = {"gpt2": ["human"], **{m: ["chat", "human"] for m in MODELS[1:]}}
FRAME_LABEL = {"human": "plain text", "chat": "chat", "paper": "chat"}
GRID_FRAME = {"human": "human", "chat": "paper"}           # jl.control calls Gemma's chat frame `paper`
PAIRS = [("think", "dont_think"), ("rewarded", "punished"), ("happy", "sad"), ("charity", "terrorist"),
         ("such_thing_pos", "such_thing_neg"), ("often_think_pos", "often_think_neg"),
         ("fwiw_pos", "fwiw_neg"), ("if_i_pos", "if_i_neg")]
PAIR_LABEL = {"think": "think / don't think", "rewarded": "rewarded / punished", "happy": "happy / sad",
              "charity": "charity / terrorist orgs", "such_thing_pos": "[no] such thing as (control)",
              "often_think_pos": "I [don't] often think about (control)",
              "fwiw_pos": "for what it's worth, I [don't] often think (control)",
              "if_i_pos": "if I think about ..., rewarded / punished (control)"}
FRACS = np.linspace(0.0, 1.0, 19)                           # the old post's depth grid (gap_scaling.py)


class Lindsey:
    """One jl.intentional run: results/intentional/{model}/lindsey_{frame}.{json,npz}."""

    def __init__(self, model, frame):
        d = D.RESULTS / "intentional" / model
        self.model, self.frame = model, frame
        self.meta = json.load(open(d / f"lindsey_{frame}.json"))
        z = np.load(d / f"lindsey_{frame}.npz")
        self.z = z
        self.ranks = z["ranks"].astype(np.int64)
        self.ranks[self.ranks == 0] = D.NO_RANK
        self.cond = np.array([t["cond"] for t in self.meta["trials"]])
        self.word = np.array([t["word"] for t in self.meta["trials"]])
        self.sent = np.array([t["sentence"] for t in self.meta["trials"]])
        self.words = self.meta["words"]
        self.tracked = np.array([bool(self.meta["tracked"][w]) for w in self.words])
        self.conds = self.meta["conditions"]

    def best(self, band="paper", held=None, skip_last=False):
        """[trial] best lens rank of the word over the band's layers and every token of the reply
        (`held`, `skip_last`: as dm_data.Grid.best)."""
        lo, hi = D.BANDS[self.model][band] if isinstance(band, str) else band
        r = self.ranks[:, lo:hi + 1, :]
        if held:
            r = np.where((self.ranks[:, -1, :] <= held)[:, None, :], D.NO_RANK, r)
        if skip_last:
            n = self.z["n_pos"].astype(int)
            r = np.where((np.arange(r.shape[2]) == (n - 1)[:, None])[:, None, :], D.NO_RANK, r)
        return r.min(axis=(1, 2))

    def mask(self, cond, words=None):
        """Trials of a condition, over the tracked words (or the given word indices)."""
        w = np.flatnonzero(self.tracked) if words is None else words
        return (self.cond == cond) & np.isin(self.word, w)

    def cos(self, centered=False):
        return self.z["cos_c" if centered else "cos"]

    def grid(self):
        """The layers of the old post's 19-point depth grid, in this model's block outputs."""
        last = self.cos().shape[1] - 1
        return np.array(sorted({round(f * last) for f in FRACS}))

    def gap(self, pos, neg, centered=False):
        """[word, sentence, layer] paired gap pos - neg in the cosine with the word's concept vector."""
        c = self.cos(centered)
        W, S = len(self.words), len(self.meta["sentences"])
        def arr(cond):
            a = np.full((W, S, c.shape[1]), np.nan)
            m = self.cond == cond
            a[self.word[m], self.sent[m]] = c[m]
            return a
        return arr(pos) - arr(neg)

    def normalized_peak(self, pos, neg, centered=False):
        """The old post's normalized peak gap (scripts/gap_scaling.py): for each word, the mean paired gap
        over sentences on the depth grid, maxed over layers and divided by the SD of the paired gap
        across sentences at that layer; and its split-half version (layer picked on the even sentences,
        value read on the odd), which has no selection bias.  Each a [word] array."""
        g = self.gap(pos, neg, centered)[:, :, self.grid()]
        full, split = [], []
        for w in range(g.shape[0]):
            j = np.argmax(g[w].mean(0))
            full.append(g[w, :, j].mean() / g[w, :, j].std())
            j = np.argmax(g[w, 0::2].mean(0))
            split.append(g[w, 1::2, j].mean() / g[w, 1::2, j].std())
        return np.array(full), np.array(split)


    def directional(self, pos, neg, centered=False):
        """The gap pos - neg in the direction it is meant to go: at the one layer where the mean over
        words of the paired gap peaks on the even sentences, the mean over words of the gap on the odd
        sentences, and its t over words (mean / SEM).  Returns (layer, mean gap, t)."""
        g = self.gap(pos, neg, centered)                                   # [word, sentence, layer]
        L = int(np.argmax(np.nanmean(g[:, 0::2], axis=(0, 1))))
        w = np.nanmean(g[:, 1::2, L], axis=1)                               # [word]
        return L, w.mean(), w.mean() / sem(w)

    def curve(self, pos, neg, centered=False):
        """[layer] mean over words of the paired gap, and its SEM over words."""
        w = np.nanmean(self.gap(pos, neg, centered), axis=1)               # [word, layer]
        return w.mean(0), w.std(0) / np.sqrt(len(w))


def sem(x):
    return x.std() / np.sqrt(len(x))


def runs():
    for m in MODELS:
        for f in FRAMES[m]:
            if (D.RESULTS / f"intentional/{m}/lindsey_{f}.json").exists():
                yield f"{D.MODEL_LABEL[m]}, {FRAME_LABEL[f]}", Lindsey(m, f)


def cosine_tables(R):
    for centered in (False, True):
        what = "centered (the residual minus its mean over the no-word replies)" if centered else "raw, as the paper"
        print(f"\n### Concept vectors, {what}: think vs. don't think\n")
        print("Normalized peak gap, mean ± SEM over the 50 words (the old post's Fig. 3 measure); split-half: the "
              "layer picked on even sentences, the gap read on odd ones. The cosines are at the peak layer of the "
              "mean think - don't-think gap, averaged over all pairs.\n")
        print("| model, prompt | normalized peak gap | split-half | the same, reversed (don't think - think) | "
              "peak layer | think | don't think | control words (mean, 95% CI half-width) | no word | "
              "directional: layer, gap, t over words |\n|---|---|---|---|---|---|---|---|---|---|")
        for label, r in R:
            full, split = r.normalized_peak("think", "dont_think", centered)
            rev = r.normalized_peak("dont_think", "think", centered)[1]
            dl, dg, dt = r.directional("think", "dont_think", centered)
            c = r.cos(centered)
            mean_gap = np.nanmean(r.gap("think", "dont_think", centered), axis=(0, 1))
            L = int(np.argmax(mean_gap))
            base = r.z["base_c" if centered else "base"][r.conds.index("think")][L]     # [control word]
            print(f"| {label} | {full.mean():.2f} ± {sem(full):.2f} | {split.mean():.2f} ± {sem(split):.2f} | "
                  f"{rev.mean():.2f} ± {sem(rev):.2f} | {L} | "
                  f"{c[r.cond == 'think', L].mean():.4f} | {c[r.cond == 'dont_think', L].mean():.4f} | "
                  f"{base.mean():.4f} ± {1.96 * base.std() / np.sqrt(len(base)):.4f} | {c[r.cond == 'none', L].mean():.4f} | "
                  f"{dl}, {dg:+.5f}, {dt:.1f} |")
        print(f"\n### Concept vectors, {what}: every pair of prompts, directional (t over words of the split-half "
              f"peak gap; positive = the affirmative prompt higher) and the old post's split-half normalized peak gap, "
              f"forward / reversed\n")
        print("| model, prompt | " + " | ".join(PAIR_LABEL[p] for p, _ in PAIRS) + " |\n|---|" + "---|" * len(PAIRS))
        for label, r in R:
            cells = []
            for p, n in PAIRS:
                s, rv = r.normalized_peak(p, n, centered)[1], r.normalized_peak(n, p, centered)[1]
                cells.append(f"t = {r.directional(p, n, centered)[2]:.1f}; {s.mean():.1f} / {rv.mean():.1f}")
            print(f"| {label} | " + " | ".join(cells) + " |")
        print(f"\n### Concept vectors, {what}: think - don't think at each layer, mean over words (SEM)\n")
        for label, r in R:
            m, e = r.curve("think", "dont_think", centered)
            print(f"- {label}: " + ", ".join(f"{L}: {1e3 * a:+.2f} ({1e3 * b:.2f})" for L, (a, b) in enumerate(zip(m, e))) + "  (x 1e-3)")


def levels(R):
    """The cosine with the word's concept vector under every prompt, over the paper's band (layers 38% to
    92% of depth; GPT-2 5-10), centered, x 1e3: where "think" sits against prompts that only mention the
    word, not only against "don't think"."""
    order = ["none", "think", "dont_think", "rewarded", "punished", "happy", "sad", "charity", "terrorist",
             "such_thing_pos", "such_thing_neg", "often_think_pos", "often_think_neg", "fwiw_pos", "fwiw_neg",
             "if_i_pos", "if_i_neg"]
    print("\n### Concept vectors (centered), the level under every prompt: mean over the paper's band and the "
          "reply's tokens, x 1e3 (SEM over words)\n")
    print("| model, prompt | " + " | ".join(order) + " |\n|---|" + "---|" * len(order))
    for label, r in R:
        lo, hi = D.BANDS[r.model]["paper"]
        for key, what in (("cos_c", ""), ("cos_c_nl", ", without the last token")):
            if key not in r.z:
                continue
            c = r.z[key][:, lo:hi + 1].mean(1)
            cells = []
            for k in order:
                w = np.array([c[(r.cond == k) & (r.word == i)].mean() for i in range(len(r.words))])
                cells.append(f"{1e3 * w.mean():.1f} ({1e3 * sem(w):.1f})")
            print(f"| {label}{what} | " + " | ".join(cells) + " |")


def directive_vs_mention(R):
    """The workspace paper's question, asked inside Lindsey's design: does the directive raise the word
    more than an affirmative phrase that mentions it without asking for anything (his control prompts)?"""
    pos = ["such_thing_pos", "often_think_pos", "fwiw_pos", "if_i_pos"]
    print("\n### Think vs. a plain mention, inside Lindsey's design: \"Think about {word} while you write\" against "
          "each affirmative control prompt\n")
    print("Concept vectors (centered): t over words of the split-half peak gap (positive = think higher). J-lens: share "
          "of (word, sentence) pairs where think ranks the word higher, and the top-5 rates.\n")
    print("| model, prompt | " + " | ".join(f"think vs. {p}" for p in pos) + " |\n|---|" + "---|" * len(pos))
    for label, r in R:
        b = r.best("ours")
        cells = []
        for p in pos:
            t = r.directional("think", p, centered=True)[2]
            ma, mb = r.mask("think"), r.mask(p)
            kb = {(w, s_): x for w, s_, x in zip(r.word[mb], r.sent[mb], b[mb])}
            frac = sign_test([(x, kb[(w, s_)]) for w, s_, x in zip(r.word[ma], r.sent[ma], b[ma])])[0]
            cells.append(f"t = {t:.1f}; lens {frac:.0%}; top 5 {pct((b[ma] <= 5).mean(), 0)} vs. {pct((b[mb] <= 5).mean(), 0)}")
        print(f"| {label} | " + " | ".join(cells) + " |")


def bands(r):
    return [("ours", "")] if r.model != "gpt2" else [("ours", ", layers 7-9"), ("paper", ", layers 5-10")]


def lens_tables(R, common):
    for words, wlabel in ((None, "each model's tracked words"), (common, f"the {len(common)} words every model tracks")):
        for k in (1, 5, 25):
            print(f"\n### J-lens, a form of the word at rank {'1' if k == 1 else f'≤ {k}'}, {wlabel}\n")
            print("| model, prompt | words | " + " | ".join(r_ for r_ in ["no word", "think", "don't think",
                  "rewarded", "punished", "happy", "sad", "charity", "terrorist"]) + " |\n|---|---|" + "---|" * 9)
            for label, r in R:
                for band, bl in bands(r):
                    b = r.best(band)
                    ws = np.flatnonzero(r.tracked) if words is None else words
                    cells = [pct((b[r.mask(c, ws)] <= k).mean()) for c in
                             ["none", "think", "dont_think", "rewarded", "punished", "happy", "sad", "charity", "terrorist"]]
                    print(f"| {label}{bl} | {len(ws)} | " + " | ".join(cells) + " |")
    print("\n### J-lens, the control prompts, a form of the word in the top 5, the paper's band\n")
    ctrl = [c for p in PAIRS[4:] for c in p]
    print("| model, prompt | " + " | ".join(ctrl) + " |\n|---|" + "---|" * len(ctrl))
    for label, r in R:
        b = r.best("ours")
        print(f"| {label} | " + " | ".join(pct((b[r.mask(c)] <= 5).mean()) for c in ctrl) + " |")
    print("\n### J-lens, median best rank, the paper's band\n")
    cs = ["none", "think", "dont_think", "rewarded", "punished"]
    print("| model, prompt | " + " | ".join(cs) + " |\n|---|" + "---|" * len(cs))
    for label, r in R:
        b = r.best("ours")
        print(f"| {label} | " + " | ".join(f"{np.median(b[r.mask(c)]):.0f}" for c in cs) + " |")
    print("\n### J-lens, paired: share of (word, sentence) pairs where the first prompt ranks the word higher\n")
    print("| model, prompt | " + " | ".join(f"{a} vs. {b_}" for a, b_ in PAIRS) + " | think vs. no word |\n|---|"
          + "---|" * (len(PAIRS) + 1))
    for label, r in R:
        b = r.best("ours")
        cells = []
        for a, n in PAIRS + [("think", "none")]:
            ma, mb = r.mask(a), r.mask(n)
            ka = {(w, s): x for w, s, x in zip(r.word[ma], r.sent[ma], b[ma])}
            kb = {(w, s): x for w, s, x in zip(r.word[mb], r.sent[mb], b[mb])}
            frac, _, p = sign_test([(ka[k], kb[k]) for k in ka])
            cells.append(f"{frac:.0%} (p = {p:.0e})")
        print(f"| {label} | " + " | ".join(cells) + " |")
    print("\n### J-lens, which tokens of the reply count: a form of the word in the top 5 (rank 1 in brackets)\n")
    print("| model, prompt | tokens counted | no word | think | don't think | rewarded | punished |\n|---|---|---|---|---|---|---|")
    for label, r in R:
        for name, kw in (("every token", {}), ("all but the last", dict(skip_last=True)),
                         ("only where no form is in the model's own top 10", dict(held=10))):
            b = r.best("ours", **kw)
            print(f"| {label} | {name} | " + " | ".join(f"{pct((b[r.mask(c)] <= 5).mean())} ({pct((b[r.mask(c)] <= 1).mean())})"
                                                          for c in ("none", "think", "dont_think", "rewarded", "punished")) + " |")
    print("\n### J-lens, each layer alone: a form of the word in the top 5, think / don't think / no word\n")
    for label, r in R:
        n = r.ranks.shape[1] - 1
        cells = []
        for L in list(range(n)) + ["out"]:
            b = r.best((L, L)) if L != "out" else r.ranks[:, -1, :].min(axis=1)
            cells.append(f"{L}: " + "/".join(pct((b[r.mask(c)] <= 5).mean(), 0) for c in ("think", "dont_think", "none")))
        print(f"- {label}: " + ", ".join(cells))
    print("\n### Does the model copy the sentence? (each token of the reply its own top prediction)\n")
    print("| model, prompt | condition | first token | the rest | whole reply |\n|---|---|---|---|---|")
    for label, r in R:
        z = r.z
        for c in ("none", "think", "dont_think"):
            m = r.cond == c
            print(f"| {label} | {c} | {pct(z['first_ok'][m].mean(), 0)} | "
                  f"{pct(z['rest_ok'][m].sum() / (z['n_pos'][m] - 1).sum(), 0)} | "
                  f"{pct((z['first_ok'][m] & (z['rest_ok'][m] == z['n_pos'][m] - 1)).mean(), 0)} |")


def paired_kw(g, fam, band, **kw):
    """dm_tables.paired, with Grid.best's options."""
    best = g.best(band, **kw)
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


def word_grid_tables():
    """The workspace paper's protocol with a named word (family `word`), next to the categories."""
    G = []
    for m in MODELS:
        for f in ("human", "paper"):
            if (D.RESULTS / f"control/{m}/modulation_grid_{f}_word.json").exists():
                G.append((f"{D.MODEL_LABEL[m]}, {FRAME_LABEL[f]}", D.Grid(m, f"{f}_word"), D.Grid(m, f)))
    if not G:
        return
    head = "| model, prompt | target | " + " | ".join(D.COND_LABEL[c] for c in ALL) + " |\n|---|---|" + "---|" * len(ALL)
    for k in (1, 5, 25):
        print(f"\n### The workspace paper's protocol, {'hit rate (rank 1)' if k == 1 else f'top {k}'}, the paper's band (GPT-2: layers 7-9)\n")
        print(head)
        for label, gw, gc in G:
            for fam, g, name in (("word", gw, "a named word"), ("topic", gc, "a category (DM_GEMMA)")):
                b = g.best("ours")
                print(f"| {label} | {name} | " + " | ".join(pct(mean_over_phrasings(g, fam, c, b <= k)) for c in ALL) + " |")
    print("\n### The workspace paper's protocol with a named word: paired comparisons, the paper's band\n")
    print("| model, prompt | mention vs. none | think about vs. mention | ignore vs. mention | "
          "don't think vs. mention | don't think vs. think about |\n|---|---|---|---|---|---|")
    for label, gw, _ in G:
        print(f"| {label} | " + " | ".join(paired(gw, "word", "ours")) + " |")
    for k in (1, 5):
        print(f"\n### The same, {'rank 1' if k == 1 else 'top 5'}, without the reply's last token (where a small model "
              f"recalls the word that followed the sentence in the prompt), and only where no form is in the model's own top 10\n")
        print(head.replace("| target |", "| tokens counted |"))
        for label, gw, gc in G:
            for name, kw in (("all but the last", dict(skip_last=True)), ("not about to say it", dict(held=10))):
                b = gw.best("ours", **kw)
                print(f"| {label} | {name} | " + " | ".join(pct(mean_over_phrasings(gw, "word", c, b <= k)) for c in ALL) + " |")
    print("\n### Named word, paired comparisons without the reply's last token, the paper's band\n")
    print("| model, prompt | mention vs. none | think about vs. mention | ignore vs. mention | "
          "don't think vs. mention | don't think vs. think about |\n|---|---|---|---|---|---|")
    for label, gw, _ in G:
        print(f"| {label} | " + " | ".join(paired_kw(gw, "word", "ours", skip_last=True)) + " |")
    print("\n### A named word in the top 25, by phrasing, the paper's band\n")
    print("| phrasing | condition | " + " | ".join(label for label, _, _ in G) + " |\n|---|---|" + "---|" * len(G))
    rates = [gw.rates("word", "ours", k=25) for _, gw, _ in G]
    for c in ALL:
        for p in rates[0][c][1]:
            print(f"| `{p}` | {D.COND_LABEL[c]} | " + " | ".join(pct(r[c][1][p], 0) for r in rates) + " |")


def headline(R):
    """One row per model and frame: Lindsey's task read both ways, and the workspace paper's protocol
    with a named word, without the reply's last token."""
    aff = ["such_thing_pos", "often_think_pos", "fwiw_pos", "if_i_pos"]
    print("\n### Headline\n")
    print("| model, prompt | concept vectors: no word / think / don't think / plain mentions (x 1e3) | J-lens, Lindsey's "
          "task: no word / think / don't think, top 5 (rank 1) | the same without the reply's last token | J-lens, the "
          "workspace protocol with a named word, without the last token: none / think about / mention, top 5 | think about "
          "vs. mention (pairs) |\n|---|---|---|---|---|---|")
    for label, r in R:
        lo, hi = D.BANDS[r.model]["paper"]
        c = r.z["cos_c"][:, lo:hi + 1].mean(1)
        lv = {k: 1e3 * c[r.cond == k].mean() for k in ["none", "think", "dont_think"] + aff}
        cells = [f"{lv['none']:.0f} / {lv['think']:.0f} / {lv['dont_think']:.0f} / {min(lv[k] for k in aff):.0f} to "
                 f"{max(lv[k] for k in aff):.0f}"]
        for kw in ({}, dict(skip_last=True)):
            b = r.best("ours", **kw)
            cells.append(" / ".join(f"{pct((b[r.mask(k)] <= 5).mean())} ({pct((b[r.mask(k)] <= 1).mean())})"
                                    for k in ("none", "think", "dont_think")))
        gf = {"human": "human_word", "chat": "paper_word"}[r.frame]
        if (D.RESULTS / f"control/{r.model}/modulation_grid_{gf}.json").exists():
            g = D.Grid(r.model, gf)
            b = g.best("ours", skip_last=True)
            cells.append(" / ".join(pct(mean_over_phrasings(g, "word", k, b <= 5)) for k in ("baseline", "focus", "mention")))
            cells.append(paired_kw(g, "word", "ours", skip_last=True)[1])
        else:
            cells += ["", ""]
        print(f"| {label} | " + " | ".join(cells) + " |")


def main():
    R = list(runs())
    common = sorted(set.intersection(*[set(np.flatnonzero(r.tracked)) for _, r in R])) if R else []
    if R:
        print(f"Words tracked by every model: {len(common)} of 50: " + ", ".join(R[0][1].words[w] for w in common))
        for label, r in R:
            print(f"- {label}: {r.tracked.sum()} words tracked")
        headline(R)
        cosine_tables(R)
        lens_tables(R, np.array(common))
        levels(R)
        directive_vs_mention(R)
    word_grid_tables()


if __name__ == "__main__":
    main()
