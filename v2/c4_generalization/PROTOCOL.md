# Criterion 4 — flexible generalization

> Paper §1: *"Flexible generalization. The same representation serves as a valid argument to many
> different downstream computations. In other words, a workspace vector lifted from one context
> and placed in another is correctly operated on by whatever function the new context supplies."*

Paper §3.4 has three pieces, and we run all three: the Fig 18 **case study** (one argument read
by many functions under a single fixed swap), the Fig 19 **systematic swap** (4 categories ×
4 functions × 4 arguments → 16 functions × 12 ordered pairs = 192 trials; 76/192 at α=1,
101/192 at α=2) with the Fig 19-right **workspace-loading** analysis, and the appendix Fig 68
**per-category grids**. We use the paper's OWN released material for the whole criterion
(`ref/jacobian-lens/data/experiments/flexible-generalization.json` — the categories, arguments,
templates, and answers, verbatim). Everything here is produced by `run.py` (results in
`results/`, figure by `figure.py`).

| | gpt2-small (band 7–9) | paper (Sonnet 4.5) |
|---|---|---|
| E1 case study: functions that follow one fixed swap (France→China) | **3/4** (capital→Beijing, language→Chinese, currency→Yuan) | Fig 18: all shown follow |
| E2 raw 192 grid, subtract-and-add, α=1 (capability-penalized) | **34/192 (18%)** | **76/192 (40%)** |
| E2 raw 192 grid, α=2 | **31/192 (16%)** | **101/192 (53%)** |
| E2 capable subset (target function answerable), α=1 | **22/57 (39%)** | ≈40% (the paper's cells are all capable) |
| E2 by category (capable subset, α=1) | countries **67%**, months 20%, numbers 38%, animals 0/3 | countries ≫ months > animals > numbers (0/48) |
| E3 workspace loading, by category | countries **+0.15**, animals +0.15, months +0.09, **numbers +0.07** | countries highest, number words lowest |
| E3 does loading predict swap success (cell level)? | **no** (Spearman −0.41, n=32) | yes |
| floor: bare templates, no frame — capability | **9/64** | (assumed capable) |

**Two-sentence read.** The broadcast/flexible-generalization effect is real in gpt2-small — a
single fixed swap of one argument is correctly operated on by several different downstream
functions (France→China gives Beijing, Chinese, Yuan), and on the subset of functions the model
can actually compute the swap lands the target answer 39% of the time, matching the paper's 40%.
But the two quantitative refinements the paper reports do **not** replicate: α=2 hurts rather
than helps (it overshoots into naming the argument), and workspace loading does not predict
which swaps succeed — what predicts success is the *kind* of function (retrieval functions follow
the swap; successor-type functions do not).

## Setup

**Task.** For each *category* of argument (countries, months, animals, number words), the paper
defines four *functions* that each apply a different operation to the same argument, and swaps the
argument's J-lens vector for another argument's, identically across every function, asking whether
each downstream function reads the swapped-in argument. The four categories, their four arguments,
and their four functions (with answers) are the paper's released data, used verbatim.

**Prompt** (deviation — 2-shot frame; see below). GPT-2 answers only **9/64** of the paper's bare
templates (`floor` in the summary), so — exactly as criterion 1 wrapped `Name a {cat}:` in a
few-shot list and criterion 3 wrapped its two-hop query in a few-shot frame — each function's query
is preceded by a 2-shot frame that teaches the *function* with two demo arguments disjoint from the
four test arguments. The frame lifts capability to 20/64. The whole prompt (a `<|endoftext|>` is
prepended), for `countries / language / France` and, under the identical frame, `.../China`:

```
Most people in Japan speak Japanese. Most people in Italy speak Italian. Most people in France speak
Most people in Japan speak Japanese. Most people in Italy speak Italian. Most people in China speak
```

and for `numbers / successor / five` and `months / next_month / April`:

```
The number that comes right after one is two. The number that comes right after twelve is thirteen. The number that comes right after five is
The month right after January is February. The month right after June is July. The month right after April is
```

The demo arguments never include a test argument, and prepending the frame also means the test
argument is never sentence-initial (so it always tokenizes with a leading space; the paper ignores
its capitalization-token positions, which here do not arise). All 64 prompts are in
`results/prompts.json`. The paper's bare templates are run verbatim, no frame, as the `floor`.

**Data.** The paper's `flexible-generalization.json`, verbatim: 4 categories × {4 args, 4 funcs},
each func a template and its four answers. The only authored text is the 16 demo prefixes
(`FRAMES` in `prompts.py`).

**Gate.** A *cell* is one (category, function, argument). It passes if the model's greedy next
token is the answer's first token: **20/64** pass, very unevenly (countries 5, months 6, animals 1,
numbers 8; per function in `summary.txt`). Retrieval/lookup functions and a few facts gpt2 happens
to know pass; most factual-recall functions (capital, currency, habitat, group) fail because gpt2
does not know the fact — a capability gap a frame cannot fix.

**Lens / swap.** See `../README.md` for the lens and the vocabulary-mean centering. The swap is the
paper's **subtract-and-add** form — the same operation criterion 1 uses for verbal report — because
§3.4 specifies its swap by the α language: *"'double strength' swap ('α = 2', doubling the strength
with which we subtract the source lens vector and add in the target)."* So, with v_s, v_t the unit
centered lens vectors and ⟨v_s,h⟩ read from the clean pass, `h ← h + α·⟨v_s,h⟩·(v_t − v_s)`, applied
at every band layer and every token position, at α=1 and α=2 (`swaps.py`). The Fig 4C **coordinate
swap** (criterion 3's operation) is run alongside for comparison; it tracks the subtract-and-add
form closely and slightly below (all numbers in `summary.txt`).

**Grading.** A swap trial's success = the target argument's answer reaches output top-1 (the paper's
"places the target-appropriate answer at the top of the model's output distribution"), graded on the
answer's first token. A pair is scored only if the target answer differs from the source answer
(*distinct*) and is not present in the frame (*no echo*); 185/192 pairs are clean.

## E1 — the case study (Fig 18)

*"We then swap the J-lens vector for France with that of another country, say China, at every token
position across a band of intermediate layers, applying the identical swap regardless of which prompt
we are in."*

One fixed swap, France→China, applied identically across all four country functions. Three of the
four **follow** the swap:

| function | clean top-1 | swapped top-1 | China's answer (rank clean→swapped) | |
|---|---|---|---|---|
| capital   | Paris  | **Beijing** | Beijing (595→1) | follows |
| language  | French | **Chinese** | Chinese (53→1)  | follows |
| currency  | Franc  | **Yuan**    | Yuan (1366→1)   | follows |
| continent | France | China       | Asia (10→5)     | no (reaches rank 5) |

Notably, capital and currency follow the swap *even though gpt2 cannot answer them for China on its
own* (China's capital reads "Shanghai", its currency isn't produced) — the capital/currency circuit
correctly maps the swapped-in China argument to Beijing/Yuan regardless. This is a clean instance of
the broadcast claim: one argument representation, read correctly by several different functions.

## E2 — the systematic swap (Fig 19 left, Fig 68 grids)

*"We measure the fraction of trials in which the swap places the target-appropriate answer at the top
of the model's output distribution. We find that this succeeds on 76 of 192 trials; by performing a
'double strength' swap … 101 of 192 succeed."*

Over the full 192 grid, subtract-and-add lands the target answer on **34/192 (18%)** at α=1 — about
half the paper's 76/192 (40%). The gap is capability: 44 of the 64 cells fail the gate, so most of
those 192 pairs are asking the model to produce an answer it cannot produce for any argument. On the
**capable subset** — pairs whose target function the model can actually compute (target cell gated,
distinct, no echo) — the swap lands the target answer on **22/57 (39%)**, essentially the paper's
40%. (Restricting further to source-AND-target gated gives 6/34, but that filter is too strict: it
discards the capital/currency case-study successes above, where the *source* fact was one gpt2 gets
wrong; target-gating is the faithful "the function is computable" condition.)

**By category** (capable subset, α=1): countries **10/15 (67%)**, months 3/15, numbers 9/24, animals
0/3. Countries are the most reliable, matching the paper. Number words are **not** the worst here,
unlike the paper — see E3.

**By function** the split is sharp and is the real story (the paper does not report it):

* **retrieval / lookup functions follow the swap** — language 7/9, square 6/6, month-number 3/3,
  capital 2/3, first-letter 3/6, currency 1/3.
* **successor-type functions do not** — next_month 0/12, successor 0/12 (and animals/class 0/3, which
  emits the animal, not its class). Swapping the argument's lens vector injects the new argument, but
  the +1 / next-in-sequence computation is not redirected: it keeps returning the *original*
  argument's successor. The Fig 68 grids (`figure.py` panel D) show this directly — language and
  square are mostly green, next_month and successor are entirely red.

**α = 2 overshoots.** Doubling the swap strength *lowers* success (α=1 34/192 → α=2 31/192), the
opposite of the paper's 76→101. The mechanism is visible in the outputs: at α=1 the swap makes the
model emit the injected *argument word itself* (rather than f(argument)) on 56/185 clean pairs; at
α=2 that rises to 98/185. Double strength pushes the residual so far toward the target argument that
the model simply names it. This is a small-model effect and a clear deviation from the paper.

## E3 — workspace loading (Fig 19 right)

*"we define a concept's workspace loading as the cosine similarity between the residual stream and
that concept's lens vector, averaged over the argument and readout positions in the unmodified
forward pass. Workspace loading of the source argument predicts swap success well. Country arguments
have the highest loading and swap most reliably; number-word arguments have the lowest loading and
swap poorly."*

The **category ordering of loading replicates**: countries +0.15 and animals +0.15 highest, number
words +0.07 lowest — exactly the paper's ordering (countries high, number words low). But the
**link from loading to swap success does not replicate**. At the cell level, loading and swap success
are, if anything, *anti*-correlated (Spearman −0.41 over 32 source cells): the lowest-loading cells
that succeed are surface functions (first-letter, month-number) where the swap trivially re-keys the
answer, while the highest-loading cells that fail are animals/class and countries continent. What
predicts success in gpt2-small is the function type (retrieval vs successor-type), not the argument's
loading. We report this as a genuine non-replication of the paper's second §3.4 claim, most likely
because gpt2's small set of computable functions is dominated by cases where loading and success come
apart; a larger, uniformly-capable function set (as the paper had) may recover the relationship.

## Deviations from the paper

The lens vectors, the released lens, the subtract-and-add swap, the α=1/α=2 strengths, the 192-pair
grid, and the paper's own category/function/answer data are the paper's and are **not** deviations.
These are:

| | |
|---|---|
| base model: **2-shot frame** per function | gpt2 answers 9/64 of the bare templates (the `floor`); the frame teaches the function with demo args disjoint from the test args, lifting capability to 20/64, as criteria 1 and 3 framed their tasks |
| **capability gate**; swap reported on the capable (target-gated) subset as well as the raw 192 | 44/64 cells fail the gate — mostly factual-recall functions gpt2 does not know (capital, currency, habitat, group); an ungated 192 rate is dominated by capability, not broadcast |
| grading on the answer's **first token**; pairs filtered to distinct + non-echoable target answers | several paper answers are multi-token (savanna, arachnid, Valentine) or truncated (5²→"twenty"); first-token grading and the distinct/echo guards keep the metric well-defined |
| **α=2 lowers success** (overshoots into naming the argument) — reported, not hidden | opposite to the paper's 76→101; a small-model effect |
| **loading does not predict success** at the cell level (Spearman −0.41) — reported as non-replication | the paper's second §3.4 claim; the category-level loading ordering does replicate |
| coordinate swap run **alongside** the subtract-and-add form | §3.4 specifies subtract-and-add (the α language); the coordinate swap is a transparency comparison and tracks it |
| vocabulary mean subtracted from each lens vector | inherited convention (changes no readout); see `../README.md` |
| band 7–9 | inherited; see `../README.md` |
