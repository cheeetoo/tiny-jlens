# Protocol — the five workspace criteria in gpt2-small

For each criterion: what the paper says, what we ran, what we found, and every place where the
base model or the tokenizer forced a deviation.  The conventions all five share — the released
lens, the centered J-lens dictionary, the workspace band, ranks, prompts — are in `README.md`;
the evidence for the band itself is the appendix at the end.

Each section is produced by one module in `jl/`, and every number in it is reproduced by
that module's `results/<criterion>/summary.txt`.

- [Criterion 1 — verbal report](#criterion-1--verbal-report)  — `c1_report`
- [Criterion 2 — directed modulation](#criterion-2--directed-modulation)  — `c2_modulation`
- [Criterion 3 — internal reasoning](#criterion-3--internal-reasoning)  — `c3_reasoning`
- [Criterion 4 — flexible generalization](#criterion-4--flexible-generalization)  — `c4_generalization`
- [Criterion 5 — selectivity](#criterion-5--selectivity)  — `c5_selectivity`
- [Appendix — the workspace band](#appendix--the-workspace-band)  — `jl/band.py`

## Criterion 1 — verbal report

> Paper §1: *"Verbal report. When the model is asked what it is thinking about, it names concepts
> represented in the workspace. Swapping one active workspace vector for another changes its
> answer to match."*

Paper §3.1 has four experiments.  We run three; the fourth (injected-thought introspection)
requires an assistant that understands "report what you detect" and has no direct base-model
form.  Everything here is produced by
`jl/c1_report.py` (`python -m jl.c1_report`); the numbers below are in `results/c1_report/`.

| | gpt2-small (band 7–9, gate-passed categories) | Sonnet 4.5 |
|---|---|---|
| 1a Spearman(lens, output) over the 10 candidates | 0.44 / 0.47 / 0.57 at L7 / L8 / L9 | "highly correlated", rising through the workspace |
| 1b swap → target in top-5 (top-1) | **100%** [91, 100] (100%), n = 38; median rank 37 → 1 | 88% |
| 1d swap along the concept vector's J-space component | **97%** | 59% |
| 1d swap along the non-J-space remainder | **0%** [0, 9] | 5% |
| 1e remainder, with J coordinates clamped to clean | **3%** | 0% |
| J-space component's share of concept-vector variance | 23–33% | 6–7% |

All-category numbers (n = 78, including the 5 categories the model cannot answer): 1b 100%,
1d J-part 77%, remainder 0%, clamped 1%.

### Setup

**Data** — the paper's: `ref/jacobian-lens/data/experiments/verbal-report.json` (14 categories
× 14 candidates) and its protocol: *"The prompt is `Think of a {category}. Answer in one word.`;
the model's greedy next token at the final `:` is taken as the answer and used as the swap-out
target. For each of the first 10 listed candidates (skipping the answer itself), swap
answer→candidate across the band at every prompt position. Grading: the swapped-in candidate's
rank in the output distribution at the final `:`."*

**Candidates** — single-token forms only (` Word`, capitalised, as the model answers).  The lens
has one direction per token, so a candidate that GPT-2 splits into several tokens (e.g. " Violin"
→ " Viol"+"in") has no direction and is dropped.  We keep the survivors in the paper's order and
take the first 10 (most categories keep 10; instruments keep 8, rivers keep 7).

**Prompt** — GPT-2 does not follow the instruction on its own (its greedy word at the colon is
` I` / ` The` / `"` in every category), so the request is the last line of a 9-line list whose
first 8 lines are the same request for *other* categories, answered.  Wording `Name a {cat}:`,
chosen for how well the model then answers.  For `sport` (all 14 in `results/c1_report/prompts.json`; a
`<|endoftext|>` token is prepended to every prompt):

```
Name a instrument: Piano
Name a planet: Neptune
Name a tree: Oak
Name a bird: Robin
Name a language: Japanese
Name a profession: Teacher
Name a beverage: Coffee
Name a organ: Heart
Name a sport:
```

**Gate** — a category counts as answered if the model's greedy word at the colon is one of its
candidates; 9/14 pass (fails: fruit → ` Fruit`, tree → ` N`, bird → ` Blue`, profession →
` Sports`, organ → ` Piano`).  Source = the greedy word (for the 5 failed categories, the
highest-ranked candidate instead); targets = the 10 candidates; the analysis is restricted to
targets that start at output rank ≥ 11, as in the paper's Fig. 6.

**Lens** — see `README.md`.  The J-lens vectors are the rows of W_U J_L (the paper's
definition); we additionally subtract the vocabulary mean from each, which changes no readout
and only affects the geometry the swap acts on.  Band = layers 7–9.

### 1a — correlation

*"We apply the J-lens at the token position immediately before the name is produced … the
ordering of the reported words is … highly correlated with the ordering among the lens tokens,
and … this correlation increases towards the end of the workspace."*  Per category, Spearman ρ
between the lens logits and the output logits of the 10 candidates at the colon.  Mean over
categories: ≈ 0 through L0–4, 0.22 (L5), 0.36 (L6), **0.44 / 0.47 / 0.57** (L7–9), 0.63 (L10).

### 1b — the swap

*"At all token positions, we swap the lens vector of the model's spontaneously chosen item with
that of a different item from the same category that was not in the top-10 of the model's
possible outputs … we subtract the projection onto the Soccer lens vector and add an
equal-magnitude projection onto the Rugby lens vector."*

The operation, exactly as that sentence describes it: at each of layers 7, 8, 9 and every token
of the prompt, take how far the residual extends along the source word's (unit) lens direction —
call it *a* — and move the residual by −*a*·(source direction) + *a*·(target direction).  *a* is
measured from a clean run; the model is then re-run with this applied everywhere, and the answer
read at the colon.  Target reaches the top-5 on 100% of trials (top-1 100%), gate-passed and
overall; the median target moves from output rank 37 to rank 1.

(The paper's Methods section describes the same intervention a second way — as reading the two
oblique coordinates of the residual along the source and target directions and exchanging them.
The two coincide when the target word is absent before the swap.  In gpt2-small the target is
not absent: at "Name a sport:" every sport word already leans along a shared "a sport comes
next" direction, so a same-category target already carries about 81% of the source's projection,
and merely exchanging the two coordinates barely moves the residual (63% top-5).  The
subtract-and-add form above is the paper's description of *this* experiment and has no free
parameter, so it is what we use.)

### 1d/1e — is the J-space privileged?

*"recording the residual stream activation prior to the Assistant's response to the prompt
'Tell me about {concept}', mean-subtracted over a baseline set of 100 other concepts … a
J-space component, the non-negative combination of its top k=16 J-lens vectors found by
gradient pursuit, and a non-J-space component, the remainder … substituting each component for
the J-lens vectors used previously, with every perturbation rescaled to the same magnitude …
clamping the relevant J-lens coordinates to their clean-pass values at every position and layer,
so that the concept cannot re-enter the J-space."*

For each candidate word we build a **concept vector**: run `Tell me about {concept}.`, take the
residual at the final token, and subtract the average of the same for 100 other candidate words.
A non-negative pursuit (k = 16) splits it into a **J-space part** (a sum of 16 lens directions)
and the **remainder**.  We then redo the swap using each part in place of the lens directions —
same push length at every layer and position as the 1b swap, only the direction differs.  The
J-space part reproduces the swap (97%); the remainder does nothing (0%); and clamping (holding
the source, target, and both parts' lens coordinates at their clean values, at every layer and
position, so the concept cannot re-enter the J-space) leaves the remainder at 3%.  The J-space
part does this while holding only 23–33% of the concept vector's length.

### Deviations from the paper

These are places where the base model or the tokenizer forced a choice; the lens vectors, the
released lens, and the swap operation are the paper's and are **not** deviations.

| | |
|---|---|
| base model: 8-shot list frame, `Name a {cat}:` | GPT-2 does not follow the instruction on its own; the frame was chosen for how well the model then answers |
| capability gate; the 5 categories it cannot answer are reported separately | 9/14 categories answer |
| single-token candidates only; first 10 of those | the lens has one direction per token; instruments keep 8, rivers 7 |
| vocabulary mean subtracted from each lens vector | GPT-2's unembedding rows share a large common component (raw lens vectors have mean pairwise cosine 0.99); subtracting the mean removes it and changes no readout |
| band 7–9 | checked separately (the band appendix below) |
| concept vectors read at the `.` of `Tell me about {c}.` (no Assistant turn); our non-negative pursuit stands in for the paper's unspecified "gradient pursuit" | base model; method not specified in the paper |
| introspection experiment run only as a base-model attempt | the paper's protocol needs an instruction-following assistant |

---

## Criterion 2 — directed modulation

> Paper §1: *"Directed modulation. When instructed to hold a concept in mind, or perform mental
> calculations, the model is capable of activating and computing with workspace vectors,
> independent of its outputs. In addition, information that is not typically represented in the
> workspace can be pulled in when the task requires it."*

Paper §3.2 (+ appendices "Modulation prompt sensitivity", "Additional modulation examples",
"Directed modulation affects the J-space more than other representations") has four threads. We
run all four. Everything here is produced by `jl/c2_modulation.py`
(`python -m jl.c2_modulation`); the numbers below are in `results/c2_modulation/`.

| | gpt2-small (band 7–9) | Sonnet 4.5 |
|---|---|---|
| 2a  mention < baseline (priming) | **86%** (p=1e-41) | baseline ≈ 0; a bare mention primes strongly |
| 2a  focus < mention (instructed activation) | **76%** (p=2e-22) | focus adds modestly over mention |
| 2a  dismissal ("ignore") < mention (downward control) | **50%** (p=0.96, none) | ignore suppresses below mention |
| 2a  negated ("don't think") < mention (white bear) | **84%** (p=2e-37) | don't-think ≈ mention (white bear) |
| 2a  word reaches the workspace (band top-25), focus | 13% | target reaches band top-1 on a substantial fraction |
| 2b  model computes the answer at all (`{expr} =` greedy) | **0/24** | present; rises with model size |
| 2c  property label summoned into the lens under the naming question | **0 positions** (n=7 items) | appears at a few stimulus positions under Q2 |
| 2d  "imagine French" header: share of a real French sentence's effect | lens **38%**, J-orth probe **8%** | instruction moves the lens, not the J-orth probe |

**One-line reading.** GPT-2's J-space is *modulable upward* — a bare mention primes a word into the
band, an instruction to focus adds more on top, and "don't think about it" backfires (the
white-bear effect) — but it has **no downward control** ("ignore X" does nothing beyond mention),
**cannot hold a silent computation** (it can't do the arithmetic at all), and **cannot summon a
property's name on demand** (it uses properties automatically but never names them in the
workspace). The one clean privileging result (2d) shows the instruction writes to the J-space
specifically. A qualified partial pass, with the boundary between what does and doesn't work
sharply drawn.

### Setup

**Data** — the paper's, from `ref/jacobian-lens/data/experiments/`:
`directed-modulation.json` (24 instruction phrasings in four `group`s, 20 carrier sentences, 24
math problems, 22 topic categories) and `top-down-summoning.json` (7 paired-question items). The
imagine materials (2d) are ours (see the deviations table).

**The frame** (deviation). The paper runs a chat: the user gives an instruction about a target X
and asks the model to copy an unrelated carrier sentence; the lens is read over the model's copy,
where the surface text is the carrier, unrelated to X. GPT-2 has no chat turns, so we use a copy
frame and read over the second (copied) occurrence of the carrier. For the concept `lemon`, the
carrier `The old painting hung crookedly on the wall.`, and the focus phrasing (a
`<|endoftext|>` is prepended; the **bold** span is where the lens is read):

```
Write "The old painting hung crookedly on the wall." Think about lemon while you write. "**The old painting hung crookedly on the wall.**
```

**Conditions** — the paper's, via the reference data's `phrasings` / `group_kind`. Five
conditions; the `baseline` is the ≈0 anchor, `mention` is the priming control that isolates the
instruction's effect (this control is the main thing missing from the earlier attempt):

| condition | kind | example phrasing | # phrasings |
|---|---|---|---|
| baseline | — | *(no instruction; X absent from the prompt)* | 1 |
| mention | control | `lemon.` / `(lemon)` / `lemon came up in conversation.` | 6 |
| focus | focus | `Think about lemon while you write.` | 5 |
| dismissal | suppress | `Ignore lemon.` / `lemon is irrelevant.` | 6 |
| negated | suppress | `Don't think about lemon.` / `Whatever you do, do not think about lemon.` | 7 |

Phrasings are pooled within a condition; per-`(word, carrier)` we take the median rank over the
condition's phrasings before pairing. `dismissal` and `negated` are the two "suppress" groups the
paper separates in the appendix (dismissal = "ignore/irrelevant", negated-think = "don't think").

**Targets** — single-token forms only (` word`, ` Word`), as everywhere in this repo: the lens
has one direction per token. **Primary form: name and track the concept word itself** (30
single-token words drawn from the reference topic members) × 12 carriers = 360 `(word, carrier)`
pairs. See the deviations table for why we name the word rather than the category.

**Metric** — the paper's is "target reaches J-lens top-1 at any (layer, position)" over the
workspace band. At 124M that is ≈0 (`hit@1` column), so, as in criterion 1, the result lives in
the graded rank: we report `hit@{1,5,10,25}`, the median best band rank, and the paired contrasts
between conditions (the actual result). **Held-not-spoken:** the `medRank(held)` column restricts
to copy positions where the tracked word is *not* in the model's output top-10, so lens presence
is genuinely "held, not about to be said"; the `blurt@10` column reports how often it *is* about
to be said.

**Layers** — headline metrics use the workspace band 7–9 only. We print L6 and L10 alongside to
show localization. **Important caveat (2a):** the concept-modulation effect strengthens
monotonically toward the motor layer L10 (where the lens ≈ the model's output), so it is strongest
*outside* the band. We keep L10 out of every headline number and lean on the held-not-spoken
control; see 2a below.

**Lens / centering / band** — see `README.md`. Unchanged from criterion 1.

### 2a — instructed hold-in-mind (concept)

> §3.2 / Fig 10: *"Under the 'think about X' instruction, the target appears in the lens on a
> substantial fraction of trials … The baseline rate is approximately zero … Under the ignore
> instruction, target presence is substantially lower than under the focus instruction, but it is
> not zero."* App Fig 65: *"just mentioning the target places it in the J-space on most trials, and
> an explicit instruction to focus adds only modestly on top … the ignore condition suppresses the
> target well below mention … forbidding the thought leaves the target in the J-space at roughly
> the mention rate, the 'white bear' effect."*

We run the five conditions and pair them per `(word, carrier)`. The results reproduce the paper's
*qualitative* structure and expose where a 124M base model diverges:

- **baseline ≈ 0** — the word is absent from the band on context alone (hit@25 0.02, median rank
  721). This is the anchor the paper relies on, and it holds.
- **mention < baseline, 86%** — a bare mention primes the word into the band. Exactly the appendix
  finding. *(This is the confound the earlier attempt had no control for: its "instruction beats
  no-instruction" is mostly this priming, since the word is in the instruction prompt and absent
  from the no-instruction prompt.)*
- **focus < mention, 76%** — an explicit focus instruction moves the word further, *beyond* mere
  mention. This is the genuine directed-activation signal, cleanly separated from priming.
- **dismissal not < mention, 50% (p = 0.96)** — "ignore X" does **nothing** relative to mention.
  Unlike Sonnet, GPT-2 has no downward control: it cannot suppress a primed concept on instruction.
- **negated < mention, 84%; negated < dismissal, 84%** — "don't think about X" makes the word the
  *most* present of any condition. The white-bear backfire, stronger here than in Sonnet.

**The motor-layer caveat.** Per-layer median best rank (focus): L7 1032 → L8 754 → L9 386 → L10
**98**. The effect grows toward the output and is strongest at the motor layer, where the lens is
essentially the output prediction. Every headline number above is band-only (7–9), and the
held-not-spoken column shows the band effect survives excluding positions where the word is about
to be output (focus held median 394 vs baseline 721) — so it is not *purely* the model preparing
to say the word. But the word reaches the workspace proper (band top-25) on only 13% of focus
trials: the modulation reliably improves the word's standing, without robustly seating it in the
workspace.

**Main-text form (name the category, track members).** The paper's Fig 10 category family names
the *category* ("citrus fruits") and tracks its *members* ("orange"). We run this too. It also
modulates (focus < baseline 79%, focus < mention 67%), but it tracks many member tokens at once,
so its absolute hit rate is not comparable to the single-word form; we report only the
target-count-invariant paired contrast.

### 2b — instructed mental computation (math)

> §3.2 / Fig 10: *"mentally evaluating a mathematical expression … a trial is positive if the
> target reaches J-lens top-1 … this tends to increase with model size."*

Capability-gated. On the reference's 24 problems, GPT-2 gets **0/24** right greedily when asked
directly (`4 * 2 =` → `:`); the answer sits near rank 8 — present but never produced. During
silent copying under a "work it out in your head" instruction, the answer never enters the band
(hit@25 0.00, median rank ~4000). The model cannot compute these, so there is nothing for the
J-space to hold. This is a capability floor, consistent with the paper's own finding that the
effect grows with model size (and GPT-2 is far below Haiku, the paper's weakest model).

### 2c — implicit task-demand ("pulled in when the task requires it")

> §3.2 / Fig 11: *"the same stimulus, preceded by one of two questions … We then apply the J-lens
> at every token position within the stimulus, and record at how many positions the property's
> label appears among the top lens tokens … under the next-word question, neither `adjective` nor
> `adj` appear … in response to the name-the-property question, adjective-related J-lens readouts
> appear at 3 stimulus positions."*

The faithful paired-question protocol (7 items from `top-down-summoning.json`): identical
stimulus, preceded by Q1 (predict the next word) or Q2 (name the latent property). The automatic
task works — the model predicts the right next word on **7/7** items — so the property *is* being
used. But the property label (`past`, `adjective`, `plural`, …) enters the band lens at **0**
positions under *either* question. GPT-2 uses properties implicitly but cannot summon their names
into the workspace on demand: the "pulled in when the task requires it" clause fails at this scale.
(The causal swap the paper pairs with this readout is moot when the label is absent to begin with.)

### 2d — is the J-space privileged? (imagine)

> §A "Directed modulation affects the J-space more than other representations": *"the instruction
> to imagine the property typically raises the property's name in the J-lens by several standard
> deviations … the same instruction leaves the J-orthogonalized property probe essentially at
> baseline … whereas a real positive stimulus moves the probe by three to six standard
> deviations."*

The analog of criterion 1's 1d/1e. A header claims an English sentence is French; we measure (a)
the lens log-prob of ` French` over the sentence tokens and (b) the sentence's projection onto a
**J-orthogonalized** French-vs-English probe (the mean-difference of held-out French/English
passages, with its top-16 J-lens component removed by non-negative pursuit). Conditions: neutral
header (baseline), claim header ("imagine … French"), and a real French sentence under a neutral
header.

The dissociation reproduces. Relative to a real French sentence, the claim produces **38%** of the
lens-`French` effect but only **8%** of the J-orthogonalized-probe effect: the instruction writes
`French` into the J-space while barely touching the model's underlying representation of the text,
whereas real French text moves the underlying representation ~12× more than the claim does. (The
claim's probe effect is small but nonzero — the header contains the token "French" — so this is a
dissociation of degree, as in the paper.)

### Deviations from the paper

These are places where the base model or the tokenizer forced a choice; the lens vectors, the
released lens, the conditions, and the reference prompt data are the paper's and are **not**
deviations.

| | |
|---|---|
| base-model copy frame `Write "{s}" {instruction} "{s}`, lens read over the copied span | GPT-2 has no chat turns; this is the base-model analog of "instruct, then copy an unrelated sentence, read over the copy" |
| **primary form names and tracks the concept word** (App Fig 65 framing) rather than naming the category and tracking unnamed members (main-text Fig 10) | the members-never-named form is fine but has a variable target count per category, so the mention-vs-instruction contrast is cleanest when one word is named and tracked; the category form is reported alongside |
| single-token target forms only (` word`, ` Word`); 30 concept words × 12 carriers | the lens has one direction per token |
| headline metrics use band 7–9; the effect is strongest at motor layer L10 and is reported but excluded | the paper places the workspace before the motor layers; at L10 the lens ≈ output, so it is not "held, not spoken" |
| graded rank + paired contrasts, not the paper's top-1 hit (which is ≈0 at 124M) | same as criterion 1; the structure lives in the ranks |
| held-not-spoken via output-top-10 exclusion (our operationalization) | the paper reads at positions where the surface text is unrelated; we make "not about to be spoken" explicit |
| 2b/2c reported as capability-gated: the model cannot compute the arithmetic, nor summon a property name | base model; the paper's own effects grow with model size |
| 2d imagine materials (English/French sentence pairs, held-out probe passages) are ours | the paper's §A materials are not in the released data; the design is the paper's |
| vocabulary mean subtracted from each lens vector; band 7–9 | inherited from criterion 1 / `README.md`; changes no readout |

---

## Criterion 3 — internal reasoning

> Paper §1: *"Internal reasoning. Workspace vectors can be used to represent the value of
> intermediate computations, when the model chains inferential steps or composes plans, and
> intervening on them is sufficient to redirect the conclusion."*

Paper §3.3 has, in effect, six pieces. We run five; the sixth (the two-step arithmetic of
Fig 17, `(4+17)*2+7`) has no base-model form — gpt2-small cannot do the arithmetic (0/… ; the
intermediates 21/42/49 never form). Everything here is produced by `jl/c3_reasoning.py`
(`python -m jl.c3_reasoning`); the numbers below are in `results/c3_reasoning/`.

| | gpt2-small (band 7–9) | paper (Sonnet 4.5 unless noted) |
|---|---|---|
| E1 unspoken intermediate in lens top-10 at some band layer | **84%** (36/43 unspoken items) | surfaces at intermediate layers (Fig 12) |
| E3 swap → target answer reaches top-1 | **71%** [68,73] (n=999) | Haiku **54%**, Sonnet **70%**, Opus **70%** |
| E4 intermediate swap onsets earlier than answer swap | **33% of depth** earlier | ~**17%** earlier |
| E5 raw J-lens coordinate swap → top-1 | **71%** | 60% |
| E5 swap along the probe's **J-space** component | **95%** | 61% |
| E5 swap along the **non-J-space** component | **1.5%** | 28% |
| E5 non-J, with J coordinates clamped to clean | **0%** | 6% |
| E5 J-space component's share of probe variance | 27–57% (L7–9) | 10–15% |

### Setup

**Task.** A two-hop factual query whose answer requires first inferring an *unspoken* bridge
entity. Two relation families, inverses through the country (which is never in the prompt or the
answer):

```
lang_capital :  language → (country) → capital
cap_language :  capital  → (country) → language
```

**Prompt** (deviation — few-shot frame; see below). GPT-2 cannot resolve the paper's riddle
phrasings on its own, so each query is the last line of a short few-shot frame that teaches the
relation with *other* countries. The whole prompt (a `<|endoftext|>` is prepended), for one
`lang_capital` item (intermediate = France, answer = Paris):

```
In the country where people speak Arabic, the capital city is called Cairo. In the country where people speak Hebrew, the capital city is called Jerusalem. In the country where people speak French, the capital city is called
```

and one `cap_language` item (intermediate = France, answer = French):

```
The country governed from Cairo has one main language, namely Arabic. The country governed from Tokyo has one main language, namely Japanese. The country governed from Paris has one main language, namely
```

The intermediate (France) is never named; the shot countries (Egypt/Israel/Japan) differ from
every test item, and any item whose country or answer collides with the shot text is dropped
(so the answer can't be echoed). Full list in `results/c3_reasoning/prompts.json`.

**Data.** The country fact table (the module's prompt-material section) is world knowledge, filtered at build time to
single-token capitals/answers. The paper's own two-hop set
(`ref/.../probe-swap.json`) is run verbatim as the capability-floor check (`floor` in the
summary).

**Gate.** An item counts if the model's greedy next token is the answer: **48/53** pass. The
swap experiments additionally restrict to target answers starting at output rank ≥ 10 (the
paper's Fig 6 rule, reused), and evaluate every valid same-family swap partner per item.

**Lens / swap.** See `README.md` for the lens and the vocabulary-mean centering. The swap is
the paper's **coordinate swap** (Methods "patching in lens coordinates", Fig 4C), clamped to the
swapped clean-pass values across the band — the operation §3.3 names (`jl/interventions.py`). This is
*not* the subtract-and-add form used for criterion 1. Both operations work here (`python -m jl.c3_reasoning swap_ops`,
same 999 trials): coordinate swap 705/999 = 70.6%, subtract-and-add 854/999 = 85.5%. We report
the coordinate swap because §3.3 names it and it is the more conservative number. (An earlier
note here said subtract-and-add gave ≈0%; that was the raw, uncentered gauge — see `README.md`.)

### E1 — the unspoken intermediate surfaces in the band

*Fig 12: "For each prompt, we first confirm that the intermediate concept appears in the J-lens
at intermediate model layers … even though the word never appears in the prompt or the output."*

At the answer position we read the lens rank of the intermediate and three controls, per layer:
**answer** (the imminent output / motor signal), **arg** (the surface cue that *is* in the
prompt — an echo control), and **null** (a random other country — a generic-category control).
The intermediate is absent through the first two-thirds of the model and drops into the lens
exactly at the band (median rank 836 → **13** → **4** at L7 → 8 → 9), while `null` never enters
(median rank ≈ 400–1300 at every layer). So it is the *specific computed* bridge entity, not
generic "country-ness" or an input echo. It is genuinely unspoken: 43/48 items have the
intermediate at output rank ≥ 10, and 36 of those 43 reach lens top-10 at some band layer.

### E2 — case study (Fig 13)

The clean-vs-swapped panel the paper leads with (spider→ant → 8→6). The paper's riddle phrasings
score 0/6 on gpt2-small, so the case study is a country two-hop: `France → China` swaps the
intermediate lens vector across the band; the model's answer flips `Paris → Beijing`
(Paris log-prob −1.15 → −5.51, Beijing −8.45 → −2.12).

### E3 — the systematic swap (Fig 15 left)

*"a set of 50 two-hop factual prompts … choosing the swap target at random from within the same
category … We measure the fraction of trials in which the swap moves the target-appropriate
answer to the top of the model's output distribution."*

Coordinate swap of the intermediate (France → China) across the band at every prompt position;
success = the target's answer (Beijing) is the model's top-1 output. **71%** over 999 trials
(48 items × their valid same-family partners), median rank of the target answer 88 → 1. Squarely
in the Sonnet/Opus range (70%), above Haiku (54%). We evaluate *all* valid partners per item
(not one random one) for statistical power, as criterion 1 did with its candidates.

### E4 — is the intermediate a smuggled-in answer? (Fig 15 right) — the depth control

*"A possible confound is that the intermediate's J-lens vector already contains the answer …
we compare the effect of swapping the J-lens vectors for the intermediate concepts vs. the
target answers, applying the swap at different layer ranges. If the intermediate swap were
acting through a smuggled-in answer component, both interventions would produce an effect at
the same depth; instead, the intermediate swap takes effect a median of approximately 17
percent earlier than the answer swap."*

This is the condition the earlier implementation was missing (it had a different two-question
control instead). For each item we apply the coordinate swap at a **single layer** and sweep the
layer, measuring the log-prob pushed onto the target answer, for the intermediate swap
(France→China) and the answer swap (Paris→Beijing) separately. The intermediate swap is already
strongly effective at the early/band layers (+2.8 at L7) where the answer swap does nothing or
hurts (+0.0 at L7, negative at L5–6); the answer swap only bites at L8+. Onset (half of a
trial's max effect): median L4 for the intermediate vs L8 for the answer — **33% of depth
earlier**, more pronounced than the paper's 17%. So the intermediate is represented and used
before the answer is computed; it is not a smuggled answer.

(Caveat: a single-layer coordinate swap *injects* the concept, so it can act at a layer before
the concept naturally forms; the load-bearing comparison is intermediate-vs-answer at the same
depths, which is unambiguous — the two effect curves have clearly different shapes.)

### E5 — is the J-space privileged? (Fig 16)

*"we fit a probe for the unspoken intermediate: the mean residual-stream activation over a set
of prompts that imply the same intermediate through different surface cues and ask different
questions about it, minus the mean over all intermediates. We decompose each probe … into a
J-space component (a non-negative combination of k=25 J-lens vectors …) and a J-orthogonal
remainder … exchanging the intermediate's probe for an alternative along the full probe
direction, along only its J-space component, or along only its remaining non-J-space component
… with the J-space coordinates … clamped to their clean-pass values."*

For each country we build the probe from six cue prompts that imply it via its capital / language
and ask about a *different* attribute (so the country name is never the next token), minus the
grand mean over countries. A non-negative pursuit (k = 25) splits it into a J-space component
and a remainder. We then swap along the full probe, the J-space part, and the remainder — each
rescaled to the full-probe magnitude ("every perturbation rescaled to the same magnitude") — and
add the clamp control. The J-space part carries the effect (**95%**, matching the raw lens swap's
71% and exceeding it once magnitude-matched); the remainder does essentially nothing (**1.5%**);
and clamping the J coordinates removes even that small residual (**0%**, and the model returns to
its clean answer — verified: the residual effect is mediated by the J-space). The J-space
component holds 27–57% of the probe's variance (vs 10–15% in the paper — the same scale effect
seen for criterion 1's concept vectors).

### Deviations from the paper

The lens vectors, the released lens, the coordinate-swap operation, and the k=25 pursuit split
are the paper's and are **not** deviations. These are:

| | |
|---|---|
| base model: **few-shot two-hop frame** | gpt2-small answers **7–9/90** of the paper's own two-hop prompts; the frame teaches the relation (as criterion 1's list frame taught the report format), and the intermediate is still unspoken (E1) and computed (`null` control) |
| **country families only**; the paper's riddle phrasings (spider→legs) dropped | 0/6 capability on the bare riddles — the capability floor |
| **coordinate swap** (Fig 4C), not the subtract-and-add of criterion 1 | §3.3 names the coordinate swap; it is also the more conservative of the two on gpt2-small (70.6% vs 85.5% for subtract-and-add, `python -m jl.c3_reasoning swap_ops`) |
| swap graded over **all** valid same-family partners per item (n = trials) | statistical power, as criterion 1 did with its 10 candidates; the paper uses one random partner |
| **probe cues authored** for the base model; our non-negative pursuit stands in for "gradient pursuit" | the paper's probe-construction prompts are not released; cues imply the country and ask a different attribute, name never next token |
| J-space share of probe variance 27–57% vs paper 10–15% | GPT-2's 768-dim residual + k=25 lens directions capture more; same scale effect as criterion 1 (23–33% vs 6–7%) |
| depth control (E4) uses **single-layer** coordinate swaps, onset = half-max | the paper's "different layer ranges"; single-layer injection can act before a concept forms, so the intermediate-vs-answer contrast (not the absolute onset) is the claim |
| vocabulary mean subtracted from each lens vector | inherited convention (changes no readout); see `README.md` |
| band 7–9 | inherited; see `README.md` |
| two-step arithmetic (Fig 17) not attempted | gpt2-small cannot compute `(4+17)*2+7`; the intermediates never form |

---

## Criterion 4 — flexible generalization

> Paper §1: *"Flexible generalization. The same representation serves as a valid argument to many
> different downstream computations. In other words, a workspace vector lifted from one context
> and placed in another is correctly operated on by whatever function the new context supplies."*

Paper §3.4 has three pieces, and we run all three: the Fig 18 **case study** (one argument read
by many functions under a single fixed swap), the Fig 19 **systematic swap** (4 categories ×
4 functions × 4 arguments → 16 functions × 12 ordered pairs = 192 trials; 76/192 at α=1,
101/192 at α=2) with the Fig 19-right **workspace-loading** analysis, and the appendix Fig 68
**per-category grids**. We use the paper's OWN released material for the whole criterion
(`ref/jacobian-lens/data/experiments/flexible-generalization.json` — the categories, arguments,
templates, and answers, verbatim). Everything here is produced by `jl/c4_generalization.py`
(`python -m jl.c4_generalization`); the numbers below are in `results/c4_generalization/`.

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

### Setup

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
`results/c4_generalization/prompts.json`. The paper's bare templates are run verbatim, no frame, as the `floor`.

**Data.** The paper's `flexible-generalization.json`, verbatim: 4 categories × {4 args, 4 funcs},
each func a template and its four answers. The only authored text is the 16 demo prefixes
(`FRAMES` in the module's prompt-material section).

**Gate.** A *cell* is one (category, function, argument). It passes if the model's greedy next
token is the answer's first token: **20/64** pass, very unevenly (countries 5, months 6, animals 1,
numbers 8; per function in `results/c4_generalization/summary.txt`). Retrieval/lookup functions and a few facts gpt2 happens
to know pass; most factual-recall functions (capital, currency, habitat, group) fail because gpt2
does not know the fact — a capability gap a frame cannot fix.

**Lens / swap.** See `README.md` for the lens and the vocabulary-mean centering. The swap is the
paper's **subtract-and-add** form — the same operation criterion 1 uses for verbal report — because
§3.4 specifies its swap by the α language: *"'double strength' swap ('α = 2', doubling the strength
with which we subtract the source lens vector and add in the target)."* So, with v_s, v_t the unit
centered lens vectors and ⟨v_s,h⟩ read from the clean pass, `h ← h + α·⟨v_s,h⟩·(v_t − v_s)`, applied
at every band layer and every token position, at α=1 and α=2 (`jl/interventions.py`). The Fig 4C **coordinate
swap** (criterion 3's operation) is run alongside for comparison; it tracks the subtract-and-add
form closely and slightly below (all numbers in `results/c4_generalization/summary.txt`).

**Grading.** A swap trial's success = the target argument's answer reaches output top-1 (the paper's
"places the target-appropriate answer at the top of the model's output distribution"), graded on the
answer's first token. A pair is scored only if the target answer differs from the source answer
(*distinct*) and is not present in the frame (*no echo*); 185/192 pairs are clean.

### E1 — the case study (Fig 18)

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

### E2 — the systematic swap (Fig 19 left, Fig 68 grids)

*"We measure the fraction of trials in which the swap places the target-appropriate answer at the top
of the model's output distribution. We find that this succeeds on 76 of 192 trials; by performing a
'double strength' swap … 101 of 192 succeed."*

Over the full 192 grid, subtract-and-add lands the target answer on **34/192 (18%)** at α=1 — about
half the paper's 76/192 (40%). The gap is capability: 44 of the 64 cells fail the gate, so most of
those 192 pairs are asking the model to produce an answer it cannot produce for any argument. On the
**capable subset** — pairs whose target function the model can actually compute (target cell gated,
distinct, no echo) — the swap lands the target answer on **22/57 (39%)**, essentially the paper's
40%. Restricting further to source-AND-target gated gives 6/34 (18%), but that is composition, not
gating: 24 of those 34 pairs are the two successor functions (next_month, successor), which go 0/24
under every gating. On the other functions gating barely matters — both-gated 6/10 (60%),
target-gated 22/33 (67%), ungated 32/161 (20%) — and all 22 target-gated hits are on non-successor
functions (`python -m jl.c4_generalization gating`). The case-study pairs above are themselves mixed-gating (capital:
source gated, target not; currency: the reverse), which is the argument for target-gating: the
function consumes the swapped-in argument even when gpt2 cannot answer one of the cells on its own.

**By category** (capable subset, α=1): countries **10/15 (67%)**, months 3/15, numbers 9/24, animals
0/3. Countries are the most reliable, matching the paper. Number words are **not** the worst here,
unlike the paper — see E3.

**By function** the split is sharp and is the real story (the paper does not report it):

* **retrieval / lookup functions follow the swap** — language 7/9, square 6/6, month-number 3/3,
  capital 2/3, first-letter 3/6, currency 1/3.
* **successor-type functions do not** — next_month 0/12, successor 0/12 (and animals/class 0/3, which
  emits the animal, not its class). Swapping the argument's lens vector injects the new argument, but
  the +1 / next-in-sequence computation is not redirected: it keeps returning the *original*
  argument's successor. The Fig 68 grids in `results/c4_generalization/results.json` show this directly — language and
  square are mostly green, next_month and successor are entirely red.

**α = 2 overshoots.** Doubling the swap strength *lowers* success (α=1 34/192 → α=2 31/192), the
opposite of the paper's 76→101. The mechanism is visible in the outputs: at α=1 the swap makes the
model emit the injected *argument word itself* (rather than f(argument)) on 56/185 clean pairs; at
α=2 that rises to 98/185. Double strength pushes the residual so far toward the target argument that
the model simply names it. This is a small-model effect and a clear deviation from the paper.

### E3 — workspace loading (Fig 19 right)

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

### Deviations from the paper

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
| vocabulary mean subtracted from each lens vector | inherited convention (changes no readout); see `README.md` |
| band 7–9 | inherited; see `README.md` |

---

## Criterion 5 — selectivity

> Paper §1: *"Selectivity. The workspace comprises a small subset of the total representational
> content of the model's activations. It is required for only a fraction of the model's
> behavior, and in particular is not involved in pervasive, routine processing like text parsing
> or grammatical fluency."*

The definition has **two clauses**: (i) the J-space is a *small subset* of representational
content, and (ii) it is *required for flexible but not automatic* processing.  §3.5 is the home
of clause (ii) and is the bulk of this criterion; clause (i) is established structurally in §4.2,
included here compactly (as S1) so the definition is covered end to end.  Everything is produced
by `jl/c5_selectivity.py`
(`python -m jl.c5_selectivity`); the numbers below are in `results/c5_selectivity/`.

| | gpt2-small (band 7–9) | paper (Sonnet 4.5 unless noted) |
|---|---|---|
| **S2a** two-hop reasoning, *selective damage* (random − J) at light ablation | **+0.62** (J 0.21 vs random 0.83) | multihop → ~0 under ablation (Fig 22) |
| **S2a** one-hop recall, selective damage at light | **+0.21** (intermediate) | one-step recall ~automatic; TriviaQA (generative) breaks |
| **S2a** induction / copy, selective damage at light | **+0.05** (survives) | parsing / extraction survive |
| **S2a** next-token match (wikitext), selective damage at light | **+0.09** (survives) | pretraining top-1 match preserved (Fig 22) |
| **S2b** deliberate report follows the language swap (α=1.5) | **90%** | report/flexible follow on ~every trial |
| **S2b** automatic continuation keeps its own language (α=1.5) | **62%** | continuation/anomaly unaffected |
| **S1** top-25 J-space share of activation variance | **~30%** (≈70% lies outside) | excess over random <10%; occupancy ~25 |

**One-line reading.** Whole-J-space ablation is *selective*: it removes the two-hop reasoning
task far more than a matched-norm random perturbation does (+0.62), while leaving induction and
next-token prediction essentially untouched (+0.05, +0.09), with one-hop recall in between.  The
same-latent language experiment shows the effect directionally too — the same swap redirects the
deliberate report where the automatic continuation resists it — though the dissociation is
**attenuated** at this scale (see S2b).

### Setup

**Ablation (S2a).**  §3.5.2: *"at each token position, across a band of layers, we identify the
k=10 most strongly activated J-lens vectors and zero out the residual stream's projection onto
each … we do not ablate any tokens that appear in the top-10 tokens of a clean forward pass."*
We do exactly this (`jl/interventions.py`): at each (position, band layer) we take the 10 highest lens
tokens that are **not** in the model's clean output top-10, and remove the residual's component
in their span (`h ← h − V V⁺h`).  Selection is read from a single clean pass.  **Strength** =
the layer range, since gpt2's band is only three layers: `light [8]`, `medium [7,8,9]`,
`heavy [6,7,8,9,10]`.  The **matched-norm random control** removes a random subspace rescaled,
per position, to the exact norm the J ablation removes there — isolating *which subspace* from
*how much norm* (the paper's random-direction control, norm-matched as c1/c2 did).

**Battery.**  Four tasks, ordered by how much they depend on assembling inferred content:

- `two_hop` (flexible) — criterion 3's two-hop country chains, **imported verbatim** as the
  paper's own multihop positive control (48 gated items).
- `one_hop` (recall) — the same country facts, one hop (`The capital of {country} is`), few-shot.
- `induction` (automatic) — copy the partner of a token repeated earlier in the context.
- `pretrain_match` (automatic) — agreement with the clean model's next token on wikitext-2
  prose (the corpus the released lens was fit on); the paper's Fig 22 collateral-damage axis.

Accuracy tasks are gated to clean-greedy-correct (so clean = 1.0); `pretrain_match` scores the
fraction of positions whose top-1 is unchanged.

**Language (S2b).**  The paper's released passages,
`ref/…/selectivity-language.json`: eight passages (two each fr/de/es/it) whose language is
evident but never named.  Deliberate task = a few-shot "name the language" cloze (built from
explicit token pieces so the passage span is exact); automatic task = continue the passage.  The
same language-label swap (criterion 3's coordinate swap, `jl/interventions.py`) is applied over the passage
tokens under both.  **Report** graded by whether the reported language flips to the swapped-in
one; **continuation** graded by whether the model still prefers to continue in the passage's own
language, scored against short per-language continuation phrases (a next-token language
classifier is unreliable for the closely-related Romance languages).

**Capacity (S1).**  wikitext activations at band layers, mean-subtracted; the fraction of
variance a non-negative pursuit captures with the top-K J-lens directions vs a same-size random
dictionary; occupancy = the K at which the J marginal gain drops below random's (paper §4.2).

**Lens.**  See `README.md`: the released gpt2-small lens, J-lens vectors = rows of W_U J_L,
vocabulary-mean subtracted (changes no readout), band 7–9.

### S2a — J-space ablation is selective (§3.5.2, Fig 22/24)

*Fig 22: "J-space ablation perturbs the model's next-token prediction substantially less than in
the multihop case … the ablation is targeted."*  We measure each task's score under no ablation
/ J-space ablation / matched-norm random, at each strength.  At **light** ablation:

|task|kind|clean|J|random|selective damage (random−J)|
|---|---|---|---|---|---|
|two_hop|flexible|1.00|**0.21**|0.83|**+0.62**|
|one_hop|recall|1.00|0.73|0.94|+0.21|
|induction|automatic|1.00|0.95|1.00|+0.05|
|pretrain_match|automatic|1.00|0.71|0.80|+0.09|

The J-subspace is what the flexible task needs: removing it drops two-hop accuracy to 0.21 while
a matched-norm random subspace leaves it at 0.83 (+0.62 selective damage), an order of magnitude
more than the automatic tasks (+0.05, +0.09).  One-hop recall is intermediate (+0.21), matching
the paper's own tension — the §3.5 intro calls one-step recall "automatic," but Fig 24 has
generative recall (TriviaQA) among the tasks that *break*; single-token capital recall sits
between parsing and multi-step reasoning.

As the ablation widens (medium → heavy) it eventually damages everything: projecting out ten
directions across five layers at every position is a large perturbation for a 768-dim residual
stream, and past a point even matched-norm random hurts.  The selectivity is therefore read from
(a) the **order** of collapse — flexible first, recall next, automatic last — and (b)
the J-vs-random gap at **light** ablation, where nothing is floored.  Both are the
paper's Fig 22/24 result; the absolute robustness is lower than Sonnet's, as expected at 124M.

### S2b — same latent, two tasks (§3.5.1, Fig 20)

*Fig 20: "the answer follows the swapped lens value under explicit report and flexible
computation, while continuation and anomaly detection are unaffected."*  We apply the **same**
language swap under the deliberate report and the automatic continuation, sweeping strength α:

|α|report flips to swapped language|continuation keeps its own language|
|---|---|---|
|0.5|19%|90%|
|1.0|38%|76%|
|1.5|**90%**|**62%**|
|2.0|90%|52%|
|3.0|100%|33%|

Panel (b): the true-language label is present in the band J-space over the passage at comparable
rank in both prompts (median band-min rank 22 report / 21 continuation), so the continuation
result is about causal *role*, not absence.  The report follows the swap at strengths where the
continuation still keeps its own language — the same latent is causal for the deliberate task,
much less for the automatic one.

**This dissociation is real but attenuated**, and I want to be explicit about why.  The paper
swaps *over the question tokens*, leaving the passage itself intact for the continuation to read;
a base model has no separate question span, so we swap *over the passage*, which unavoidably
perturbs the passage's own continuation.  A strong enough swap (α≥2) therefore does eventually
push the continuation's language too — so unlike Sonnet, gpt2-small's automatic continuation is
not *perfectly* J-independent here.  The clean, unambiguous selectivity result in this criterion
is S2a's ablation battery; S2b corroborates it directionally on a single shared latent.

### S1 — the J-space is a small subset (§4.2, Fig 30)

*Fig 30b: "The excess variance explained is modest, never exceeding 10%, indicating that the
model's activations are dominated by information outside the J-space."*  On wikitext activations,
the top-25 J-space directions capture **~30%** of activation variance (L7 29% / L8 32% / L9 36%),
so roughly **70% lies outside** the J-space — a limited slice, as the definition requires.  A
same-size random dictionary captures a comparable-or-larger share (≈37%), so the J-space is **not
a variance-dominating subspace**; it is a specific *structured* subframe (occupancy — the K at
which it stops beating a random dictionary — is only 2–5, far below Sonnet's ~25, consistent with
a much smaller model).  The sign of the excess-over-random differs from the paper's small
positive: an iid vocabulary-sized random dictionary is a very strong reconstruction control, and
the centered gpt2 lens is more coherent than iid directions; both readings support "small,
specific subset."

### floor — line-length counting (§3.5.1, Fig 21)

The linecount experiment is a **base-model capability floor**.  Over the released passages, a
count token reaches the band J-space top-25 on 0–1 of 11 passages under every condition
(linewrap / direct / first-letter), and the model's greedy answers are `been` / `a` / `the`.
gpt2-small cannot represent character counts, so the count neither enters the J-space nor can be
reported — as with the criterion-2 math floor.

### Deviations from the paper

The lens vectors, the released lens, the top-k ablation, the coordinate swap, and the released
language passages are the paper's and are **not** deviations.  These are:

| | |
|---|---|
| ablation **strength = layer range** (`light [8]`, `medium [7,8,9]`, `heavy [6..10]`) | gpt2's band is only three layers; the paper's light/medium/heavy layer ranges don't map |
| ablation is aggressive at 768-dim: medium/heavy damage automatic tasks too | selectivity is read from the **order** of collapse and the J-vs-random gap at **light**, where nothing is floored |
| **matched-norm** random control (paper: "randomly chosen directions") | the stronger control (isolates subspace from norm), as c1/c2 did |
| battery is base-model-appropriate: two-hop (c3), one-hop, induction, wikitext next-token | gpt2 cannot do the paper's 14-task battery (MMLU, SQuAD, Caesar cipher, sonnets, …); these four cover the flexible↔automatic axis it can do |
| two-hop task **imported verbatim from c3** | the positive control is byte-identical to the multihop eval c3 validated |
| S2b swap applied **over the passage**, not the question tokens | the base model has no separate question span; this perturbs the continuation, so the dissociation is **attenuated** (continuation not perfectly J-independent) |
| S2b report via **few-shot language-ID cloze** (7/8 gate); continuation graded by **language-preference log-prob** | base model can't be instructed; a next-token classifier is unreliable for Romance languages |
| S1 (§4.2) included to complete the definition; random control = iid vocab-sized dictionary | §4 is the structural chapter, out of scope for the other criteria; the excess-over-random sign differs from the paper (see S1) |
| J-space variance share ~30% vs paper <10% | GPT-2's 768-dim residual + k=25 directions capture more; same scale effect as c1 (23–33%) and c3 (27–57%) |
| **line-length counting** run only as a floor; **experiential reports** (§3.5.3) and **naming-vs-avoiding** (App.) dropped | gpt2 can't count characters; the other two need an instruction-following assistant with an experiential register |
| vocabulary mean subtracted; band 7–9 | inherited conventions (change no readout); see `README.md` |

---

## Appendix — the workspace band

`python -m jl.band stats` (paper Fig 28 metrics) and `python -m jl.band cka`
(dictionary similarity, controlling for the dominant direction).  48 wikitext-103 validation sequences × 128 tokens; dictionary metrics on a 3000-token
vocabulary subsample of the centered J-lens vectors.  CPU, ~2 min + ~4 min.

### Layer-wise statistics (`results/band/band.json`)

| L | top-1 agree | top-10 agree | excess kurtosis | autocorr − null | effdim (90% var) |
|---|---|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.5 | +0.001 | 0.13 |
| 1 | 0.00 | 0.01 | 0.6 | -0.000 | 0.16 |
| 2 | 0.00 | 0.01 | 0.6 | +0.000 | 0.19 |
| 3 | 0.00 | 0.01 | 0.4 | +0.002 | 0.20 |
| 4 | 0.00 | 0.02 | 0.5 | +0.004 | 0.22 |
| 5 | 0.00 | 0.03 | 0.4 | +0.002 | 0.22 |
| 6 | 0.01 | 0.04 | 0.4 | +0.007 | 0.26 |
| 7 | 0.02 | 0.10 | 0.4 | +0.014 | 0.30 |
| 8 | 0.04 | 0.13 | 0.5 | +0.021 | 0.35 |
| 9 | 0.11 | 0.29 | 0.6 | +0.023 | 0.43 |
| 10 | 0.24 | 0.51 | 0.5 | +0.020 | 0.50 |
| 11 | 1.00 | 1.00 | 0.4 | +0.008 | 0.66 |

* **top-k agreement** of the lens readout with the model's own next-token prediction: ≈0 through
  L6, 2/4/11% at L7/8/9, then 24% at L10 and 100% at L11 (the lens target).
* **autocorrelation** of lens top-1 across adjacent positions, minus the shuffled null: ≈0 through
  L5, then +.008 (L6), +.013, +.020, +.024 (L9), +.021 (L10), +.008 (L11).
* **excess kurtosis** of the readout logits is flat (0.4–0.6) at every layer.  The paper's kurtosis
  rise through the workspace band does not occur.
* **effective dimensionality** rises smoothly (.13 → .50 over L0–10) with no onset.

### Dictionary similarity (`results/band/cka.json`)

Plain linear CKA between layer dictionaries is ≥0.94 for every pair in L0–10 (L11 ≈ 0.45).  That is
one direction: after mean-centering, the top principal component still carries 26–37% of each
dictionary's variance at L0–10 (2% at L11).  Removing the top 5 PCs, or taking the mean squared
canonical correlation between top-50 subspaces, gives a smooth diagonal drift and no block:

* adjacent-layer similarity, top-5 PCs removed (L0↔1 … L10↔11): 0.95 0.95 0.96 0.98 0.96 0.96 0.96 0.93 0.93 0.95 0.92
* adjacent-layer similarity, mean CCA r=50:                     0.89 0.88 0.92 0.92 0.90 0.89 0.88 0.85 0.83 0.78 0.76
* two-apart, mean CCA r=50 (L0↔2 … L9↔11):                      0.82 0.82 0.87 0.84 0.85 0.83 0.77 0.75 0.69 0.63

Similarity decays with layer distance at every depth and, if anything, decays faster late (L9↔10
0.78) than early (L2↔3 0.92).  The paper's shared-subspace block across the workspace band is absent.

### Choice

Band = layers **7–9**: autocorrelation above the null from L6, top-1 agreement with the output still
<15% at L9, L10 clearly motor.  L6 and L10 are both arguable; nothing in the structural statistics
picks out 7–9 sharply, and the post should say so rather than present the band as given.

### MLP gain (`python -m jl.band mlp_gain`, `results/band/mlp_gain.json`)

Paper Fig 32: output norm of block L+1's MLP on a unit direction, normalized by the median over
random unit directions.  2000 J-lens vectors per layer.

| L | centered J-lens | raw J-lens | neuron output dirs |
|---|---|---|---|
| 0 | 1.05 | 6.01 | 1.04 |
| 3 | 0.68 | 1.43 | 0.96 |
| 6 | 1.32 | 3.04 | 1.07 |
| 7 | 1.23 | 2.78 | 1.03 |
| 8 | 1.32 | 2.16 | 0.97 |
| 9 | 1.36 | 1.64 | 0.91 |
| 10 | 1.54 | 0.46 | 0.93 |

Centered J-lens directions are amplified 1.2–1.5× in L6–10 (neuron directions ≈1.0×); the paper
reports ≈10× through Claude's workspace band.  The raw-vector column is the cone direction and is
not meaningful.
