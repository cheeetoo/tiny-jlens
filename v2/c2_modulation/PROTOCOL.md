# Criterion 2 — directed modulation

> Paper §1: *"Directed modulation. When instructed to hold a concept in mind, or perform mental
> calculations, the model is capable of activating and computing with workspace vectors,
> independent of its outputs. In addition, information that is not typically represented in the
> workspace can be pulled in when the task requires it."*

Paper §3.2 (+ appendices "Modulation prompt sensitivity", "Additional modulation examples",
"Directed modulation affects the J-space more than other representations") has four threads. We
run all four. Everything here is produced by `run.py` (results in `results/`, figure by
`figure.py`).

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

## Setup

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

**Lens / centering / band** — see `../README.md`. Unchanged from criterion 1.

## 2a — instructed hold-in-mind (concept)

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

## 2b — instructed mental computation (math)

> §3.2 / Fig 10: *"mentally evaluating a mathematical expression … a trial is positive if the
> target reaches J-lens top-1 … this tends to increase with model size."*

Capability-gated. On the reference's 24 problems, GPT-2 gets **0/24** right greedily when asked
directly (`4 * 2 =` → `:`); the answer sits near rank 8 — present but never produced. During
silent copying under a "work it out in your head" instruction, the answer never enters the band
(hit@25 0.00, median rank ~4000). The model cannot compute these, so there is nothing for the
J-space to hold. This is a capability floor, consistent with the paper's own finding that the
effect grows with model size (and GPT-2 is far below Haiku, the paper's weakest model).

## 2c — implicit task-demand ("pulled in when the task requires it")

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

## 2d — is the J-space privileged? (imagine)

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

## Deviations from the paper

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
| vocabulary mean subtracted from each lens vector; band 7–9 | inherited from criterion 1 / `../README.md`; changes no readout |
