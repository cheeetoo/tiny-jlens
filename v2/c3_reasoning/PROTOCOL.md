# Criterion 3 — internal reasoning

> Paper §1: *"Internal reasoning. Workspace vectors can be used to represent the value of
> intermediate computations, when the model chains inferential steps or composes plans, and
> intervening on them is sufficient to redirect the conclusion."*

Paper §3.3 has, in effect, six pieces. We run five; the sixth (the two-step arithmetic of
Fig 17, `(4+17)*2+7`) has no base-model form — gpt2-small cannot do the arithmetic (0/… ; the
intermediates 21/42/49 never form). Everything here is produced by `run.py` (results in
`results/`, figure by `figure.py`).

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

## Setup

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
(so the answer can't be echoed). Full list in `results/prompts.json`.

**Data.** The country fact table (`prompts.py`) is world knowledge, filtered at build time to
single-token capitals/answers. The paper's own two-hop set
(`ref/.../probe-swap.json`) is run verbatim as the capability-floor check (`floor` in the
summary).

**Gate.** An item counts if the model's greedy next token is the answer: **48/53** pass. The
swap experiments additionally restrict to target answers starting at output rank ≥ 10 (the
paper's Fig 6 rule, reused), and evaluate every valid same-family swap partner per item.

**Lens / swap.** See `../README.md` for the lens and the vocabulary-mean centering. The swap is
the paper's **coordinate swap** (Methods "patching in lens coordinates", Fig 4C), clamped to the
swapped clean-pass values across the band — the operation §3.3 names (`swaps.py`). This is
*not* the subtract-and-add form used for criterion 1: for this experiment the subtract-and-add
form drives the answer flip on ≈0% of trials here, while the coordinate swap gives 71%.

## E1 — the unspoken intermediate surfaces in the band

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

## E2 — case study (Fig 13)

The clean-vs-swapped panel the paper leads with (spider→ant → 8→6). The paper's riddle phrasings
score 0/6 on gpt2-small, so the case study is a country two-hop: `France → China` swaps the
intermediate lens vector across the band; the model's answer flips `Paris → Beijing`
(Paris log-prob −1.15 → −5.51, Beijing −8.45 → −2.12).

## E3 — the systematic swap (Fig 15 left)

*"a set of 50 two-hop factual prompts … choosing the swap target at random from within the same
category … We measure the fraction of trials in which the swap moves the target-appropriate
answer to the top of the model's output distribution."*

Coordinate swap of the intermediate (France → China) across the band at every prompt position;
success = the target's answer (Beijing) is the model's top-1 output. **71%** over 999 trials
(48 items × their valid same-family partners), median rank of the target answer 88 → 1. Squarely
in the Sonnet/Opus range (70%), above Haiku (54%). We evaluate *all* valid partners per item
(not one random one) for statistical power, as criterion 1 did with its candidates.

## E4 — is the intermediate a smuggled-in answer? (Fig 15 right) — the depth control

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

## E5 — is the J-space privileged? (Fig 16)

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

## Deviations from the paper

The lens vectors, the released lens, the coordinate-swap operation, and the k=25 pursuit split
are the paper's and are **not** deviations. These are:

| | |
|---|---|
| base model: **few-shot two-hop frame** | gpt2-small answers **7–9/90** of the paper's own two-hop prompts; the frame teaches the relation (as criterion 1's list frame taught the report format), and the intermediate is still unspoken (E1) and computed (`null` control) |
| **country families only**; the paper's riddle phrasings (spider→legs) dropped | 0/6 capability on the bare riddles — the capability floor |
| **coordinate swap** (Fig 4C), not the subtract-and-add of criterion 1 | §3.3 names the coordinate swap; on gpt2-small subtract-and-add flips ≈0% here while the coordinate swap gives 71% (the reverse of criterion 1) |
| swap graded over **all** valid same-family partners per item (n = trials) | statistical power, as criterion 1 did with its 10 candidates; the paper uses one random partner |
| **probe cues authored** for the base model; our non-negative pursuit stands in for "gradient pursuit" | the paper's probe-construction prompts are not released; cues imply the country and ask a different attribute, name never next token |
| J-space share of probe variance 27–57% vs paper 10–15% | GPT-2's 768-dim residual + k=25 lens directions capture more; same scale effect as criterion 1 (23–33% vs 6–7%) |
| depth control (E4) uses **single-layer** coordinate swaps, onset = half-max | the paper's "different layer ranges"; single-layer injection can act before a concept forms, so the intermediate-vs-answer contrast (not the absolute onset) is the claim |
| vocabulary mean subtracted from each lens vector | inherited convention (changes no readout); see `../README.md` |
| band 7–9 | inherited; see `../README.md` |
| two-step arithmetic (Fig 17) not attempted | gpt2-small cannot compute `(4+17)*2+7`; the intermediates never form |
