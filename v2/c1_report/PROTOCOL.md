# Criterion 1 — verbal report

> Paper §1: *"Verbal report. When the model is asked what it is thinking about, it names concepts
> represented in the workspace. Swapping one active workspace vector for another changes its
> answer to match."*

Paper §3.1 has four experiments.  We run three; the fourth (injected-thought introspection)
requires an assistant that understands "report what you detect" and has no direct base-model
form (a base-model attempt is in `../explore/introspection/`).  Everything here is produced by
`run.py` (results in `results/`, figure by `figure.py`).

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

## Setup

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
chosen for how well the model then answers.  For `sport` (all 14 in `results/prompts.json`; a
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

**Lens** — see `../README.md`.  The J-lens vectors are the rows of W_U J_L (the paper's
definition); we additionally subtract the vocabulary mean from each, which changes no readout
and only affects the geometry the swap acts on.  Band = layers 7–9.

## 1a — correlation

*"We apply the J-lens at the token position immediately before the name is produced … the
ordering of the reported words is … highly correlated with the ordering among the lens tokens,
and … this correlation increases towards the end of the workspace."*  Per category, Spearman ρ
between the lens logits and the output logits of the 10 candidates at the colon.  Mean over
categories: ≈ 0 through L0–4, 0.22 (L5), 0.36 (L6), **0.44 / 0.47 / 0.57** (L7–9), 0.63 (L10).

## 1b — the swap

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

## 1d/1e — is the J-space privileged?

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

## Deviations from the paper

These are places where the base model or the tokenizer forced a choice; the lens vectors, the
released lens, and the swap operation are the paper's and are **not** deviations.

| | |
|---|---|
| base model: 8-shot list frame, `Name a {cat}:` | GPT-2 does not follow the instruction on its own; the frame was chosen for how well the model then answers |
| capability gate; the 5 categories it cannot answer are reported separately | 9/14 categories answer |
| single-token candidates only; first 10 of those | the lens has one direction per token; instruments keep 8, rivers 7 |
| vocabulary mean subtracted from each lens vector | GPT-2's unembedding rows share a large common component (raw lens vectors have mean pairwise cosine 0.99); subtracting the mean removes it and changes no readout |
| band 7–9 | checked separately (`../explore/band/`) |
| concept vectors read at the `.` of `Tell me about {c}.` (no Assistant turn); our non-negative pursuit stands in for the paper's unspecified "gradient pursuit" | base model; method not specified in the paper |
| introspection experiment run only as a base-model attempt | the paper's protocol needs an instruction-following assistant |
