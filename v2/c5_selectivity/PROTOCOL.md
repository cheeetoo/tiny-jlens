# Criterion 5 — selectivity

> Paper §1: *"Selectivity. The workspace comprises a small subset of the total representational
> content of the model's activations. It is required for only a fraction of the model's
> behavior, and in particular is not involved in pervasive, routine processing like text parsing
> or grammatical fluency."*

The definition has **two clauses**: (i) the J-space is a *small subset* of representational
content, and (ii) it is *required for flexible but not automatic* processing.  §3.5 is the home
of clause (ii) and is the bulk of this criterion; clause (i) is established structurally in §4.2,
included here compactly (as S1) so the definition is covered end to end.  Everything is produced
by `run.py` (results in `results/`, figure by `figure.py`).

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

## Setup

**Ablation (S2a).**  §3.5.2: *"at each token position, across a band of layers, we identify the
k=10 most strongly activated J-lens vectors and zero out the residual stream's projection onto
each … we do not ablate any tokens that appear in the top-10 tokens of a clean forward pass."*
We do exactly this (`ablation.py`): at each (position, band layer) we take the 10 highest lens
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
same language-label swap (criterion 3's coordinate swap, `swaps.py`) is applied over the passage
tokens under both.  **Report** graded by whether the reported language flips to the swapped-in
one; **continuation** graded by whether the model still prefers to continue in the passage's own
language, scored against short per-language continuation phrases (a next-token language
classifier is unreliable for the closely-related Romance languages).

**Capacity (S1).**  wikitext activations at band layers, mean-subtracted; the fraction of
variance a non-negative pursuit captures with the top-K J-lens directions vs a same-size random
dictionary; occupancy = the K at which the J marginal gain drops below random's (paper §4.2).

**Lens.**  See `../README.md`: the released gpt2-small lens, J-lens vectors = rows of W_U J_L,
vocabulary-mean subtracted (changes no readout), band 7–9.

## S2a — J-space ablation is selective (§3.5.2, Fig 22/24)

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
(a) the **order** of collapse — flexible first, recall next, automatic last (Figure B) — and (b)
the J-vs-random gap at **light** ablation, where nothing is floored (Figure A).  Both are the
paper's Fig 22/24 result; the absolute robustness is lower than Sonnet's, as expected at 124M.

## S2b — same latent, two tasks (§3.5.1, Fig 20)

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

## S1 — the J-space is a small subset (§4.2, Fig 30)

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

## floor — line-length counting (§3.5.1, Fig 21)

The linecount experiment is a **base-model capability floor**.  Over the released passages, a
count token reaches the band J-space top-25 on 0–1 of 11 passages under every condition
(linewrap / direct / first-letter), and the model's greedy answers are `been` / `a` / `the`.
gpt2-small cannot represent character counts, so the count neither enters the J-space nor can be
reported — as with the criterion-2 math floor.

## Deviations from the paper

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
| vocabulary mean subtracted; band 7–9 | inherited conventions (change no readout); see `../README.md` |
