# A Workspace in Miniature: testing the "privileged set" in SmolLM2-135M and GPT-2-small

*tiny-jlens project report — 2026-08-17 (day 2). Confirmatory results complete for
SmolLM2-135M-Instruct; GPT-2-small stretch results included; SmolLM2-360M scale point in
progress. Pre-registration: [BRIEF.md](BRIEF.md); frozen protocol:
[CONFIRMATORY.md](CONFIRMATORY.md); full audit trail: [LAB-LOG.md](LAB-LOG.md).*

## TL;DR

Gurnee et al. (2026) showed that Claude models maintain a **privileged set** of
verbalizable representations — reportable, instructable, used in internal reasoning,
flexibly re-usable, and selectively engaged — and framed this as evidence bearing on
conscious access. Eleos AI's commentary isolated the **privileged set** (five evidence
families) as the paper's clearly-established core and the morally relevant part.

We asked: **can the same evidence pattern be instantiated in a 135M-parameter model —
and in GPT-2?** Using Anthropic's own reference implementation and protocol (validated
to 0.99+ agreement against their released lenses), with a pre-registered design, frozen
verdict bars, and held-out confirmatory items:

| Criterion | Headline confirmatory result (SmolLM2-135M) | Paper reference | Verdict |
|---|---|---|---|
| **C1 Report** | report↔lens Spearman 0.55 [0.42, 0.66], rising through the band; swap-to-report **84% top-5** [76, 90]; J-beats-non-J paired rank 29/31 (p<0.001); clamp kills non-J route 9→0 (p=0.004); **but** matched-norm rate asymmetry 36% vs 25% misses the 3× bar | 88% top-5; 59% vs 5% | **Hints** |
| **C2 Modulation** | paper's hit metric **null** (focus=baseline=2.5%); "imagine" dissociation fails; demand-driven maintenance decisive (23/24, p<10⁻⁵). **Revised operationalization (see §4b): Shown** — instructed concepts held at late layers, think median rank 45 vs baseline 564 (36/36, p=3×10⁻¹¹), think vs don't-think 34/36, blurt-controlled, confirmed on held-out items against pre-declared bars | substantial hit rates | **Not shown** (orig.) / **Shown** (revised) |
| **C3 Reasoning** | unspoken intermediate in lens at answer position **72%** [57, 83]; intermediate swap redirects answer **63% top-1** [48, 76] (Haiku 54%, Sonnet 70%); cross-function consistency 15/30 both-flip; probe J-component 43% vs non-J 14% (**3.0×**), clamp → 0; band-robust | 54–70% swap | **Shown** |
| **C4 Flexibility** | same-vector swap across functions **73% top-1** [56, 86] — but only the `countries` category survives the capability filter (grid breadth unmet) | 40–53% overall; countries "almost perfect" | **Hints** (capped) |
| **C5 Selectivity** | same-latent design: flexible tasks follow the swap **78%** [63, 88] while automatic continuation changes 3/19 (**15.8%**, bar ≤15% — miss by 0.8pp; flexible-vs-automatic contrast p<0.0001); whole-J-space ablation battery **fails** its pre-declared conjunction | ~100% vs ~0%; battery dissociation clean | **Hints** |

**Pre-registered overall claim (all five ≥ Hints, ≥3 Shown): NOT met** — 1 Shown, 3
Hints, 1 Not shown. What *is* met is specific and striking: the causal core of the
pattern — verbalizable content that the model reports, reasons over, and re-uses, with
influence that routes through J-lens coordinates — instantiates at 135M with effect
sizes at or above the paper's Haiku 4.5 numbers. What fails to transfer is equally
specific: **instruction-driven** covert modulation, the whole-space ablation
dissociation, component-variance privilege at paper thresholds, and the structural
signatures of a workspace (capacity, sparse-code geometry).

**Update (day 2, later)**: prompted by independent prior work (the user's LW study
showing internal state control with no size trend down to 270M, with tiny-model
effects in late layers), we built a revised, rank-sensitive C2 operationalization,
froze new bars, and confirmed on held-out words and sentences: **instruction-driven
modulation is present at 135M after all** — my original protocol was insensitive
(band excluded the late layers where the effect lives; a top-1 hit metric cannot see
a 1000→100 rank shift). The original frozen verdict stands *for the original
operationalization*; both are reported. The scale story changes accordingly: control
exists at every rung including **base GPT-2** (think median 17 vs baseline 749,
36/36; 23/23 under the strictest blurt filter; think vs don't-think 21/23); what
scale buys is the *dominance* of held content (rank ~20–50 at ≤135M → rank 1 at
360M+), which is exactly why top-k metrics manufacture an apparent threshold.

**GPT-2-small** (2019; using the lens the paper's authors themselves released): on its
capability-filtered two-hop items, unspoken intermediates appear in the lens at **79%**
and intermediate swaps redirect its answers at **83% top-1**; report-swap 70% top-5;
report↔lens correlation rises to 0.59. The probe-split and cross-function controls are
not feasible on GPT-2 (frame too degenerate; capability too narrow).

## 1. Why this experiment

The workspace paper moved careful people. Eleos called it "the most significant
evidence of consciousness in LLMs so far uncovered by mechanistic interpretability
research" and concluded it "should prompt a meaningful update to the research
community's thinking about LLM moral status." The implicit inference:

> (P1) Claude exhibits the privileged-set evidence pattern.
> (P2) That pattern is defeasible evidence of morally relevant cognitive access.
> (C) Take Claude's moral status more seriously.

We stress-tested (P2) by attempting to instantiate the same evidence pattern in models
almost everyone antecedently writes off. This is counterexample methodology: whatever
happens, someone updates. The result forces a *specific* choice now:

1. **Bite the bullet**: extend nonzero consideration to SmolLM2-135M (and partially
   GPT-2), whose lens-readable intermediates causally govern answers at
   Claude-adjacent rates.
2. **Discount the checklist**: conclude the privileged-set pattern as operationalized
   is insufficient for the moral update — and now say *which part* was carrying the
   moral weight. Our results sharpen this option considerably: if what matters is
   top-down control, the 135M model fails it; if what matters is reportable,
   causally-load-bearing internal content, the 135M model has it.
3. **Locate the difference in degree**: the paper's pattern comes apart at small scale
   into a causal-functional core (transfers) and a structural/agentic shell (does
   not). Anyone updating on the paper should say which component their update tracked.

Had the pattern failed to appear at all, that would have strengthened the original
paper's evidential weight ("not free with steering vectors"). Instead we got something
more informative: a **decomposition** of the pattern.

## 2. What "this" is

Eleos's three-level distinction: *privileged set* < *privileged stream* < *GWT
workspace*. We targeted the weakest, the privileged set — "certain representations
display the characteristics of cognitive accessibility" — operationalized by the five
evidence families of the paper (§3.1–3.5), **including the controls that make each
family mean something** (matched-norm non-J comparisons, clamps, no-instruction
baselines, automatic-task invariance). We did not target the stream or workspace
claims, and none of our results speak to phenomenal consciousness.

## 3. Methods and validity

- **Lens**: Anthropic's reference implementation (`anthropics/jacobian-lens`), their
  fitting recipe (wikitext-103, 128-token sequences, n=1000; target = final layer).
  **Pipeline validation**: our gpt2-small lens agrees with Neuronpedia's released lens
  at r = 0.993–0.9996 per layer — *above* the same-recipe sampling-noise ceiling
  (A-vs-B halves: 0.964–0.998). Readout code reproduces the reference `apply()`
  exactly (0.0 max diff). Our positive control reproduced the paper's flagship
  readout on Qwen3.5-0.8B with the authors' prefit lens (Poland surfacing
  mid-network before Warsaw).
- **Workspace band**: located by pre-registered, outcome-independent metrics
  (content-onset & motor-cutoff on held-out wikitext) → **L19–L26 of 30** for
  SmolLM2-135M (last quarter; the paper's Claude band is 38–92%). The original
  kurtosis/autocorr thresholds from the paper's metric behavior do not transfer and
  were amended pre-confirmation (logged). All headline interventions are robust to
  band starts {17, 19, 21}.
- **Capability filter**: every task battery is restricted to items the model performs
  correctly *before* any lens measurement (the paper's own baseline rule), applied
  symmetrically to tasks we predicted would pass and fail.
- **Explore/confirm split**: all analysis choices frozen
  ([CONFIRMATORY.md](CONFIRMATORY.md)) before confirmatory runs on the final n=1000
  lens with fresh seeds and held-out items (new categories, passages, carriers,
  two-hop items). Exploration→confirmation results were stable (e.g. C3 swap 56%→63%;
  C1b 87%→84%).
- **J-lens is doing the work**: on capability-filtered two-hop items, J-lens pass@5 =
  30/44 vs logit lens **1/44**. At tiny scale the Jacobian transport is nearly the
  entire signal — these results are about the lens-identified space, not raw
  unembedding readout.

Key adaptations (all logged; full list in LAB-LOG): task content simplified within
each family's logical form; C1b uses the paper's §3.1 projection-swap form (its own
description of that experiment); C3's depth-timing control replaced by a stronger
cross-function-consistency control (depth separation of 17% × band is sub-resolution
in an 8-layer band); gradient-pursuit k=50 column added with crowding justification;
C5a's anomaly task dropped (the model cannot discriminate spliced passages at all —
capability, measured).

## 4. Results in brief

**C1 — Report.** The model's report of "think of a {category}" is predicted by
late-band lens rankings (Spearman rising 0.21→0.58 across the band — the paper's
signature shape), and *caused* by them: swapping the chosen item's lens projection for
another category member redirects the spoken report in 84% of held-out trials
(top-5; 60% top-1) across 18 categories. Concept-vector splits show J-components beat
matched-norm non-J components in rank almost always (29/31), and clamping J-coordinates
to clean values eliminates the non-J route entirely (9/36 → 0/36) — every causal path
to the report runs through the verbalizable coordinates. What fails: the paper's
*rate* asymmetry (59% vs 5% for Claude) compresses to 36% vs 25% here. **Injected
thoughts** (secondary): steering a concept's lens vector on the user turn makes the
model "report the injected thought" dose-responsively (7/24 top-1 at strength 16, 0/24
at strength 0) without indiscriminate blurting (3/24) — the paper's introspection
protocol, working in a 135M model.

**C2 — Modulation (original operationalization).** The one clean failure under the
paper's metric — but see §4b below; the failure turned out to be the metric's. "Think about X while copying" produces no
top-1 lens hits above baseline (the paper's metric), and "imagine this is
French/past-tense/caps" barely moves anything (z<1.1 where the paper reports several
SDs). The continuous-metric structure that does exist is exactly the paper's shape in
miniature — mention primes (+0.5 nats), *focus adds nothing beyond mention* (the
paper's own finding), ignore actively cancels mention back to baseline — but
magnitudes are far below bar. Meanwhile the *demand-driven* analog is decisive: when
the held item will be needed after the copying task, it is maintained in the lens
throughout copying (median rank 6.5) vs. no-demand (median 64), 23/24 paired trials,
p<10⁻⁵. At 135M, workspace contents follow **task demands, not verbal instructions
about mental state**.

**C2b — Modulation, revised (Shown).** Mirroring the protocol of the user's prior
study (instruction after the sentence; paired think vs don't-think, which cancels
mention-priming; measurement at all layers; explicit blurt controls), the instructed
word's J-lens rank at the transcribed-sentence tokens drops from a baseline median
of 564 to 45 under "think about X" (36/36 held-out pairs, p=3×10⁻¹¹), sits at 94
under "don't think about X" (content sensitivity 34/36; suppression real but
incomplete — the white-bear pattern, as in the paper), and the model is *not*
preparing to say the word (in its actual output top-10 in only 4/36 trials, with
the ordering intact among non-blurt trials). The effect lives at L23–28 — partly
in layers the band analysis had classified as "motor," which at this scale
evidently carry held-but-unspoken content too. All bars for this revision were
frozen before its confirmatory run; the original verdict is reported unchanged
alongside it.

**C3 — Reasoning (Shown).** On 43 capability-filtered two-hop items ("The capital of
the country where Polish is the primary language is…"), the unspoken intermediate
(Poland) is in the lens top-10 at the answer position in 72%, and swapping the
intermediate's lens coordinates flips the answer to the counterfactual (Warsaw→the
swap-country's capital) in 63% — above Haiku 4.5's published 54%. Anti-smuggling: the
*same* country swap redirects *two different questions* (capital & language) to their
respective correct answers in 15/30 pairs (29/30 flip at least one) — impossible for a
vector that merely smuggles one answer. Probe-splits: J-component carries 3.0× the
matched-norm non-J flip rate, and the non-J route dies under J-coordinate clamping
(→0). Robust across band starts.

**C4 — Flexibility.** Within the one category that fully survives the capability
filter (countries), one identical swap serves capital/language/continent functions at
73% top-1 — the paper's own countries cell was near-perfect. But months/animals/numbers
cells cannot reach the pre-registered grid spec at 135M (a capability wall, not a lens
failure), so C4 is capped at Hints by our own rule.

**C5 — Selectivity.** The same-latent design works: with the passage's language
sitting in the lens in *all* conditions, swapping it flips report and
language-to-country answers (78%) while the automatic continuation task stays in the
original language in 16/19 gradable trials (2 English-drift degradations, 1 true
redirect leak). Flexible-vs-automatic contrast p<0.0001; the absolute invariance bar
(≤15%) is missed by 0.8pp. The **whole-J-space ablation battery fails** its
pre-declared conjunction: only language→country collapses (63–75%) while
sentiment/copy/next-token prediction hold; the passage tasks don't drop at all (long
prompts re-derive the ablated content each layer from un-ablated token streams), and
the norm-matched noise control dents language→country almost as much. At this scale,
top-k lens ablation cannot cleanly excise "the workspace" — see §5.

## 4c. The late-layer audit (prompted by the user's "things happen later than
you expect" — which was correct)

After the C2 reversal we audited every place the L27–28 exclusion could have
mattered. Findings:

- **The task-content geography is a cliff, not a band.** Reasoning intermediates are
  present at the answer position in 0/44 items at L14–L21, then 32/44 at L23,
  sustained through L28. At 135M there is effectively one late region (L23–28) doing
  workspace-and-motor duty together; the pre-registered band rule (L19 onset) had
  detected the *generic* onset of lens signal on web text, not where task content
  lives. Including L27–28 nudges C3 readout from 72% to 77% (the confirmatory 72%
  stands as scored). Swap experiments were unaffected by design — motor-layer swaps
  were excluded to keep interventions non-trivial, which is conservative.
- **C2c ("imagine") partially wakes up at the live layers**: lens effects roughly
  double or triple (language z 0.6→2.0) with the J-orthogonal probe mostly flat —
  the same family as C2-revised, much weaker, and still far from the paper's
  several-SD dissociation. Reported as weak-present rather than absent.
- **C5b's failure is robust to band aim — and to dose.** A dose–response sweep
  (k = 1, 2, 3, 5, 10 at the live band) shows the ablation has no gentle setting:
  k=1 already removes 8.1% of residual norm (vs 9.7% at k=10) because the top lens
  direction at nearly every position is the shared cone axis. At low k the "shallow"
  sentiment task is already broken (the shared axis is load-bearing for it) while
  flexible tasks are intact; at high k the flexible tasks collapse but shallow ones
  are still broken. No dose satisfies the shallow-intact/flexible-collapsed
  conjunction. Notably, the paper itself reports the beginning of this trend: "on
  Haiku 4.5, J-space ablation degrades coherence before yielding any qualitative
  change in responses" — our result is plausibly that same phenomenon one to two
  orders of magnitude further down, i.e., closer to replicating the paper's
  small-model observation than failing its large-model one. Re-running the battery on
  L23–28: sentiment (a "shallow" task) collapses as hard as the flexible tasks,
  passage-language report is indestructible under every condition (long prompts
  rebuild the content each layer), and matched noise damages two-hop as much as the
  true ablation. Whole-space ablation does not produce clean dissociations at 135M
  in any band we tried; the selectivity evidence properly rests on the targeted
  same-latent swap design (C5a), which is also the paper's primary §3.5 experiment.
- **The cone persists at L27–28** (top-PC share 0.69–0.70), so the matched-norm
  variance-privilege failure is not rescued by late layers; restricting the probe
  split to the live region raises both components' absolute effects (J 57%, non-J
  36%) while *shrinking* the ratio to 1.6× — near the output, matched-norm
  perturbations of any kind gain power, and the J/non-J comparison loses
  selectivity. The confirmatory 3.0× at the original band stands as scored.

## 5. What does not transfer — the interesting part

- **The J-frame is a degenerate cone.** At d_model=576, the 49k lens vectors have mean
  pairwise |cos| ≈ 0.78; a single direction carries ~78% of their variance; after
  removing it they are near-orthogonal (|cos| 0.07 ≈ 2× random). The paper's picture
  of a sparse frame of distinguishable verbalizable directions degrades badly.
- **Occupancy ≈ 1** (paper: ~25). Under the paper's own criterion (sparse
  decomposition vs matched random control), the J-space at 135M holds roughly *one*
  meaningful atom of a generic activation — the shared verbalization direction — with
  concept-specific content living in a thin, heavily-shared tail. Greedy pursuit
  exhausts after ~2 atoms.
- **Consequences**: variance-share "privilege" tests (J vs non-J at matched norm)
  compress; whole-space ablation lacks a clean target; capacity claims are
  unevaluable. Yet the *causal* privilege tests (clamping J-coordinates blocks all
  routes to behavior) pass decisively everywhere we ran them.
- **Instruction vs demand**: the missing criterion (C2) and the failing C2c
  dissociation both concern *verbal instruction* steering internal state. Everything
  demand-driven works. A reading: what 135M models lack is not workspace-like content
  but top-down *agentic control over* that content.
- **Everything is late and thin**: workspace-like content occupies roughly the last
  quarter of depth (vs. the paper's middle 55%), and the "motor" regime is compressed
  into the final 2–3 layers.

## 6. GPT-2-small

With the authors' own released lens (their pipeline, their fit), frozen band rule →
L7:10 of 12: capability filter passes 25/136 two-hop items (mostly
language→capital); **readout 79%** at the answer position; **swap 83% top-1**;
few-shot report correlation rising to 0.59; report-swap 70% top-5 (coordinate form,
α=2; the projection form fails on GPT-2 — per-model method sensitivity, reported).
Probe-split decomposition does not engage at all (gradient pursuit captures 0.16% of
probe variance — the frame is even more degenerate), and no second function family is
within GPT-2's capability, so the privilege and anti-smuggling controls could not be
run. GPT-2 therefore shows the *readout + causal-swap core* of C1/C3 only — with the
paper authors' own artifact, in the model most often used as the definition of "a
model nobody worries about."

## 6b. The scale ladder

Running the same code on models with authors-released lenses (qwen3.5-0.8b, gpt2,
pythia-70m; bands by our frozen rule) turns the decomposition into a dose–response
curve:

| | pythia-70m | gpt2-124M | SmolLM2-135M | SmolLM2-360M | qwen3.5-0.8B | Claude (paper) |
|---|---|---|---|---|---|---|
| Two-hop capability (items passing filter) | 0/136 | 25/136 | 44/136 | 42/136 | 51/136 | — |
| C3 readout / swap (fixed 135M-frontier pool) | — | 79% / 83% | 72% / 63% | 62% / 52% | 45% / 45%* | — / 54–70% |
| C1 report corr (late band) / swap top-1 | — | 0.59 / 41%† | 0.55 / 60% | 0.75 / **100%** | 0.83 / 98% | high / — |
| C1c privilege ratio: J vs non-J (matched norm) | — | n/a (frame degenerate) | 36% vs 25% (1.4×) | 86% vs 47% (1.8×) | **88% vs 12% (7.4×)** | 59% vs 5% (11×) |
| C2 instruction modulation, original metric (focus ÷ baseline hits) | — | — | 1.0× (null) | 3.0× | 8.5× | substantial |
| C2 revised: think vs baseline median lens rank | — | **17 vs 749** (base model!) | 45 vs 564 | 1 vs 206 | 1 vs 854 | — |
| C2c "imagine" lens effect (z) | — | — | 0.6–0.7 | 1.1–1.45, probe flat | 0.2–0.8 (cross-family) | several SD |
| Workspace band (fraction of depth) | — | 58–83% | 63–87% | 56–91% | 55–95% | 38–92% |
| Kurtosis workspace-onset signature | — | flat | flat | flat | **present** | present |

† gpt2 top-5 70% (coordinate form).

*qwen ran the frozen 135M protocol unmodified except band — no per-model tuning; its
lower C3 rates likely reflect band/anchor mismatch (any-position readout is 92%) and
are reported as-is.

The ordering: at 70M the capability filter empties and the question dissolves; from
GPT-2/135M upward, both the **causal core** (readout, report- and reasoning-swaps,
demand-selectivity, clamp-mediation) *and* instruction-driven control (revised
metric) are present. What actually scales: the dominance of held/instructed content
in top-of-lens terms (which is why the original top-k metric manufactured an
apparent 1.0× → 3.0× → 8.5× "threshold"), and the variance-privilege ratio
(1.4× → 1.8× → 7.4×). A methodological caution cuts both ways: the span-max blurt
flag saturates on larger models (the held word enters their output top-10 somewhere
in the span in ~all trials), so the held-vs-spoken dissociation is only cleanly
measured at the small end; a position-level analysis would be needed above ~360M.

One theoretically loaded wrinkle: on the *fixed* item pool (built at the 135M
capability frontier), C3 readout and swap rates *fall* with scale (72/63 → 62/52 →
45/45). The paper's own selectivity logic predicts exactly this direction: tasks that
demand workspace mediation from a small model become automatic for a larger one, and
automatic computation bypasses the J-space. We flag this as an interpretation, not an
established mechanism — but it means "how much workspace a task engages" is a joint
property of task and model, which any scale-based criterion must reckon with.

## 7. What this means

Against the pre-registered bars, the strict conclusion: **the full privileged-set
pattern, as the paper operationalized it, does not instantiate in SmolLM2-135M — but
its causal-functional core does, at effect sizes comparable to the smallest production
Claude model the paper reports.** The decomposition:

**Transfers to 135M (and partly to GPT-2):** verbalizable content that (i) predicts
and causally determines verbal report, (ii) carries unspoken intermediates that
causally mediate multi-step answers via the lens coordinates specifically (clamp
tests), (iii) serves multiple downstream functions, (iv) is engaged by flexible tasks
and bypassed by automatic ones, (v) is loaded by task demands, and (vi) supports
dose-responsive "injected thought" reports with selectivity.

**Also transfers (revised metric):** instruction-driven covert modulation — down to
base GPT-2 — as a rank shift rather than top-of-lens dominance.

**Does not transfer:** the imagine-instruction dissociation; whole-space ablation
dissociation; matched-norm variance privilege at paper thresholds; sparse-code
structure (occupancy, frame geometry); breadth across task categories; top-of-lens
dominance of held content.

For the three-way update in §1: option 2 and 3 readers now have exactly what they
need — the checklist was not a unit. If your update from the workspace paper tracked
*reportable, causally load-bearing internal content*, that exists in models you give
no weight to, and consistency requires either a tiny-model update or a revised
criterion. If your update tracked *top-down control, capacity, and structural
workspace organization*, the 135M result leaves you where you were — and identifies
which experiments carry that weight (the paper's §3.2, §3.5 ablation, §4.2), which
are precisely the ones Eleos's "privileged stream" worries pointed at. What no one
can say anymore is that the five-family evidence pattern is an indivisible package
that arrives only with frontier scale.

Two cautions cut against over-reading the positive half. First, the effects here are
narrower in breadth than Claude's even where rates match: fewer categories, shorter
prompts, single-token concepts. Second, the paper's authors were explicit that the
J-lens is an approximate window; at 135M the approximation is visibly coarser (cone
geometry), so "the model's J-space" is a blurrier object than "Claude's J-space" —
in both directions (some failures may be lens artifacts; some successes ride on a
few shared directions).

## 8. Integrity notes

- Pre-registration written before any experiment ([BRIEF.md](BRIEF.md)); per-criterion
  bars and concession criteria frozen day 1; amendments (band rule; C3 timing→crossfn;
  k=50 column; C5a anomaly drop) all made pre-confirmation and logged with reasons.
- Every confirmatory number reported regardless of outcome; the pre-registered
  headline claim is reported as **not met**.
- The search is reported (R4): all exploration passes, failures, bugs (digit
  tokenization, swap-oscillation semantics), and protocol variants are in
  [LAB-LOG.md](LAB-LOG.md). Two method bugs were found and fixed *in our own code*
  against the paper's semantics; both fixes moved results toward the paper's protocol,
  not away from it.
- Verdict judgment call, flagged: C1's "Hints" (rather than Not-shown) reads the
  clamp-mediation test as a privilege contrast; the matched-norm rate-asymmetry
  clause fails and is reported as failing.

## 9. Next

- Template-lens extension for multi-token concepts at tiny scale; a proper capacity
  operationalization for degenerate frames; per-model protocol tuning for the ladder
  points (all ladder numbers above run the frozen 135M protocol unmodified).
- Write-up for external audience (the LW/AF version), if desired.
