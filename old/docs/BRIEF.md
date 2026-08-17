# tiny-jlens: Can a "privileged set" be found in a tiny language model?

*Written 2026-08-16, before any experiment was run. This document is the project's
pre-registration. Sections 1–4 (what we are looking for, why, and the rules) are frozen;
any later edit to them must be logged in the changelog at the bottom with a justification.
Section 5+ (practical plans) may evolve freely.*

---

## 1. What "this" is

Gurnee et al. (2026), "Verbalizable Representations Form a Global Workspace in Language
Models" (the *workspace paper*), introduce the Jacobian lens (J-lens) and argue that the
J-space — the set of directions it identifies — functions as a global workspace in Claude
models. Butlin, Shiller, Plunkett & Long (Eleos, 2026-07-06) distinguish three
successively stronger readings of that claim:

- **Privileged set** — "In some LLMs, certain representations display the characteristics
  of cognitive accessibility."
- **Privileged stream** — those representations form a *unified* stream (shared mechanisms
  of entry and effect, limited capacity, competition).
- **GWT workspace** — the stream has the full architecture of Global Workspace Theory
  (modules, global broadcast, selection).

Eleos judge that the paper "provides strong evidence of privileged representations"
(the **set**), suggestive-but-inconclusive evidence for a **stream**, and deliberately
weaker claims about the full **GWT workspace**. They also argue the set may be what
matters: "The existence of these cognitively accessible representations may be what
matters most, both morally and from the point of view of understanding LLM cognition,"
and that the paper "should prompt a meaningful update to the research community's
thinking about LLM moral status."

**Our target is the privileged set** — the weakest of the three claims, and the one Eleos
identifies as (a) clearly established in Claude and (b) morally relevant on its own.

Eleos operationalize the privileged set by the five bodies of evidence in the paper
(their §2 bullet list, mapping to the paper's §3.1–3.5). We adopt these as our five
criteria, *including* the controls that make each one mean something:

| # | Criterion | Paper's operationalization | The part that makes it "privileged" |
|---|-----------|---------------------------|-------------------------------------|
| **C1** | **Report** | Asked to name a category instance, the model names the item most aligned with late-workspace J-lens readouts (rank correlation); swapping J-lens coordinates changes the report to match; steering a J-lens vector into the context makes the concept reportable when introspection is elicited, *without* indiscriminate verbalization elsewhere. | Matched-magnitude swaps along the **non-J-space component** of the same concept vectors barely work (5% vs 59/88% in the paper), and their residual effect dies when J-coordinates are clamped. |
| **C2** | **Directed modulation** | Instructed to hold a concept in mind (or compute silently) while doing an unrelated surface task, the concept/solution appears in J-lens readouts at surface-task tokens; no-instruction baseline ≈ 0; "ignore X" < "think about X". Implicit task demands (which question will be asked) modulate what enters the J-space over identical stimulus tokens. | The same instructions do **not** move a J-orthogonalized probe for the property — instructions write to the J-space specifically, not to the model's whole representation of the stimulus. |
| **C3** | **Internal reasoning** | On tasks requiring an unspoken intermediate, the intermediate appears in mid-layer J-lens readouts; swapping its J-lens coordinates redirects the final answer correspondingly (paper: 54–70% top-1 across model scales); the intermediate swap bites at earlier layers than an answer swap (the intermediate is not a smuggled answer). | The J-space component of an independently-derived intermediate probe (10–15% of its variance) carries the causal effect (61% ≈ 60% for raw J-lens swap); the non-J remainder gets 28% → 6% when J-coordinates are clamped. |
| **C4** | **Flexible generalization (broadcast)** | One and the same J-lens swap (e.g. France→China), applied identically across prompts that each apply a *different function* to the argument, redirects each function's output correspondingly (paper: 76/192 at α=1, 101/192 at α=2, over 16 functions × 12 swap pairs). Success is predicted by the source concept's workspace loading. | The *same* vector serves many downstream consumers — the signature that distinguishes a broadcast representation from a per-task trick. |
| **C5** | **Selectivity (flexible ≠ automatic)** | With the same latent variable needed for all tasks: explicit report and flexible-inference answers follow a J-lens swap; continuation/anomaly-detection (automatic) tasks are unaffected — even when the variable is sitting in the lens in all conditions. In a second family, the variable enters the J-space only when the task demands report/flexible use. Whole-J-space ablation (top-k projections removed, output tokens spared) leaves routine prediction and shallow tasks near-intact while collapsing multi-step/flexible tasks. | The J-space is required for a *small, specific* fraction of behavior — a privileged sliver atop a large volume of automatic processing, not a generally load-bearing subspace. |

A tiny model "has a privileged set" in our sense iff a J-space fit to it by the paper's
recipe passes C1–C5 under the verdict rules of §4. Structural/stream evidence
(layer-banding, capacity, broadcast heads) is a secondary, nice-to-have layer — we will
measure the cheap parts, but the claim we defend is the set, not the stream.

**What we are *not* claiming, under any outcome:** that tiny models are conscious; that
they are moral patients; that the J-space is "the" workspace rather than an approximation
(Eleos's J-space/W-space gap applies to us too); that a privileged set is a *general*
property of tiny models; or anything about phenomenal consciousness. We are testing
whether a specific, influential evidence pattern can be instantiated at tiny scale.

## 2. Why this is worth doing

The workspace paper moved many people. Eleos — a careful, skeptical shop — called it
"the most significant evidence of consciousness in LLMs so far uncovered by mechanistic
interpretability research" and concluded it warrants "a meaningful update" on moral
status. The implicit inference is:

> (P1) Claude exhibits the privileged-set evidence pattern.
> (P2) That pattern is (defeasible) evidence of morally relevant cognitive access.
> (C) So we should take Claude's moral status more seriously.

Our experiment stress-tests (P2) by attempting to instantiate the *same evidence
pattern* in a model that a large majority of people — with some justification —
antecedently place at or near zero moral consideration: SmolLM2-135M, and if possible
GPT-2. This is counterexample methodology. Whatever we find forces an update somewhere:

- **If the pattern appears in a tiny model**, a reader must choose:
  1. *Bite the bullet*: extend nonzero (tiny, but nonzero) consideration to tiny models.
     Not absurd — small credence × small capacity is a coherent position — but few hold it,
     and most people's stated reasons for the Claude-update commit them to more than they
     realize.
  2. *Discount the checklist*: conclude that the privileged-set pattern, as currently
     operationalized, is not sufficient for the moral update — what matters must be
     something the pattern doesn't capture (scale, richness, valence, self-model,
     capacity, unification into a stream...). This is the update we consider most likely,
     and it is valuable: it forces the criteria to become quantitative and specific
     rather than existential.
  3. *Find a principled disanalogy* between our results and the paper's (weaker effects,
     narrower task coverage, a criterion that only passes in degraded form). Also
     valuable — it converts a vague "Claude has it" into "it comes in degrees, and here
     is the dimension along which degree matters." Our job is to make this option
     available honestly: report effect sizes side-by-side with the paper's.
- **If the pattern does not appear**, despite a strong and unconstrained effort (§3),
  that is genuine evidence that the pattern is *nontrivial* — it is not something you get
  for free from "transformer LM + steering vectors exist." That would strengthen the
  original paper's evidential weight, and we would say so plainly. This outcome is live:
  the paper itself reports every key effect growing with model scale (modulation "tends
  to increase with model size"; intermediate-swap success 54% on Haiku vs 70% on
  Sonnet/Opus; Haiku's coherence collapses under ablations that Sonnet tolerates), and
  Haiku 4.5 is orders of magnitude larger than SmolLM2-135M.

Secondary payoffs either way: the first characterization of J-lens behavior at tiny,
fully-open scale (the paper studies only production Claude models and leaves
"whether smaller models have an equally rich workspace, a proportionally smaller one, a
less reliable one, or none at all" as an open question — we answer its smallest case);
and a fully reproducible open-source replication package of the paper's core protocol.

## 3. Why "trying really hard" is principled here — and its limits

Our claim is existential: *there exists* a tiny model (and adapted task battery) in which
the pattern appears. For an existence claim, searching aggressively over models, tasks,
prompts, layers, and hyperparameters is not p-hacking — it is how one finds a
constructive proof. A single verified example suffices, however it was found.

But four rules keep the search from corrupting the result:

- **R1 — Fixed target.** The definition of the pattern (§1) and the verdict rules (§4)
  are frozen now, before any experiment. Success may not be redefined after seeing data.
- **R2 — Controls are not optional.** Each criterion's meaning lives in its controls
  (non-J components at matched magnitude, clamps, no-instruction baselines, automatic-task
  invariance, random-direction ablation controls). A criterion "passes" only with its
  controls run and behaving as in the paper. We may not drop a control because it is
  inconvenient; a control that fails is a failed criterion, not a footnote.
- **R3 — Explore/confirm split.** Free exploration (model choice, task construction,
  layer bands, α, k, prompt phrasings) is unlimited, but every headline number comes from
  a **confirmatory run**: analysis choices frozen, then executed on fresh items (newly
  sampled prompts/concepts within the same family) with all controls, reported regardless
  of outcome. Item-level cherry-picking inside a confirmatory run is forbidden; the only
  permitted filter is the pre-declared capability filter (below).
- **R4 — Report the search.** The write-up must state what was tried and abandoned
  (models, task families, protocol variants), so a reader can see the size of the
  garden of forking paths even though, for an ∃-claim, it does not invalidate the result.

**The capability filter (principled task adaptation).** A 135M model cannot do many
tasks Claude can. Cognitive access is access-*for use*; one tests report and flexible use
over the system's actual repertoire (you would not test a rat's access with two-hop
trivia). So: for each criterion we may substitute easier task instances, provided
(a) the *logical form* of the experiment is preserved (unspoken intermediate; same swap
applied across multiple functions; automatic vs. flexible contrast over the same latent
variable; etc.); (b) the capability filter — "model performs the base task correctly" —
is applied *before* and *independently of* any lens measurement, and symmetrically to
tasks we want to pass and tasks we want to fail; (c) every substitution is documented
in a paper-vs-ours table. If a criterion can only be exhibited in a form whose logical
structure differs from the paper's (not merely easier content), it is at best **partial**
(§4), stated loudly.

**Fixed search space (so "we conceded" is well-defined).** Models: SmolLM2-135M-Instruct
(primary), SmolLM2-360M-Instruct, Qwen2.5-0.5B-Instruct (escalation), GPT-2 small/medium
(base-model stretch goal, attempted only if a SmolLM2/Qwen model passes). Lens recipe:
the paper's, including its published ablation space (target = final or penultimate layer;
QK frozen or not; all/future/present targets; mean/median aggregation; corpus size up to
~1000×128; pretraining-like corpora only, disjoint from all eval prompts). Interventions:
steering, ablation, pseudoinverse coordinate swaps at α ∈ [0.5, 4], k ∈ [4, 50], any
contiguous layer band. Prompt families: any, subject to the capability filter. Novel
methods beyond this space are allowed only as *additions* (clearly labeled "beyond the
paper"), never as silent replacements for a failed core experiment.

**Concession.** If, after a genuine pass through the search space on all three
instruct models, some criterion still fails its minimum bar (§4), we concede that
criterion at tiny scale and publish the negative result with the same care as a positive
one. We commit to this in advance: a clean, well-controlled "the pattern is absent below
~X parameters" is a publishable and useful finding, not a failure of the project.

## 4. Verdict rules (frozen)

Each criterion gets one of three verdicts, decided by its confirmatory run:

- **Shown**: every key contrast has the paper's direction, is statistically distinguishable
  from its null/control (95% CI excluding it, or p < 0.05 with the paper's own statistic),
  **and** the headline effect is within roughly the paper's range scaled by at most ~2×
  degradation (specific bars below).
- **Hints** (partial): every key contrast has the paper's direction and the *privilege*
  contrasts (J vs non-J, flexible vs automatic) are statistically significant, but
  absolute effects are weaker than the Shown bar, or one secondary sub-experiment of the
  criterion fails while the primary passes.
- **Not shown**: a key contrast is null or reversed in the confirmatory run, or a
  required control misbehaves (e.g. non-J components swap as well as J components; or
  automatic tasks follow the swap too — which would mean we found generic steering, not a
  privileged set).

Minimum bars per criterion (with the paper's reference values):

- **C1 Report.** (a) Lens-vs-output rank correlation across ≥10 candidate items per
  category, ≥8 categories: mean Spearman ρ > 0 with 95% CI excluding 0 [paper: "highly
  correlated", increasing toward late workspace]. (b) Swap-to-report: target enters top-5
  of the output distribution in ≥40% of trials at some α ≤ 2, source excluded from top-10
  pre-swap [paper: 88% top-5 for pure J-lens vectors]. (c) Privilege: J-space component
  success ≥ 3× non-J-space component success at matched perturbation norm [paper: 59% vs
  5%]; clamping J-coordinates reduces the non-J effect by ≥ half [paper: → ~0]. The
  injected-thought/introspection protocol is attempted but graded as *secondary* (a 135M
  model may lack the instruction-following to engage the protocol at all; if it cannot
  engage even after prompt adaptation, C1 is decided by (a)–(c) and the introspection
  piece is reported as not-engageable — explicitly, in the summary table).
- **C2 Modulation.** (a) Target-in-lens rate under "think about X" ≥ 5× the
  no-instruction baseline rate, baseline < 10%, over ≥30 concepts × ≥3 phrasings [paper:
  baseline ≈ 0, substantial fraction under instruction]. (b) ignore-instruction rate
  strictly between baseline and focus rate (directional) [paper: same]. (c) Privilege:
  instruction moves the lens measure by ≥ 2 SE while a J-orthogonalized property probe
  moves < 1 SE, and a real stimulus moves the probe ≥ 3 SE [paper: several SD vs ~1 SD vs
  3–6 SD]. Implicit-task-demand variant: attempted; graded secondary (may exceed tiny-model
  capability; if unattainable, say so).
- **C3 Reasoning.** Over ≥30 capability-filtered items with unspoken intermediates:
  (a) intermediate reaches lens top-10 at some workspace layer in ≥50% of items [paper:
  routine]; (b) intermediate swap flips the answer to the target-consistent one (top-1) in
  ≥30% of items at α ≤ 2 [paper: 54% Haiku / 70% Sonnet]; (c) intermediate-swap effective
  depth earlier than answer-swap depth (median difference > 0) [paper: ~17% of depth];
  (d) privilege: J-component of a mean-difference intermediate probe ≥ 2× the flip rate of
  its non-J remainder at matched norm [paper: 61% vs 28%], clamp halves the remainder's
  effect [paper: 28%→6%].
- **C4 Flexibility.** ≥4 argument categories × ≥3 functions each, ≥8 swap pairs per
  function: same-vector swap succeeds (target-consistent answer top-1) in ≥25% of trials
  overall at α ≤ 2 [paper: 40% at α=1, 53% at α=2], with ≥1 category ≥50%, and success
  correlated with workspace loading (positive rank correlation, CI excluding 0).
- **C5 Selectivity.** (a) Same-latent design: swap flips report/flexible answers in ≥60%
  of items while changing automatic-task outputs in ≤15% (and the latent is present in
  the lens in all conditions at comparable rates) [paper: ~100% vs ~0%]. (b) J-space
  ablation: on a capability-filtered battery of ≥6 tasks (≥3 shallow, ≥3 flexible),
  shallow tasks retain ≥80% of baseline while ≥2 flexible tasks drop by ≥40%; a
  norm-matched random-direction ablation leaves both classes ≥80% [paper: shallow
  unaffected under heavy ablation, flexible collapse; random controls inert]. Pretraining
  next-token top-1 agreement under ablation must stay far above the flexible-task
  retention [paper: "bulk of ordinary text prediction intact"].

**Band selection (frozen before any SmolLM2 lens data was seen).** The workspace
band used by all confirmatory runs is fixed by the layer-band analysis, not by
experiment outcomes: band END = last layer with lens-vs-model top-1 next-token
agreement < 0.5 (i.e. before the "motor" jump); band START = first layer where
excess kurtosis exceeds 2× the early-layer median AND top-1 autocorrelation
exceeds its shuffled null by > 0.05 (if the two disagree, the later layer).
Exploration may test sub-bands; any confirmatory deviation from the rule must be
frozen and logged in LAB-LOG before that confirmatory run. Because swaps applied
in the motor regime (J ≈ identity) edit logits quasi-directly and would be
trivial, motor layers are excluded from all intervention bands; C1/C3 include a
layer-window analysis demonstrating effects arise before the motor regime.

**Overall headline.** "A privileged set in the Eleos sense is present in model M" requires
all five criteria ≥ Hints and at least three Shown, with C1, C3 and C5 among those ≥
Hints necessarily including their privilege/selectivity controls. Anything less is
reported as the mixed result it is, criterion by criterion. We pre-commit to publishing
the per-criterion table whatever it contains.

Statistical conventions: Wilson 95% CIs for rates; bootstrap over items for correlations
and medians; per-experiment nulls as defined in the paper (position-shuffled,
random-direction, category-shuffled) — chosen per experiment *before* its confirmatory run.

## 5. Anticipated tiny-model complications (from the advice list + paper)

- Workspace band likely later & narrower than 38–92% of depth; possibly no clean band.
  Layer-band analysis (kurtosis, next-token top-k, autocorrelation, CKA, effective rank)
  runs *first* and fixes the band used everywhere else.
- 49k lens vectors in 576 dims → much heavier correlation among J-lens vectors than in
  Claude. Mitigations: gradient-pursuit decompositions (as in the paper), pseudoinverse
  swaps (already handle non-orthogonality), possibly larger α; report vector-correlation
  statistics so readers can see the crowding.
- SmolLM2 ties embedding/unembedding weights (verify); logit-lens behaves differently
  under tying; J-lens should be agnostic but worth checking against the paper's
  J-vs-logit-lens comparisons.
- Small models find "simple" tasks effortful (not automatic) — the automatic/flexible
  line sits elsewhere than for Claude. We locate it empirically (which tasks survive
  J-ablation) and, per the paper's own suggestion, treat J-space-independence as the
  operational definition of "automatic" — while keeping the *report/flexible side*
  defined by task demands, so C5 cannot become circular: the C5(a) task assignment
  (which tasks are "automatic") is fixed before any ablation is run, from human task
  analysis, exactly as the paper fixed theirs.
- Instruct-post-training at 135M is shallow; introspection protocols may not engage.
  Base-model (GPT-2) versions replace the instruction channel with few-shot patterning —
  a real adaptation, flagged as such; the paper's §6 (workspace present in base models)
  licenses the attempt but its report/modulation experiments will be the hardest to port.

## 6. Plan of record (updatable)

0. Environment; fetch anthropics/jacobian-lens reference implementation + Neuronpedia
   prefit lenses; validate our fit against theirs if any open-model lens is published.
1. Capability survey of SmolLM2-135M-Instruct (what tasks can it actually do?).
2. Fit J-lens (all layers, default recipe, 1000×128 pretraining-like tokens);
   sanity: motor-regime convergence to next-token prediction at final layers,
   J-vs-logit-lens divergence at earlier layers.
3. Layer-band analysis → fix workspace band.
4. C1–C5 exploration → freeze → confirmatory runs (order: C1, C3, C5, C2, C4 —
   report and reasoning first because they are the most diagnostic; selectivity next
   because it is the soul of "privileged").
5. Secondary: capacity/occupancy, MLP gain, broadcast heads (cheap structural extras).
6. Escalate models as needed; GPT-2 stretch goal.
7. Write-up with paper-vs-ours table for every number.

## Changelog

- 2026-08-16: Initial version, written after reading Gurnee et al. (2026) and the Eleos
  commentary in full, before any code or experiment.
- 2026-08-16 (late, pre-confirmation): **Band-rule amendment.** The original rule's
  thresholds (kurtosis > 2× early median; autocorr excess > 0.05) were derived from the
  paper's reported metric behavior and do not transfer to SmolLM2-135M: lens-readout
  excess kurtosis is flat (~0.2–0.8) at every layer and autocorr excess peaks at ~0.05,
  so the rule returns an empty band. Amended rule, still computed only on held-out
  wikitext (no experiment outcomes): band END = last layer with lens-vs-model top-1
  next-token agreement < 0.5 (unchanged); band START = earliest layer with top-10
  lens-vs-model agreement ≥ 0.10 (early layers sit at 0.03–0.07; the crossing is sharp:
  L18 = 0.069 → L19 = 0.101 → L20 = 0.184). Result: band = L19–L26 (of 30). Transparency
  note: preview experiments had already used L19:26 (from the same onset signals in the
  same analysis) before this amendment was written; to guard against band-shopping, every
  confirmatory headline is additionally reported for band starts {17, 19, 21} (end 26)
  as a pre-declared sensitivity analysis.
