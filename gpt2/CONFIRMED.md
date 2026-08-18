# Confirmatory protocol — gpt2-small (frozen)

**Status: FROZEN 2026-08-18, before any confirmatory run.** Analysis choices
below may not change after this commit; results are reported whatever they
are. Materials: `pools_confirm.py` (held out — never used in any exploration
run; day-2's burned held-out sets excluded). The capability filter runs on
these materials first, symmetrically, before any lens measurement. BRIEF.md
§1–4 (old/docs/) remains the governing definition of criteria and verdicts.

## Global frozen choices

- Model: gpt2 (124M), the authors' released lens. (gpt2-medium repeats the
  suite when its n=1000 lens lands; interim-lens medium numbers are labeled
  exploration.)
- Gauge: all geometry (decomposition, clamps, coordinate reads, ablation) in
  the centered gauge; readouts are gauge-invariant (results/CONE.md).
- Interventions: clamped coordinate swaps, centered, α=1, layers L7–L10
  ("late"); pre-declared sensitivity window L5–L9 ("mid") for C3b. C1b uses
  α=2 primary (α=1 reported) per exploration.
- Grading: greedy top-1 with article extension (fixed ARTICLES set), surface
  variant sets; " Southeast" pre-declared correct for the continent of
  {Thailand, Vietnam, Cambodia, Malaysia, Indonesia}. Readout tables over all
  fitted layers; "in lens" = rank < 10.
- Statistics: Wilson 95% CIs on rates; exact binomial sign tests for paired
  orderings. Seed 1234 for all sampling.
- Runner: `TJL_CONFIRM=1` swaps materials; code paths identical to
  exploration scripts (same commits).

## Bars

- **C1a** (corr): mean lens-vs-output Spearman over qualifying held-out
  categories (member-in-top-5), at L9–L10: > 0 with bootstrap 95% CI
  excluding 0; rising trend across layers reported.
- **C1b** (swap-to-report): target enters graded top-5 in ≥40% of swaps at
  α≤2 (centered), targets outside output top-10 pre-swap.
- **C1d** (inject, secondary): at s=0.25, L7–10, inject-outside-last-3:
  report top-5 ≥40% AND ≥2× the matched blurt (control-prompt) top-5 rate.
- **C2** (modulation): conditions focus / ignore / base (the bake-off pair;
  both are paper condition types). Metric: best lens rank of the word over
  (transcription positions × all layers). Bars: (1) focus<base in ≥80% of
  pairs, sign p<0.001; (2) focus<ignore in ≥65%, p<0.01; (3) ignore<base
  majority (white-bear: suppression incomplete — directional); (4) the
  focus<base ordering holds among non-blurt trials (blurt = word in model's
  output top-10 anywhere in the span; rate reported).
- **C2 property dissociation** (42, secondary, no pass bar — exploratory
  port): claim-is-French vs real-French: lens z, J-orth probe z, full probe
  z reported; exploration expectation stated in advance: partial dissociation
  (claim moves the orth probe, ~3× less than real).
- **C3a** (readout): over UNSPOKEN-flagged capability-passing items
  (intermediate outside output top-10): in lens top-10 at some layer ≥50%.
- **C3b** (swap): counterfactual answer graded top-1 ≥30% (α=1, late window;
  mid window sensitivity column). Target-answer-outside-top-10 guard.
- **C3c** (crossfn): identical swap under two functions; both-flip ≥25% of
  eligible pairs, n≥8; any-flip and partial-failure modes reported.
- **C3d** (probe privilege): matched-norm component swaps: J-component flip
  rate ≥2× non-J; non-J with J-coordinates clamped ≤ half of non-J.
- **C4**: overall ≥25% top-1 (α≤2) with ≥1 category ≥50%; same-pair
  multi-function redirect rate reported. Grid breadth reported plainly
  (expected to cap C4 at Hints per BRIEF's grid spec).
- **C5a**: flexible (report + country) follows the swap ≥60%; automatic
  continuation truly redirects ≤15% (degradation-to-other counted
  separately); latent presence in the lens reported for all conditions.
- **C5b** (ablation, centered, protection = clean top-10): primary k=1,
  secondary k=3, layers L7–10. Bars per BRIEF: all three shallow tasks
  (cont_lang, copy, wikitext-agreement) retain ≥80% while ≥2 flexible tasks
  drop ≥40%, and both matched controls (rand-projection, norm-matched noise)
  leave flexible tasks ≥80%... Pre-declared expectation from exploration:
  wikitext-agreement will MISS the 80% bar (≈0.63 at k=1) — if only wikitext
  misses while the ordering (shallow ≫ flexible) and control contrasts hold,
  C5b is graded Hints, not Shown; stated before the run.
- **Occupancy / band / cone**: descriptive, no bars.

## Verdict rule

Per criterion: Shown / Hints / Not shown per BRIEF §4 (all key contrasts
paper-directional and significant + bars met = Shown; direction + privilege
contrasts significant but weaker = Hints; null/reversed key contrast or
misbehaving control = Not shown). Overall headline requires all five ≥ Hints
and ≥3 Shown, with C1/C3/C5 including their privilege/selectivity controls.
Every number is published regardless of outcome.
