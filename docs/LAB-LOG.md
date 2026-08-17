# Lab log

Chronological record of every run and decision, per BRIEF R3/R4 (report the
search). Entries: what ran, key numbers, what was decided. Exploration runs are
marked [E]; confirmatory runs [C] (analysis frozen before launch).

## 2026-08-16

- Read Gurnee et al. 2026 + Eleos commentary in full; wrote BRIEF.md
  (pre-registration: 5 criteria, verdict rules, concession criteria).
- Found Anthropic reference implementation (anthropics/jacobian-lens) with the
  paper's prompt datasets; installed it as the fitting/back end. Found
  Neuronpedia prefit lenses incl. gpt2-small, pythia-70m, gemma-3-270m —
  gpt2-small lens (authors' own pipeline) downloaded for validation + later use.
  Neuronpedia fit recipe (config.yaml): wikitext-103-raw-v1 train, max_chars
  2000, n=1000 (early stop delta 2e-3), seq 128, dim_batch 128, target=final,
  bf16. Our corpus loader mirrors it.
- [E] Capability survey SmolLM2-135M-Instruct (runs/smollm2-135m-it-capability.json):
  - category report: instances for ~6/14 categories (echo-the-category failure
    mode on others; prompt iteration needed)
  - flex-gen cells: countries/capital+continent+currency 4/4, language 2/4;
    months/season 3/4; animals/class 3/4; numbers/first_letter 3/4
  - two-hop probe-swap: ~20/90 genuinely correct (families: lang→capital,
    city→language/currency, person→country, food→animal); grading v1 had
    substring bugs (fixed in C3 script: exact first-token match)
  - copying near-perfect (7-8/8), language naming 8/8, sentiment 5/6,
    tier-1 math 5/8; hello-in-language 0/8, odd-one-out 0/4
- Smoke test on prefit gpt2-small lens: my LensKit readout == reference apply
  (0.0 max diff). gpt2-small J-lens: interpretable content only at L10/11
  ("currency/Yen/USD" on the boot prompt) — early layers junk tokens. gpt2
  can't answer the boot two-hop at all. Confirms SmolLM2 as primary target.
- Fitting: gpt2 halves A/B (150 wikitext prompts each, disjoint) for pipeline
  validation vs Neuronpedia prefit. SmolLM2-135M-Instruct n=1000 fit queued
  after (was too slow under contention; restarted post-gpt2).
- Wrote experiment scripts (exploration-ready): c1_report, c1d_introspection,
  c2_modulation, c2c_modulation_privilege, c3_reasoning, c4_flexibility,
  c5a_selectivity_language, c5b_ablation, band_analysis, validate_fit_gpt2.
- Froze the band-selection rule in BRIEF §4 (kurtosis + autocorr onset; motor
  cutoff at lens-model top-1 agreement 0.5) before seeing any SmolLM2 lens.
- Wrote held-out confirmatory pools (confirm_pools.py): fresh language
  passages, sentiment, analogies, two-hop items, categories, carriers.
- **Fit validation**: our gpt2 lens A (150 wikitext prompts) vs Neuronpedia
  prefit (277 prompts): per-layer matrix correlation 0.989 (L0) → 0.999 (L10);
  norm ratio 0.90-0.98. Pipeline validated against the authors' own artifact.
  (A-vs-B noise ceiling + readout Jaccard to follow when fit B lands.)
- **Full fit validation**: ours(A+B, 300 prompts) vs Neuronpedia(277):
  corr 0.993-0.9996/layer, top-10 readout Jaccard 0.77-0.87 — both EXCEED the
  A-vs-B same-recipe noise ceiling (corr 0.964-0.998, Jaccard 0.64-0.78).
  Fitting pipeline == authors' pipeline up to prompt sampling. [gpt2-small]

## 2026-08-16 (evening) — preview explorations on 25-prompt checkpoint lens [E]
- Band preview (band_preview25.json): autocorr-null jumps at L19; lens-model
  top1 crosses 0.5 at L27 → preview band L19:26 (~last quarter of depth,
  later+narrower than paper's 38-92%). Kurtosis metric flat (does NOT
  reproduce paper's onset shape at this scale/lens size) — flagged; recheck on
  full lens; band rule may need amendment (would be logged pre-confirmation).
- Found+fixed grading bug: SmolLM2 splits digits (" 5"->[' ','5']) so numeric
  answers were graded on the space token. first_token_id now skips whitespace
  tokens; degenerate swap pairs (same first token) skipped.
- Found+fixed REAL method bug: dynamic per-layer re-swap oscillates (sigma is
  an involution — consecutive band layers toggle the swap). Replaced with the
  paper's clamped semantics (coords held at sigma(clean) at every band layer).
  Effect: C3 swap went 0/43 -> 24/43 top-1 at alpha=1 (56%; Haiku=54%,
  Sonnet=70% in paper). alpha=2 degrades (11/43) — alpha=1 is the regime.
  Failures are mostly near-misses (rank 1-3) with +5..+10 dlp.
- C3 readout (final-position, fixed grading): 31/43 = 72% top-10 [bar: 50%].
- C3 capability filter: 44/136 items (15 ref + 29 custom).
- C1 preview [E]: 1a Spearman rises monotonically L19 0.28 -> L26 0.65 over 14
  categories (paper's exact pattern: correlation grows toward late workspace).
  1b swap-to-report: 17/84 top-1, 23/84 top-5 after article-skip anchor fix
  (below bar; retest on full lens). 1c: J-comp 5/9 vs nonJ 5/9 top-5 (language
  category shows textbook privilege, color/organ don't); clamped-nonJ 0/9
  (nonJ effect routes through J-space). Hypothesis: gradient pursuit on the
  noisy preview lens under-extracts, leaving J-content in the "remainder" —
  diagnose on full lens (project remainder through lens; check mass).
- C2 preview [E]: NULL. No focus/suppress/mention/baseline separation; all
  median best-ranks ~chance for min-over-cells statistic. Condition separation
  is lens-noise-independent -> genuine risk that covert "hold in mind during
  copying" is absent at 135M. Full-lens retest with protocol variants
  (all-layer readout, rank<=5/10, continuous logprob metric, stronger
  phrasings) before any verdict.
- C5a preview [E]: presence control 8/8 top-3 in all conditions; report
  follows swap 3/8 (fr/es yes, de/it no); continuation invariant 5/8 (2 broke
  to English = degradation not redirect); country task breaks under swap (0/8
  follow) — prefill weak ('predominantly') + swap disrupts. Redesigned:
  country prefill now completion-style; anomaly rebuilt as spliced-vs-clean
  yes/no logit-margin discrimination + margin-sign invariance.
- PAUSING GPU exploration to let the n=1000 fit sprint (~2.5h solo).

## 2026-08-16 (night) — 250-prompt exploration lens; band frozen
- Fit survived infra outage (background process on the box); snapshot at 250
  prompts = same convergence class as published Neuronpedia lenses (gpt2: 277,
  qwen0.8b: 233). lens_explore.pt is the exploration lens; the n=1000 lens
  remains the confirmatory lens.
- Band analysis at 250 prompts reproduces preview: onset L19, motor L27+.
  BRIEF band rule amended (see BRIEF changelog): kurtosis/autocorr thresholds
  do not transfer; new outcome-independent rule (top-10 agreement >= 0.10
  onset; top-1 < 0.5 motor cutoff) -> BAND = L19:26. Sensitivity analysis over
  band starts {17,19,21} pre-declared for all confirmatory headlines.

## 2026-08-16/17 (night) — C3 full exploration [E] + geometry diagnostics
- C3 on 250-prompt lens, band L19:26: readout final-pos 30/43 top-10 (70%);
  clamped swap 24/43 top-1 at a=1 (56%); probe phase n=28: full 16/28 top-5,
  J-comp 17/28, nonJ 16/28, **nonJ-with-J-coords-clamped 1/28 (median rank
  3 -> 365)**. Mean J_var_share of probes 5.4% vs ~1% for random dirs.
- Timing control: three-regime profile (early input-rewrite channel works,
  mid-network dip, band effect) — the paper's intermediate-earlier-than-answer
  depth separation is not resolvable in an 8-layer band (median diff 0).
  AMENDMENT (pre-confirmation): replace C3(c) depth-timing with a stronger
  anti-smuggling control — cross-function consistency (one intermediate swap
  must redirect two different questions to their respective correct
  counterfactual answers; a smuggled answer vector cannot do both). Depth
  profiles still reported descriptively.
- Geometry: lens vectors form a near-degenerate cone at d=576 — top PC = 78%
  of variance, mean pairwise |cos| 0.78 -> 0.07 after removing 1 PC (~2x the
  0.033 random level). Explains why matched-norm J/nonJ component asymmetry
  (paper: 59%-vs-5%, 61%-vs-28%) washes out here while clamp-mediation stays
  decisive. Verdict impact: C3(d)/C1(c) asymmetry clause likely fails as
  operationalized; mediation clause passes strongly. Will report both,
  verdicts per frozen rules. k-sweep (k=50) queued to test the
  spread-across-frame-tail hypothesis.

## 2026-08-17 (early) — chain A digest [E]
- Lens-eval at 135M: J-lens pass@5 30/44 vs logit lens 1/44 on capability-
  filtered two-hop intermediates. The Jacobian transport is nearly all of the
  signal at this scale (in-paper the gap was "modest on multihop" for Claude).
- C1a replicates on 250-lens (0.29->0.65 rising Spearman). C1b 24/84 top-5:
  failure structure fully diagnostic — fragment/markup argmax sources
  ('co','par','fl','**') have no usable lens vector; patched source rule =
  argmax if clean word else top candidate-list token. city/color/fruit/sport
  already 23/24 top-5.
- C2 V1 (hold-then-report, task demand present): held item IS in lens during
  copying (ranks 0-5) where answer extraction succeeded. V3 continuous metric:
  focus-baseline +0.79 logprob at L23. The C2 null was metric strictness +
  no-demand: modulation exists, demand-driven. c2 patched with hit10 +
  continuous metrics; V1 extraction fixed.
- C4: 20/24 top-1 (83%) but only countries survived strict cell filter; added
  months2/animals2/numbers2 custom categories with single-token answers.
- C5a round 2: report 4/8, country 4/8 follow (prefill fix worked);
  continuation invariant 5/8 (breaks are to-English degradation, not
  redirects); anomaly: model cannot discriminate spliced passages (1/8) ->
  anomaly task dropped for capability, logged. Round 3 queued: single-pair
  swaps + German:Spanish/Italian:French alt map.
- C3 probe k=50 with non-circular probes: J 9/28 vs nonJ 2/28 top-1 (4.5x,
  paper ratio 2.2x), clamp -> 0. Asymmetry emerges at k=50; occupancy
  measurement queued to justify k at this scale (crowded frame).

## 2026-08-17 — chain B digest [E]
- C2 continuous metric over n=360 topic trials: mention(-18.77) ≈ focus(-18.74)
  > baseline(-19.25) ≈ suppress(-19.31) — mention-priming (+0.5 nats) and
  ignore-cancels-mention both present (paper's Fig-65 pattern in miniature);
  focus adds nothing beyond mention; strict top-1 hits ~0 at all conds; math
  family null (covert arithmetic beyond capability).
- C2 V1 hold-then-report (WITH later-use demand): held item in lens during
  copying, typical best rank 0-25. V1b matched no-demand control queued —
  demand-vs-no-demand is the scientifically clean C2 contrast at this scale.
- C1d introspection dose-response: s=8 RR 0.14 (6/24 top-1), s=16 RR 0.20
  (8/24 top-1), blurt-at-other-positions only 2/24 (selectivity holds).
- C5a round 3 (single-pair swaps, de->Spanish/it->French alts): flexible
  follows 12/16 (75%) ✓; continuation invariant 6/8 with 1 real break
  (es2->English), 1 ungradable; German remains stubborn (unchanged, not
  redirected). Anomaly task dropped (capability: discriminates 1/8).
- C4: capability filter too strict from grading artifacts (case, articles,
  digit-vs-word). v3 grading: variant sets + article-skip + digit alternates.
- Occupancy v1 gave median 1 with an unfair control (orthonormal projections
  vs greedy-nonneg); v2 uses same-algorithm random-dictionary control.

## 2026-08-17 — chain C digest [E]
- V1b demand-vs-no-demand: held item maintained in lens during copying ONLY
  under later-use demand — 14/14 paired rows demand<no-demand (e.g. 0 vs 45,
  3 vs 23, 6 vs 71). The clean C2-family contrast at this scale (implicit
  task-demand modulation, cf. paper §3.2 paired-question protocol).
- C2c: "imagine X" moves lens AND probe weakly (z<1.1) — no clean
  dissociation; real stimuli move probes z=11-26. Instruction-driven top-down
  modulation is weak at 135M; demand-driven modulation is strong (V1b).
- C1c n=30 (fixed sources): J 12/30 vs nonJ 8/30 top-5 (direction ok, below
  3x bar), clamp 0/30 (decisive). C1b unchanged 24/84 — target prior-distance
  gating (eligible targets sit at rank 2-16k in peaked categories); trying
  proj-swap (paper §3.1 form) + alpha 2.
- C1d dose-response saturates ~RR 0.2 (9/24 top-1 at s=24, blurts 3/24).
- Occupancy v2 (fair same-algorithm random-dictionary control): median 1,
  excess var ~-0.01. STRUCTURAL DISANALOGY: at 135M the J-frame is a
  degenerate cone + tail, not a ~25-slot sparse code (paper: occupancy ~25).
  Functional criteria are carried by a much thinner representational
  structure. Major finding for the report.
- C4 grading fixes recovered only: countries capital/continent 4/4,
  language 2/4, animals/class 2/4, numbers2/bigger5 2/4. Chat-format grid
  queued as the last capability attempt.

## 2026-08-17 ~01:45 — chain D digest + CONFIRMATORY FROZEN
- C1b projection swap (paper §3.1 form): 53/84 top-1, 73/84 top-5 (87%; paper
  88%). Coordinate form alpha=2: 34/84. C1b confirmed method = proj a=1.
- C3 band sensitivity: 24/43 (17:26), 24/43 (19:26), 27/43 (21:26) — robust.
- C4 chat cells: months/season 4/4 recovered; union-format mode added; grid
  still countries-heavy — C4 expected to cap at Hints per BRIEF spec.
- C5b: protection-overlap mechanism identified (two-hop intermediates are
  predictable next tokens mid-prompt -> protected -> ablation-robust;
  lang_country/passage tasks protection-clean -> collapse 75-100% under
  light/medium while sentiment/copy/wikitext hold). Removed-norm bookkeeping:
  J-ablate 8.2% vs random-projection control 14% (control over-destructive;
  noise_matched is the fair control). k sweep: k=25+ hurts shallow tasks too.
  Confirmatory: k=10, light/medium, paper-exact protection + diagnostics.
- CONFIRMATORY.md frozen (all PENDs resolved) before any confirmatory run.

## 2026-08-17 02:30-03:00 — CONFIRMATORY RESULTS [C] (n=1000 lens, frozen protocol)
Note: two suite instances raced from a dead-watcher misdiagnosis; identical
frozen protocol + seed -> identical outputs (verified by solo rerun: 13s,
same numbers). Results valid.

VERDICT TABLE (frozen bars, BRIEF §4):
- C1 Report: 1a Spearman(L24-26) 0.548 CI[0.42,0.66] ✓; 1b proj-swap top-5
  84% CI[76,90] (paper 88%) ✓✓; 1c J-vs-nonJ rates 36% vs 25% (<3x bar ✗)
  but paired J<nonJ 29/31 p<0.001 ✓ and clamp 9->0 p=0.004 ✓✓; 1d s=16 RR
  0.25, 7/24 top-1, blurts 3/24. → **HINTS** (misses only the 3x rate
  asymmetry; mediation-privilege decisive).
- C2 Modulation: primary hit-rate bar FAILS (focus=baseline=2.5%); continuous
  contrasts significant and paper-directional (focus-baseline +0.50
  CI[0.42,0.58]; suppress-mention -0.48 CI[-0.52,-0.44]; mention≈focus, as
  paper found); C2c imagine-dissociation fails (z<1.1); V1b demand-vs-
  no-demand 23/24 p<1e-5 (median rank 6.5 vs 64) [flagged adaptation].
  → **NOT SHOWN** on pre-registered bar; strongest partials of any criterion.
- C3 Reasoning: (a) 72% CI[57,83] ✓ (b) 63% CI[48,76] ✓✓ (Haiku 54/Sonnet 70)
  (c) crossfn 15/30 both-flip (29/30 any) ✓ (d) J 43% vs nonJ 14% = 3.0x ✓,
  clamp 4->0 ✓✓; band-robust (27-28/43 across starts 17/19/21).
  → **SHOWN** (all clauses).
- C4 Flexibility: 73% CI[56,86] top-1 within countries (paper countries
  "almost perfect"); grid breadth unmet (capability wall) → **HINTS** (capped
  per frozen rule). Loading correlation weak/ns.
- C5 Selectivity: (a) flexible follows 78% CI[63,88] ✓; automatic changed
  3/19=15.8% (bar <=15%: miss by 0.8pp; 1 of 3 changes is a true redirect
  leak, 2 are English-drift degradations); flexible-vs-automatic contrast
  p<0.0001 → (a) effectively holds with a hair-miss on the absolute bar.
  (b) ablation battery FAILS pre-declared conjunction (only lang_country
  dropped >=40%; passage tasks flat — long-prompt redundancy re-forms
  content; noise control comparable on lang_country). → **HINTS**.

OVERALL (frozen headline): requires all >= Hints and >=3 Shown. Achieved:
1 Shown (C3), 3 Hints (C1, C4, C5), 1 Not shown (C2). **Headline claim NOT
met**; per pre-registration, reported criterion-by-criterion as the mixed
result it is.

GPT-2-small (authors' own prefit lens, frozen band rule -> L7:10):
capability filter 25/136; readout 19/24 (79%); swap 20/24 (83%) top-1 a=1.
Probe-split does not transfer (GP captures 0.16% of probes); crossfn n=0
(second function family needed for gpt2).

Scale ladder: SmolLM2-360M lens fit launched.

## 2026-08-17 03:00-04:00 — report, ladder extensions [E]
- REPORT.md written; artifact published (claude.ai/code/artifact/107c4043-...).
- GPT-2 stretch completed: few-shot 1a Spearman ->0.59; 1b proj-swap FAILS
  (anti-directional 55/56) but coordinate-clamped swap works: 39/56 top-5
  (70%) at a=2 — per-model method sensitivity (both forms are paper forms).
  crossfn infeasible on gpt2 (no country passes two functions).
- pythia-70m (authors' prefit lens): 0/136 two-hop capability — the floor
  where the question dissolves. gemma-3-270m gated (no HF token).
- qwen3.5-0.8b band (frozen-rule metrics): onset needed a persistence
  qualifier (early blip at L2); band L12:21. NOTE: kurtosis rises 0.6->1.5
  here — the paper's kurtosis onset signature begins working at 800M.
- Qwen-0.8b chain launched: the key question is whether C2 (the 135M failure)
  turns on at 800M with identical code — brackets the C2 threshold with the
  360M point landing later today.

## 2026-08-17 ~04:30 — qwen3.5-0.8b ladder results [E] (authors' prefit lens)
- **C2 TURNS ON at 800M**: focus hit-rate 21.2% vs baseline 2.5% (8.5x — passes
  the original pre-registered 5x bar); hit10 63.7% vs 10%; median rank 5 vs
  222; full paper ordering incl. focus>mention (21.2% vs 6.2%) and
  suppress<mention. Identical code that returned null at 135M — implementation
  concern fully discharged; C2 threshold bracketed (135M<t<800M; 360M pending).
- C1 at 800M: 1a rises to 0.83; 1b 79/81 top-1 (98%); **1c full variance
  privilege: J 37/42 (88%) vs nonJ 5/42 (12%), clamp->0** — the paper's
  asymmetry pattern appears (even stronger than Claude's 59/5 ratio-wise).
- C2c imagine-dissociation STILL absent at 800M (z<1.1; real-stim probes
  z=2.8-25.8) — not merely a tiny-model failure.
- C3 at 800M under the unmodified 135M protocol: filter 51/136, readout
  final-pos 45% (any-pos 92%), swap 45% — lower than 135M; likely band/anchor
  mismatch (no per-model tuning done, deliberately); reported as-is.
- V1b not clean on qwen (extraction artifacts + ceiling); 135M-targeted design.
- pythia-70m: 0/136 capability floor. gemma-3-270m: gated repo, skipped.
- REPORT.md + artifact updated with the dose-response ladder section.

## 2026-08-17 ~09:40 — SmolLM2-360M ladder battery [E]
- Band L18:29 of 32 (56-91% of depth — widening toward paper's 38-92%).
- C2: focus 7.5% vs baseline 2.5% hits (3.0x; below 5x bar), hit10 33.8% vs
  10%, median rank 30 vs 78. DOSE-RESPONSE: 1.0x (135M) -> 3.0x (360M) ->
  8.5x (800M, passes). Strict-bar threshold between 360M and 800M; onset
  between 135M and 360M.
- C1: 1b 84/84 top-1 (100%). 1c J 86% vs nonJ 47% (1.8x), clamp ->1/36.
  Variance-privilege ratio monotone: 1.4x -> 1.8x -> 7.4x.
- C2c: within-family imagine-lens effect grows (0.68->1.12 tense,
  0.59->1.45 language) with probe still ~flat — dissociation emerging;
  cross-family (qwen) comparison noisy.
- C3 fixed-pool non-monotonicity: readout 72->62->45%, swap 63->52->45%
  across 135M->360M->800M. Interpretation (speculative, flagged): items
  designed at the 135M capability frontier become progressively automatic
  for larger models; the paper's own selectivity logic predicts reduced
  workspace mediation for automatized tasks.
- V1b at 360M: 17/22 demand<nodemand (p~0.008), medians 9.5 vs 31.5.

## 2026-08-17 ~12:30 — C2-REVISED: instruction control IS present at 135M [C]
- Motivation: user's prior LW study (no size trend in internal state control
  down to 270M; tiny-model effects late-layer, held-not-blurted). Their
  protocol mirrored: post-sentence instruction templates, think/dont_think
  paired contrast, probe + lens measures at ALL layers, blurt controls.
- Exploration: think median best-rank (L23:28) 28 vs baseline 472;
  think<dont 34/36; think blurt 4/36. Mention echo-contaminated (blurt 22/36).
- CONFIRMATORY (held-out words+sentences, frozen bars in CONFIRMATORY.md
  addendum): ALL FOUR BARS PASS — think 45 vs baseline 564 (36/36, p=3e-11);
  think<dont 34/36 (p=2e-8); dont<baseline 34/36 (white-bear: suppression
  partial); blurt 4/36 with ordering preserved among non-blurt trials.
- VERDICT: C2-revised = SHOWN at 135M under rank-sensitive late-layer
  operationalization. Original frozen C2 verdict (paper's top-1 hit metric,
  band L19:26) remains Not shown — both reported. Original protocol was
  triply insensitive: band excluded L27-28 (where the effect peaks), top-1
  hits can't see a 1000->100 rank shift, and template/contrast differed.
- Consequence: the "instruction-modulation switches on with scale" story is
  WRONG as stated; scale increases effect strength in top-of-lens terms, not
  the existence of control. Ladder rerun under revised protocol in progress
  (gpt2-raw / 360M / qwen). Report + artifact to be updated.
- C2-revised ladder: gpt2(base,raw) think 17 vs base 749 (23/23 strict
  non-blurt; think<dont 21/23); 360M think 1 vs 206; qwen think 1 vs 854.
  Span-max blurt flag saturates at 360M+ (36/36) — held-vs-spoken dissociation
  cleanly measurable only at the small end; position-level analysis needed
  above ~360M (noted in report). Old "threshold" fully explained as top-k
  metric crossing the dominance level. REPORT/artifact/LW-draft updated.

## 2026-08-17 ~14:00 — late-layer audit [E] (user's original hint, systematized)
- C3 per-layer readout at final position: 0/44 for L14-21, cliff to 32/44 at
  L23, sustained thru L28. Single live region L23-28; band rule had found
  generic wikitext onset (L19), not task content. C3 readout 77% incl. late
  layers (confirmatory 72% stands as scored).
- C2c at L23:28: imagine lens-z doubles/triples (lang 0.6->2.0) with
  J-orthogonal probe mostly flat; weak-present, not absent; still << paper.
- C5b at L23:28: failure ROBUST to band aim — sentiment crashes like flexible
  tasks, passage_report indestructible (20/20 everywhere), noise control ~=
  ablation on twohop. Whole-space ablation doesn't dissociate at 135M in any
  band; selectivity evidence rests on C5a (paper's primary design anyway).
- Cone persists at L27-28 (top-PC 0.69-0.70). Probe split at L23:28: J 16/28
  vs nonJ 10/28 (ratio shrinks to 1.6x as both gain power near output);
  original-band 3.0x stands.
- Blurt clarification recorded: at 135M/gpt2 the held word is in the model's
  actual output top-10 in only 4/36 / 13/36 trials (held-not-spoken); at
  360M+ the span-max flag saturates (dominance).
