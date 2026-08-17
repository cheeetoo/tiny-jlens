# Confirmatory protocol (draft → frozen before confirmatory runs)

Status: **FROZEN 2026-08-17 ~01:45 UTC**, before any confirmatory run.
No further edits except appending results. Pending items resolved from
chain D exploration as noted inline.

Global choices (all criteria):
- Model: SmolLM2-135M-Instruct. Lens: runs/smollm2-135m-it/lens.pt (n=1000
  wikitext, paper recipe) — the confirmatory lens, distinct from the
  exploration lens (250 prompts).
- Band: L19:26 (frozen amendment in BRIEF). Sensitivity: band starts
  {17, 19, 21} reported for every headline swap/ablation statistic.
- Swap machinery: clamped coordinate swap (paper Fig. 4C semantics), alpha=1
  primary; alpha=2 secondary column. RESOLVED: C1b uses the paper's §3.1
  projection swap (clamped, alpha=1) — chain D: proj 73/84 top-5 vs coord
  24/84; the projection form is the method §3.1 itself describes for this
  experiment. All other criteria use the coordinate swap.
- Statistics: Wilson 95% CIs on rates; paired bootstrap (10k resamples) over
  items for continuous contrasts; Spearman CIs by bootstrap over categories.
- Fresh material: seeds 1234 for all sampled choices; held-out items from
  confirm_pools.py wherever the pool family exists.

## C1 — verbal report
- 1a: 14 paper categories + 4 confirm categories. Statistic: mean Spearman
  over categories at each band layer; headline = mean at L24-26 with
  bootstrap CI; also report the L19->L26 trend.
- 1b: sources by pick_source rule; eligible targets exclude clean top-10 and
  source variants; n_targets=6/category, seed 1234. Headline: top-5 rate
  (Wilson CI), projection swap alpha=1; coordinate-swap alpha=1 reported as a
  secondary column; per-category table.
- 1c: concept vectors from "Tell me about {m}." last-position mean-diff;
  split by gradient pursuit k=16 (paper's C1 value) AND k=50 (scale-adapted,
  crowding-justified; both reported, k=50 flagged as adaptation). Matched-norm
  J vs nonJ swap + clamp control. Headline: J vs nonJ top-5 rates + clamped
  rate, k=16 primary for comparability, k=50 as adaptation column.
- 1d (secondary): introspection protocol, strengths {0, 8, 16}, 40 concepts.
  Headline: top-1 report rate and median RR at s=16 vs s=0; blurt rate must
  stay < 20% (selectivity).

## C2 — directed modulation
- Main battery (paper protocol): 10 topics x 4 carriers x conditions
  {baseline, focus x2, mention x2, dismissal x2, negated-think x2} with
  CONFIRM_CARRIERS added. Metrics: (i) paper hit metric (top-1) — reported
  as-is even if ~0; (ii) hit@10; (iii) continuous mean log-softmax of target
  over carrier span x band. Headline contrasts (paired bootstrap over
  (topic, carrier)): mention-baseline > 0; focus-baseline > 0;
  suppress-mention < 0. Math family: reported (expected null, capability).
- Demand variant (V1b; scale-adapted analog of the paper's implicit-task-
  demand experiment, flagged as adaptation): 8 categories x 3 carriers,
  demand vs no-demand paired best-rank; headline: paired median rank
  difference with bootstrap CI + sign test.
- 2c privilege: as scripted (tense/language/caps), n=12 stims per side.
  Reported as-is (exploration suggests weak imagine effects; this sub-part
  likely fails and will be reported as such).

## C3 — internal reasoning
- Items: ref probe-swap + custom pool + CONFIRM_TWOHOP, capability filter,
  fresh swap partners seed 1234. Degenerate-pair guard on.
- (a) readout: final-position top-10 rate over band (Wilson CI).
- (b) swap: clamped coord swap alpha=1, top-1 rate (Wilson CI); alpha=2 col.
- (c) anti-smuggling [AMENDED, logged]: cross-function consistency (phase
  `crossfn`) — for country pairs present in both lang-capital and
  city-language kept sets, apply the identical intermediate swap under both
  prompts; consistent = both answers flip to their respective correct
  counterfactuals. Headline: both-flip rate over eligible pairs, and
  both-flip / any-flip ratio. Depth profiles reported descriptively.
- (d) probe split: non-circular probe prompts, k=25 (paper) and k=50
  (adaptation) columns; matched-norm J vs nonJ + clamp. Headline as in bars.

## C4 — flexible generalization
- Grid: union format (per cell, raw-completion or chat-question, whichever
  the capability filter validates; --union). Filter >=3/4 args per cell;
  swaps between correct args only; alpha=1 primary.
- Headline: overall top-1 rate; per-category table; loading-success
  correlation (point-biserial, bootstrap CI). If <4 categories survive
  capability filtering, C4 verdict is capped at Hints per BRIEF (grid spec
  unmet) — reported plainly.

## C5 — selectivity
- (a) language design: 8 ref + 12 confirm passages; tasks report/country
  (flexible) vs continue (automatic); anomaly dropped (capability, logged).
  Single-pair swaps, alt-map German:Spanish, Italian:French, alpha=1.
  Headlines: flexible follow rate; automatic language-change rate (gradable
  trials); presence-in-lens rate across conditions. Continuation grading by
  stopword/charset classifier; ungradable trials excluded from the change
  rate and counted separately.
- (b) ablation battery: tasks = sentiment (10 + confirm 10), analogy
  (8 + confirm 8), lang->country (10), two-hop (custom+confirm, filtered),
  one-hop recall cells, forced-copy agreement, wikitext next-token agreement,
  PLUS passage_report / passage_country / (passage_continue in C5a run) on
  the C5a stimuli — protection-clean flexible tasks by design.
  Pre-declared predictions: shallow/automatic = {sentiment, copy,
  wikitext-agree} robust (>=80% retention at light/medium); flexible =
  {lang->country, passage_report, passage_country} impaired (>=40% drop for
  >=2 at light or medium); {two-hop, analogy, one-hop} reported as
  observed-with-diagnosis (exploration showed two-hop is protection-limited:
  the intermediate is a predictable next token mid-prompt and lands in the
  protected set; analogy appears genuinely automatic at this scale).
  Protection: paper-exact (clean top-10 per position) for the verdict;
  an unprotected column and the protection-overlap statistic (fraction of
  positions whose protected set contains the concept tokens) are reported as
  diagnostics. Controls: norm-matched noise (primary; paper §3.5.3-style) and
  random-direction projection (reported with removed-norm bookkeeping — it
  removes ~1.8x more norm than the ablation and is thus conservative in the
  wrong direction). k=10, strengths light/medium/heavy as scripted.

## Order of execution
1. C3, C1 (flagships), 2. C5a + C5b, 3. C2 battery + V1b + 2c, 4. C4, 5. C1d.
All runs write to runs/confirm_*.json; every number goes into the report
regardless of outcome.
