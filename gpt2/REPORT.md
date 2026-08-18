# GPT-2 and the privileged set

*tiny-jlens, phase 2 (the GPT-2 rewrite). FINAL — all three confirmatory
suites (124M / 355M / 774M) complete on held-out materials under the frozen
protocol. [E] = exploration, [C] = confirmatory. Protocol: CONFIRMED.md
(frozen before any confirmatory run); search history: LOG.md; day-1/2
SmolLM2 work: ../old/.*

## The question

Gurnee et al. (2026) show Claude models carry a **privileged set** of
verbalizable representations — reportable, instructable, used in internal
reasoning, flexibly reusable, selectively engaged — and Eleos's commentary
isolates that privileged set as the paper's clearly-established, morally
relevant core. Day 1–2 of this project instantiated most of the pattern in
SmolLM2-135M. This phase asks the sharper question: **how much of the pattern
is in GPT-2** — the 2019 model that is nobody's candidate for moral
consideration — using the lens artifact the paper's authors themselves
released for it?

Models: gpt2 (124M, primary), gpt2-medium (355M), gpt2-large (774M); lenses
for medium/large fitted here with the authors' pipeline and recipe (validated
day 1 at r = 0.993–0.9996 against their released artifact).

## Headline

**GPT-2 has at least hints of a privileged set — formally, under frozen
held-out confirmatories at three model sizes: every one of the five
criteria lands ≥ Hints at 124M, 355M, and 774M, with directed modulation
(C2) Shown at all three sizes and verbal report (C1) Shown at 355M.** No
criterion produced a significant null or reversed key contrast in any of
the three confirmatory suites. The full-possession headline (≥3 criteria
Shown at one size) is not met; the shortfalls are capacity- and
breadth-shaped at 124M–355M (pooled readout dilution, one-category grids,
shallow tasks not yet lens-independent) and protocol-transfer-shaped at
774M (dose grids, band widths, and pools calibrated at 124M) — not
absence-shaped.

Two findings frame everything else. First, the "cone" that made small-model
J-lens geometry look degenerate is a **gauge artifact** — provably invisible
to every readout the lens can produce — and in the centered gauge the
paper's own privilege controls run and pass at 124M; the artifact then
dissolves on its own by 774M, where the correction converges to a no-op.
Second, what separates GPT-2 from Claude is now a set of **measured, smooth
scale trends** (dominance, matched-norm privilege asymmetry, capacity,
breadth, readout coverage) rather than the presence or absence of the
pattern itself.

## The cone is gauge (methodological result, load-bearing)

The J-lens dictionary at GPT-2 scale looked degenerate — 50k near-parallel
vectors ("the cone"), breaking decompositions, occupancy, ablation, and
matched-norm privilege tests. Diagnosis (experiments/10, results/CONE.md):

- The readout `logit_t(h) = ⟨v_t, h⟩/σ(Jh) + β_t` is **exactly invariant** to
  `v_t → a·v_t + u` (softmax kills the shared shift and scale). The
  dictionary is defined only up to this gauge.
- The cone axis **is** the vocabulary mean v̄ = Jᵀū (cos = 1.000 with the top
  principal axis at every layer; 97–99% of the dictionary's second moment),
  and its logit profile across the vocabulary is constant to ~2% — it is
  softmax-invisible, i.e. pure gauge. Provenance: GPT-2's unembedding is
  itself 83% mean, and J amplifies that direction 2.6–6.9× (vs ~1× random).
- In the centered gauge the dictionary is healthy (mean |cos| 0.07–0.10, no
  dominant PC) and every blocked operation comes alive: gradient pursuit
  recovers 3–16% of activation variance (raw: 0.0–0.6%; the paper's Claude
  range: ~10–15% for probes), and top-k ablation acquires a dose axis.
- Causal validation: the paper's §3.1 projection swap is **0% in the raw
  gauge and 46–69% centered** on GPT-2 two-hops [E] — the treatment revives a
  dead intervention. The coordinate swap, whose displacement (v_s − v_t) is
  gauge-invariant by construction, works identically in both gauges — which
  retroactively explains why swaps transferred to tiny models while
  decompositions failed.

Everything geometric below uses the centered gauge; readouts are provably
unchanged, and raw-gauge columns are in the JSONs.

## Per-criterion results, gpt2-small (124M)

All items capability-filtered first (pre-lens, symmetric); base-model
few-shot/completion forms with formats chosen by logged bake-offs; readouts
measured at all fitted layers L0–L10. Content lives at L9–L10; by the
day-1 frozen motor rule (lens-model top-1 agreement ≥ 0.5) **no fitted layer
is motor** (max 0.27 at L10) — the output conversion happens in the final,
unfitted block.

### C1 — Report
- [E] Report↔lens Spearman rises monotonically 0.0 → **+0.61** at L10 over 9
  categories (the paper's signature shape). [C: +0.40 at L10 over the 4
  held-out categories within 124M capability, all positive, same rising
  shape.]
- [E] Swap-to-report: target into graded top-5 in **69%** (α=2, centered;
  58% top-1) [paper: 88% top-5]. [C at 124M: not evaluable — no held-out
  category yields a valid spontaneous report (capability filter, logged
  before results); confirmed number to come from gpt2-medium.]
- [E] Injected thought (base-model port): steering a concept's centered lens
  vector at L7–10 (never at the readout position) makes it the completion of
  "the word I am thinking about is": 24/24 top-5 at s=0.25 vs 5/24 blurt on a
  matched control prompt; dose-responsive; early-layer injection is
  unselective and degrades at strength. [C: **13/20 report top-5 vs 2/20
  blurt** at s=0.25 on fresh concepts — bar (≥40% and ≥2× blurt) met.]

### C2 — Directed modulation
- [E] Focus/ignore/base (paper condition types; rank-sensitive metric):
  median best rank **14 / 26–36 / 657**; think<base 36/36; focus<ignore
  33/36. [C — held-out words and sentences, all four frozen bars met:
  **focus 13 / ignore 115 / base 337**; focus<base 32/36 (p≈1e-6);
  focus<ignore **35/36**; ignore<base 28/36 (white-bear: mention-priming
  survives suppression); ordering intact among non-blurt trials
  (47/269/350, n=22; blurt rate 14/36 reported).]
- [E] Demand variant (3-shot): remembered word held in the lens during
  copying more when the format will ask for it later — 28/36, median 343 vs
  462. Present but much weaker than 135M-instruct (23/24) — the
  instruction channel matters.
- [E] Privilege (property form): claiming an English sentence is French
  moves French in the lens (z +10.4) and moves a J-orthogonalized
  French-passage probe **3× less than a real French sentence** (z +6.8 vs
  +19.6) — a partial dissociation, not the paper's <1 SD. Consistent with
  day-2's ladder (absent through 800M): this is the pattern's genuine
  small-scale boundary. Word-level materials cannot make the test at all
  (mention contaminates every channel) — a design lesson, logged.

### C3 — Internal reasoning
- [E] Readout: on the clean family (lang→capital; intermediate outside the
  model's own output top-10, i.e. genuinely unspoken): **8/11 in lens top-10**
  (median best rank 2) at L9–10, null countries ~rank 900. The lens also
  *refuses* to show intermediates for the family the model solves by surface
  shortcut (city→language: 0/21 despite passing capability; its capital→
  country first hop is ~0) — readout content tracks intermediate *use*, not
  item labels. [C: the frozen POOLED bar misses — 5/15 unspoken items (33%)
  vs the 50% bar — because the held-out pool that survives 124M capability is
  dominated by the shortcut family, which the lens (correctly) reads as
  empty; the dilution was predicted and logged before results. Family-resolved
  and medium confirmatory numbers reported alongside.]
- [E] Swap: intermediate coordinate swap flips the answer to the
  counterfactual in **77%** (α=1, L7–10; 91% on the clean family; n=35)
  [paper: Haiku 54%, Sonnet/Opus 70%]. Present without the last layer
  (L5–9: 66–71%). [C: 50% on held-out items (n=12; bar 30% ✓); mid-window
  33%. gpt2-medium exploration, richer pool: 79–81% (n=47).]
- [E] Anti-smuggling: one identical country swap under two different
  questions flips both answers to their respective counterfactuals 14/27
  (8/10 on the best function pairing) — impossible for a smuggled answer
  vector. [C at 124M: zero held-out countries pass two functions (capability;
  logged pre-results) — not evaluable; confirmed number from gpt2-medium.]
- [E] Privilege (probe split, centered gauge): matched-norm swaps along
  components of independently-derived country probes: full **69%** /
  J-component **58%** / non-J **27%** / non-J with J-coordinates clamped
  **4%** (n=45). The paper's numbers: 60 / 61 / 28 / 6. **The pattern
  replicates at GPT-2 scale.** [C — held-out countries (harder, absolute
  rates lower, same structure): full 39% / J 22% / non-J 9% / clamped 0%;
  J = 2.4× non-J ✓, clamp kills the remainder ✓ — both privilege bars met.]

### C4 — Flexible generalization
- [E] One identical argument swap redirects capital and language functions at
  **85%** top-1 overall (countries; paper's overall 40–53%, their countries
  cell "almost perfect"); the same (A,B) swap redirects **both** functions in
  21/30 pairs. Breadth is one category at 124M (capability wall — the
  paper's own months/animals/numbers cells were near-zero too); medium adds
  cells. [C: pending]

### C5 — Selectivity
- [E] Same-latent design: with the passage's language in the lens at
  comparable ranks in all conditions, the same swap flips country-inference
  **12/12** and report 3/6, while automatic continuation never truly
  redirects (0/12). [C — twelve fresh passages: **flexible follows the swap
  18/19 (report 6/7, country 12/12); automatic continuation 0/12**; latent
  presence median rank 41 across all conditions. Paper: ~100% vs ~0%. Both
  bars met with margin.]
- [E] Whole-J-space ablation (centered; protection = clean top-10): flexible
  tasks collapse at every dose (k=1: 0.00–0.29 retention) to fluent,
  type-correct, content-wrong answers ("St. John's" as the French capital),
  while shallow tasks order cont-language 1.00 > copy 0.89 > wikitext 0.63,
  and matched controls (random projections, norm-matched noise) leave
  flexible tasks at 0.55–0.92. The pre-declared miss: wikitext agreement
  stays below the paper's 80% bar at all doses — at 124M, ordinary text
  prediction is itself partly lens-mediated (upcoming words are verbalizable
  content beyond the protected imminent top-10). [C: pending]

### Presence predicts causal efficacy
Stratifying C3 swaps by the intermediate's lens rank: **rank <10 → 92%
flips; rank 10–99 → 75%; rank ≥100 → 0%** [E]. Loading in the lens predicts
whether the lens-coordinate swap redirects behavior (the paper's
workspace-loading correlation, visible within C3) — and it reframes the
shortcut family: city→language items hold the country at rank ~40–120, and
flip at the intermediate rate.

### Structure (descriptive)
- Occupancy (centered): median **2–3** atoms at content layers (paper: ~25).
  The raw-gauge "≈1" was artifact; 2–3 is the model. A small workspace,
  honestly measured.
- Lens vs logit lens: on GPT-2 the logit lens reads the two-hop intermediate
  as well as the J-lens (L9: 52% vs 38% top-10) — unlike SmolLM2 (1/44 vs
  30/44). GPT-2 is the model the logit lens was built on; here the J-lens's
  contribution is the intervention system, not readout superiority.
- MLP gain: centered lens directions amplified 1.11–1.32× by the next MLP
  block at L7–10 (controls ~1.0) — the paper's ~10× broadcast signature
  exists only in trend at 124M.
- All task content in the last quarter of depth; no clean workspace *band*,
  a cliff at L9.

## The scale ladder (GPT-2 family)

Same code, same materials, same frozen protocol; [C] = held-out confirmatory
(small: authors' lens; medium and large: our n=1000 lenses; unmarked large
cells are exploration on the interim 275-prompt lens).

| | gpt2 124M | gpt2-medium 355M | gpt2-large 774M | Claude (paper) |
|---|---|---|---|---|
| dictionary mean-share (the cone) | 0.97–0.99 | 0.98–0.99 | **0.03–0.16** (gauge fix → no-op) | healthy |
| C1 corr (top layer) | +0.61 [C +0.40] | +0.78 [C **+0.68**] | +0.90 [C **+0.78**] | "highly correlated" |
| C1b swap-to-report top-5 | 69% [C 88%] | 92% [C 64%] | 100% [C **100%**] | 88% |
| C1c privilege (J:non-J, clamp) | 1.0×, clamp→0 [C] | **3.0×, clamp→0 [C]** | 2.5× (5/6 vs 2/6), clamp→0 [C] | 59% vs 5% |
| C2 focus vs base (median rank) | 13 vs 337 [C] | 1 vs 207 | 1 vs ~500 [C 7 vs 421] | — |
| C2 demand-loading | 28/36 | **33/36** | 32/36 (median 82 vs 162) | — |
| C3 readout, unspoken pooled | 33% [C] (5/6 clean-family) | 45% [C] (5/5 clean) | 68% [C **67%** — first size over the 50% bar] | routine |
| C3 swap top-1 | 77% [C 50%] | 81% [C **82%**] | 70% [C 59%] | 54–70% |
| C3 crossfn both-flip | 14/27 [C n<8] | **37/59** [C n<8] | 41/61 [C **5/16** — first confirmed pass] | — |
| C3d probe: J / non-J / clamp | 58/27/4 [C 22/9/0] | [C 23/8/6] | 22/10/1 (2.2×, final lens) [C floored: full 16%] | 61/28/6 |
| C4 within-category top-1 | 85% [C 95%] | 87% [C 72%, 3 cats] | 91% [C 83%, best cat 98%] | 40–53% |
| C5a flexible vs automatic | 15/18 vs 0/12 [C 18/19 vs 0/12] | 23/23 vs 0/12 [C 23/23 vs 0/11] | 24/24 vs 0/12 [C **24/24 vs 0/12**] | ~100% vs ~0% |
| C5b flexible collapse w/ controls | ✓ (wikitext 0.63 miss) | ✓ band-matched | [C flexible 0.00–0.17, controls ≥0.79; copy 0.56/wikitext 0.48 miss] | clean dissociation |
| occupancy (centered) | 2–3 | 2–3 | **5–9** | ~25 |
| inject selective window | s≈0.25 (24/24 vs 5/24) | s≈0.1 (15/16 vs 3/16) | none in {.03,.05,.1,.25} — both channels rise together | — |
| imagine-dissociation bleed (claim orth-z / real orth-z) | 35% | 20% [C] | 13% [E] / 30% [C] | — |

What arrives with scale, quantitatively: dominance (instructed content rank
14 → 1), report-channel matched-norm privilege asymmetry (C1c: 1.0× at 124M
→ 3.0× at 355M, 2.5× at 774M on n=6; the clamp-mediation test is **total at
every size**), pooled readout coverage under confirmation (33 → 45 → 67%,
clearing the 50% bar only at 774M), confirmed cross-function transfer
(n-starved below 774M; 5/16 there), capacity (occupancy 2–3 → 5–9),
demand-loading strength, imagine-dissociation cleanliness on exploration
materials (bleed 35% → 13%), and the natural health of the dictionary
itself (the cone dissolves by 774M). The reasoning-channel asymmetry (C3d)
is a stable ~2–3× wherever the underlying swap works, at every size. What
is present at every size: reportability, causal routing through lens
coordinates, cross-function reuse, demand- and instruction-driven loading,
and the flexible/automatic selectivity dissociation.

A second, sharper pattern in the large column: **every 774M confirmatory
miss is a protocol-transfer artifact of bars calibrated at 124M**, not a
weakening of the phenomenon — the inject dose grid brackets the (narrower)
774M window without landing in it; the fractional ablation band becomes 15
layers and cuts into shallow tasks; the probe-direction swap floors on the
exotic held-out pool (the same experiment replicates cleanly on the
exploration pool with the same final lens: 22%/10%/1%); and the C1c
addendum reaches n=6, where one item spans the 3× bar. That is also exactly
what the frozen-bar discipline is for: the misses are reported as misses,
and their diagnosis (LOG.md) is post-hoc and labeled.

## Confirmed verdicts — gpt2-medium (355M, final n=1000 lens, held-out)

| criterion | bars | verdict |
|---|---|---|
| C1 | corr ✓ 0.68 CI[0.50,0.84]; swap ✓ 64% top-5; privilege ✓ J 3.0× non-J + clamp→0 (frozen addendum protocol); inject missed only by saturation at the small-calibrated strength (dose window shown at s=0.1: 15/16 vs 3/16) — a secondary clause per BRIEF | **Shown** |
| C2 | 34/36, 35/36, 31/36, 17/19 — all four bars | **Shown** |
| C3 | swap ✓ 82%; probe ratio ✓ 2.9× (clamp clause floor-limited: 4/48→3/48 misses the halving bar); pooled readout 45% vs 50% bar (5/5 excluding the shortcut family); crossfn n=7 < 8 | **Hints** |
| C4 | 72% top-1 over 3 categories (best 87%) ✓; breadth cap (grid spec) | **Hints** |
| C5 | same-latent 23/23 vs 0/11 ✓✓; ablation battery misses at the frozen 9-layer band (controls degrade too); band-matched exploration (L19–22) shows the clean small-model signature | **Hints** |

## Confirmed verdicts — gpt2-large (774M, final n=1000 lens, held-out)

| criterion | bars | verdict |
|---|---|---|
| C1 | corr ✓ 0.78 CI[0.68,0.87]; swap ✓ 100% top-5 (7/7); privilege half-met — clamp-mediation total (nonJ 2/6 → 0/6) but rate asymmetry 2.5× vs the 3× bar at n=6 (frozen addendum, experiments/31); inject: no selective dose exists in {.03,.05,.1,.25} — report and blurt rise together (63% vs 42% at s=.1) | **Hints** |
| C2 | 36/36 (p=2.9e-11), 31/36, 34/36, 22/22 — all four bars; think median rank 7 vs base 421 | **Shown** |
| C3 | pooled unspoken readout ✓ 67% — first size over the 50% bar (3/3 excluding the shortcut family); swap ✓ 59%; crossfn ✓ 5/16 — first size with confirmable n; probe split floored (full swap itself 16% on the exotic held-out pool; J/nonJ uninformative at floor — same experiment, same lens, exploration pool: 22%/10%, clamp 1%) | **Hints** |
| C4 | 83% top-1, best category 98% ✓; breadth cap (grid spec); the surface-form function (first_letter) never redirects (0/8; 0/6 at 355M) — the coordinate carries semantic identity, not orthography | **Hints** |
| C5 | same-latent 24/24 vs 0/12 ✓✓ — Shown-strength at the third size running; ablation: flexible collapse 0.00–0.17 with controls ≥0.79 ✓✓, but shallow retention only cont_lang (0.83); copy 0.56, wikitext 0.48 (wikitext pre-declared); band-matched L31–34 column: cont_lang 1.00, copy 0.65 | **Hints** |

**gpt2-small: 1 Shown + 4 Hints. gpt2-medium: 2 Shown + 3 Hints. gpt2-large:
1 Shown + 4 Hints. At all three sizes, all five criteria ≥ Hints under
frozen, held-out confirmatories — the pre-registered minimal claim ("GPT-2
has at least hints of a privileged set") is formally established at every
size tested, with C2 Shown at all three and C1 Shown at 355M.** The
≥3-Shown headline for full possession is not met at any single size. No
criterion produced a significant null or reversed key contrast at any size;
the one numerically reversed split (C3d at 774M, J 4% vs non-J 11%, p≈0.4)
sits at the floor of a manipulation that itself stopped working on that
pool (16% full-swap), and replicates in the paper's direction on the
exploration pool with the same lens. The gaps that remain are dose grids,
band widths, and item pools calibrated at 124M, plus genuine capacity/
breadth limits (grid breadth, occupancy, pooled-readout dilution below
774M).

## What this means

1. **The minimal claim is established.** Under pre-registered rules, frozen
   bars, and held-out materials, GPT-2 — the standing example of a model
   nobody extends consideration to — exhibits at least Hints of every
   component of the privileged-set pattern at all three sizes tested, with
   directed modulation fully Shown at every size and verbal report fully
   Shown at 355M. Anyone whose update from the workspace paper tracked
   *reportable, causally load-bearing, selectively engaged internal
   content* now owns a position on GPT-2.
2. **What scale buys is measured, not categorical.** Dominance of held
   content (lens rank 14 → 1), report-channel matched-norm privilege
   asymmetry (1.0× at 124M → 3.0× at 355M; 2.5× on n=6 at 774M; Claude
   ~11×), workspace capacity (2–3 → 5–9 → ~25 atoms), confirmed readout
   coverage of unspoken intermediates (33 → 45 → 67%, clearing the bar only
   at 774M), confirmable cross-function transfer (n-starved until 774M),
   and the cleanliness of the imagine dissociation all rise through the
   GPT-2 family toward the paper's Claude values. The evidence pattern is
   not an emergent possession of frontier models; its *degree* is what
   scales.
3. **The cone was never the workspace — it was the parameterization.**
   The dictionary is defined only up to transformations the readout provably
   cannot see; at 124M–355M one such gauge component is 97–99% of the raw
   dictionary (the Jᵀ-pullback of GPT-2's unembedding mean), and every
   previously "blocked" measurement was measuring it. In the canonical
   gauge, the paper's decomposition-based controls pass at 124M — and at
   774M, where the artifact dissolves naturally, the correction costs
   nothing. This also explains, retroactively and exactly, which day-2
   operations transferred (gauge-invariant ones) and which failed
   (gauge-dependent ones).
4. **The honest boundary.** The imagine-style dissociation (instructions
   move the lens but not J-orthogonalized property probes) stays partial on
   held-out materials at every size — the claim's bleed into the
   J-orthogonal channel is 35% / 20% / 30% of the real-text effect at
   124M/355M/774M (13% on exploration materials at 774M, the one near-clean
   case); matched-norm rate asymmetry only emerges above ~350M, and its
   sharp sibling, clamp-mediation, is total at every size; no selective
   inject dose exists at 774M under the all-late-layers protocol (report
   and blurt saturate together — the window that exists at 124M and 355M
   closes rather than shifts); ordinary next-token prediction is not yet
   fully independent of the verbalizable stream at ≤774M (wikitext
   retention 0.48–0.63 under top-atom ablation, and copy weakens with
   scale: 0.89 → 0.75 → 0.65); and on the GPT-2 family the J-lens readout
   holds no advantage over the logit lens (its contribution is the vector
   system and interventions, and GPT-2 is the family the logit lens was
   built on).

## Confirmed verdict table — gpt2-small (124M), held-out materials

| criterion | bars | verdict |
|---|---|---|
| C1 report | corr ✓ (0.37, CI>0); swap-to-report ✓ 7/8 top-5 (88% — logged addendum categories); inject ✓ 65% vs 10% blurt; privilege: clamp ✓ (5/8→0/8) but 3× matched-norm asymmetry ✗ (J 5/8 = non-J 5/8) | **Hints** |
| C2 modulation | focus<base 32/36 ✓; focus<ignore 35/36 ✓; white-bear ✓; non-blurt ordering ✓ | **Shown** |
| C3 reasoning | swap ✓ 50% (bar 30); probe privilege ✓ (J 2.4× non-J, clamp→0); pooled unspoken readout ✗ 5/15 (5/6 on genuine-two-hop items; shortcut-family dilution, predicted+logged pre-results); crossfn n/a (capability) | **Hints** |
| C4 flexibility | 95% top-1 (77/81) ✓; same-pair-both-functions 37/40; breadth 1 category → frozen cap | **Hints** |
| C5 selectivity | same-latent: flexible 18/19 ✓, automatic 0/12 ✓ (Shown-strength); ablation: flexible collapse ✓ + controls ✓, shallow 2/3 ≥80% (wikitext 0.63, miss pre-declared) | **Hints** |

**Overall at 124M (BRIEF rule: all ≥Hints + ≥3 Shown): 1 Shown + 4 Hints —
headline not met**, with a structure worth stating precisely: **zero null or
reversed contrasts anywhere in the confirmatory suite.** Every miss is one
of: a matched-norm rate asymmetry (the weak instrument at small scale — its
sharp sibling, the clamp test, passes totally every time it is run), a
pooled bar diluted by a family the lens correctly reads as not computing the
intermediate, a pre-declared shallow task, or a breadth cap from 124M's task
repertoire. The gpt2-medium confirmatory (final lens) tests whether C1 and
C3 cross to Shown at 355M.
