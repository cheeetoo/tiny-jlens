# GPT-2 and the privileged set

*tiny-jlens, phase 2 (the GPT-2 rewrite). DRAFT — exploration numbers [E] are
final; confirmatory numbers [C] land as the frozen suite completes. Protocol:
CONFIRMED.md (frozen before any confirmatory run); search history: LOG.md;
day-1/2 SmolLM2 work: ../old/.*

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

## Headline (to be finalized after confirmatory)

GPT-2-small shows **considerably more than hints** of the privileged-set
pattern in exploration: every criterion's core contrast instantiates, four of
five with effect sizes at or above the paper's own smaller-model numbers, and
the privilege/selectivity controls — the parts that make the pattern mean
something — pass where they were previously thought blocked. The clean
boundary of what does *not* transfer: top-of-lens dominance, the
imagine-instruction dissociation (partial only), workspace capacity
(occupancy ~2–3 vs ~25), and one shallow task (ordinary next-token
prediction) that at 124M is itself partly lens-mediated.

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

## The scale ladder

[gpt2-small → medium → large table lands when the medium/large confirmatory
chains complete; day-2 ladder (SmolLM2 135M/360M, qwen 0.8B) in ../old/.]

## What this means (draft)

1. The five-family evidence pattern instantiates in GPT-2-small far beyond
   "hints" at the level of *content and causal routing*: reportable,
   causally load-bearing, flexibly reusable, selectively engaged, unspoken
   intermediates — with the privilege controls (matched-norm component
   swaps, clamps, automatic-task invariance, ablation specificity vs matched
   controls) behaving as in the paper.
2. What separates GPT-2 from Claude is now specific and quantitative:
   **capacity** (2–3 atoms vs ~25), **dominance** (rank ~14–50 vs rank 1),
   **breadth** (one function category vs sixteen), **top-down purity** (the
   imagine dissociation is partial; suppression barely distinguishable from
   attention), and **the automatic margin** (ordinary text prediction is not
   yet fully independent of the verbalizable stream).
3. The cone was never a property of GPT-2's workspace — it was a gauge
   artifact of the dictionary parameterization, and removing it is a
   correction licensed by an exact invariance of the readout, not a method
   change. With it removed, the paper's own controls run and pass at 124M.

*(Verdict table per CONFIRMED.md bars to be inserted when the confirmatory
suite completes; every bar reported as met or missed.)*
