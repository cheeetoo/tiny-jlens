# Handoff: state of tiny-jlens and the GPT-2 agenda

*Written 2026-08-17 by the Claude that ran days 1–2, for the next Claude. Read
BRIEF.md (pre-registration) and LAB-LOG.md (full history) before running anything.
The user's direction: focus on GPT-2, iterate hard to instantiate the privileged-set
pattern as well as it can be instantiated, and attack the cone problem head-on.*

## What is done and solid (don't redo)

- Fitting pipeline == Anthropic's, validated (gpt2 lens r=0.993–0.9996 vs their
  released artifact, above sampling-noise ceiling). Lenses on disk: SmolLM2-135M
  (n=1000), SmolLM2-360M (n=1000), prefit gpt2-small / qwen3.5-0.8b / pythia-70m
  in `lenses/`. gemma-3-270m is gated (needs HF token).
- SmolLM2-135M confirmatory verdicts (frozen bars): C3 Shown; C1/C4/C5 Hints;
  C2 Not-shown *under the original metric* but **Shown under the revised
  rank-sensitive late-layer operationalization** (bars frozen before its own
  held-out confirmation — see CONFIRMATORY.md addendum). Both C2 verdicts reported.
- GPT-2-small already has: reasoning readout 79% / swap 83% (runs/gpt2_c3.json);
  report correlation ρ→0.59 and report-swap 70% top-5 (coordinate form, α=2 —
  runs/gpt2_c1_coord.json); instructed holding think 17 vs baseline 749, 23/23
  under the strict blurt filter (runs/c2rev_gpt2.json).
- The cone: measured, not hypothesized. At 135M the 49k lens vectors have mean
  pairwise |cos| ≈ 0.78; one PC carries ~78% of their variance; residual wiggles
  are near-orthogonal (|cos| 0.07). gpt2-small is worse (GP captures 0.16% of
  probes). Occupancy ≈ 1 by the paper's criterion. This is the identified blocker
  for: variance-privilege comparisons, occupancy/capacity, whole-space ablation
  (dose sweep k=1..10: k=1 already removes 8.1% of norm — the top atom IS the
  cone axis; no gentle dose exists; see LAB-LOG 2026-08-17 ~14:00+).

## The cone-treatment research direction (the likely "little trick")

The user reports a previous Claude found a trick that made all five properties
show clearly; the most likely candidate, now independently motivated:
**center or whiten the lens dictionary before using it for decompositions and
interventions.**

Principled justification to verify FIRST (one-liner): project the cone axis
through the unembedding — if its logit contribution is near-uniform across the
vocabulary, it carries ~no token-discriminative content and removing it from the
"concept content" definition is a correction, not a knob. Related anchors:
- The paper's own template lens whitens with (Σ+λI)⁻¹ (appendix) — same family.
- The user's LW post mean-centered residuals for Gemma-3 (massive activations).
Options to evaluate: (a) subtract the per-layer mean lens vector / top-PC of the
lens cloud from every v_t; (b) full whitening in the template-lens style;
(c) run gradient pursuit / occupancy / ablation in the wiggle-space (cone
projected out) while leaving readouts untouched (readouts are already ~invariant).
Expected payoffs if it works: meaningful occupancy, a real ablation handle
(ablate top-k wiggles, spare the shared axis — plausibly fixes C5b), restored
matched-norm privilege asymmetry (C1c/C3d), and workable decompositions on gpt2.

**Discipline (non-negotiable, the user feels strongly):** verify the uniform-logit
justification before building on it; pre-register the transformed method and its
bars in a CONFIRMATORY addendum before confirmatory runs; report every result
with-and-without the treatment; if a previous Claude's artifacts are consulted,
reconstruct claims from scratch under this repo's controls — import nothing.

## The GPT-2 iteration agenda

Escalation: gpt2-small (prefit lens) → medium/large (fit with scripts/fit_lens.py,
~1–4h each; large's d=1280 should soften the cone). Per family, current status →
what to try:
- **C1**: has corr + swap. Missing: clamp-mediation privilege (needs NO gradient
  pursuit — clamp the specific source/target token coordinates during a non-J
  perturbation; decomposition-free, cone-robust); injected-thought protocol in
  few-shot form.
- **C2**: has think/don't-think holding. Missing: the demand-vs-no-demand variant
  in few-shot form ("...then say which one you were thinking of"); per-position
  blurt analysis (span-max saturates on stronger models).
- **C3**: has readout + swap. Missing: cross-function consistency (needs a second
  function family within capability — likely unlocked at medium/large; try
  lang→continent, city→language, lang→currency); clamp-mediation as above.
- **C4**: needs a capability grid at medium/large (small has ~1 family).
- **C5a**: untried on GPT-2 — few-shot language selectivity (report-language and
  country-of-language vs continuation invariance). Continuation-in-language works
  on base models naturally; the report side needs few-shot elicitation.
- **C5b**: attempt only after a cone treatment; without it the dose sweep shows
  there is no handle (and note the paper's own Haiku caveat — coherence degrades
  before qualitative change — our result extends that trend).

## Traps already hit once (do not re-hit)

1. SmolLM2/gpt2 digit tokenization: " 5" → [" ", "5"]; grade with
   first-content-token (see first_token_id in scripts) and variant sets.
2. Swaps must be CLAMPED to swapped clean-pass values (clamped_swap_edits);
   naive re-swapping per layer oscillates (σ is an involution).
3. Projection-swap (paper §3.1 form) works on SmolLM2 but *backfires* on gpt2;
   coordinate-clamped form at α=2 works there. Per-model, test both.
4. Readout anchors: skip article tokens ("...is" → " a" → extend prefill);
   fragment/markup argmax sources have no usable lens vector (pick_source rule).
5. Task content lives LATER than generic band metrics suggest (cliff at L23 of 30
   on 135M; verify the equivalent on each gpt2 size before fixing bands) and
   top-k hit metrics cannot see 1000→100 rank shifts — always record full ranks.
6. Mention conditions are echo-contaminated (blurt 22/36) — think/don't-think
   pairing plus per-condition blurt checks is the clean contrast.
7. Ablation protection (spare clean top-10 outputs) overlaps load-bearing content
   on short prompts — quantify overlap; long prompts rebuild ablated content.
8. Capability-filter everything first, symmetrically; explore freely, then freeze
   bars + held-out items before any number becomes a claim (BRIEF R1–R4).

## Infrastructure map

- `src/tinyjlens/`: lensops (LensKit, gradient pursuit), interventions (steer /
  clamped & proj swaps / clamp / topk-ablate, KV-safe hooks), prompts, corpus,
  item pools (twohop_pool, confirm_pools).
- `scripts/`: fit_lens, band_analysis, c1_report, c2_modulation, c2_probe_control
  (the revised C2 — closest template for new GPT-2 work), c2_variants, c2c,
  c3_reasoning (phases incl. crossfn/probe), c4_flexibility, c5a, c5b, occupancy,
  lenseval_compare, run_confirmatory.sh.
- All results in `runs/*.json` with logs; every run documented in LAB-LOG.md.
- Published artifact: claude.ai/code/artifact/107c4043-0e2e-4d71-8677-61f220358e1a
  (update via the original conversation, or pass its URL as `url`).
- Draft post: docs/LW-DRAFT.md (user will rewrite; keep numbers current for them).

## Claim discipline for the eventual post

Target claim shape: "GPT-2 has hints of a workspace/privileged set." Hints, not
possession. Every number labeled with which GPT-2 size produced it. The SmolLM2
full-battery results are the load-bearing supporting evidence — keep them in the
story. The failed original headline bar and both C2 verdicts stay visible; that
honesty is what makes the positive results credible.
