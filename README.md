# tiny-jlens

Can the "privileged set" evidence pattern from *Verbalizable Representations
Form a Global Workspace in Language Models* (Gurnee et al. 2026) be
instantiated in **gpt2-small** (124M), using the J-lens artifact the authors
released for it?

## Layout

- `gpt2/core.py` — model + lens loading, exact lens readouts, J-lens
  vectors, residual-stream interventions (validated to 0.0 against the
  reference `apply()`; see the module docstring for all conventions).
- `gpt2/pools.py` — task materials (country facts, two-hop families, report
  categories, C2/C5 items; single-token constraints enforced at build time).
- `gpt2/experiments/` — numbered, one file per experiment; each has a
  docstring saying what it measures. `python gpt2/experiments/NN_name.py
  [phase] [model]`, run from `gpt2/`.
- `gpt2/results/` — result JSONs from the runs behind the write-up.
- `paper.md`, `eleos-commentary.txt` — source materials.
- `ref/jacobian-lens` — Anthropic's reference implementation (installed
  editable; fitting + readout backend).
- `lenses/gpt2-small/` — the authors' released GPT-2 lens.

Setup: `pip install -e ref/jacobian-lens` plus
torch/transformers/datasets/scipy/langid.

## The cone is gauge (read this before the geometry)

Every geometric operation in the code runs on the **centered** dictionary
(`centered=True`). This is not a tuning knob; it is a canonical-gauge choice:

- The exact lens readout is `logit_t(h) = ⟨v_t, h⟩/σ(Jh) + β_t`. Replace
  every dictionary vector by `v_t + u` (any fixed `u`): every logit shifts by
  the same per-position constant `⟨u, h⟩/σ(Jh)`, and softmax kills shared
  shifts. **No readout the lens produces can distinguish the dictionary from
  any of its translates**, so raw-dictionary geometry (cosines, spans,
  projections, pseudoinverse coordinates) is not a property of the lens until
  a gauge is fixed. The invariance group is translations only — a shared
  rescale `a·v_t` is *not* softmax-invisible (β_t and σ do not transform
  with it), and centering `u = −v̄` is the canonical translate: the
  minimum-total-norm representative of the gauge class.
- At GPT-2 scale one gauge component dominates: the vocabulary-mean vector
  `v̄ = Jᵀū` is 97–99% of the dictionary's second moment (cos = 1.000 with
  the top principal axis at every layer), and its logit profile is constant
  across the vocabulary to ~2% — softmax-invisible, pure gauge. This is the
  previously-reported "cone".
- In the centered gauge (mean removed) the dictionary is healthy (mean
  |cos| 0.07–0.10) and the blocked operations come alive: gradient pursuit
  recovers 3–16% of activation variance (raw: 0.0–0.6%), top-k ablation
  acquires a dose axis, and the paper's §3.1 projection swap goes from 0%
  (raw) to 46–69% (centered) on GPT-2 two-hops.
- Interventions along *differences* of lens vectors (the coordinate swap
  moves along `v_s − v_t`, where `v̄` cancels) point along a gauge-invariant
  direction already — which is why swaps always worked on GPT-2 while
  decompositions failed. (The pinv coordinate *read* still changes with the
  gauge, so raw and centered swaps agree closely but not identically.)

`experiments/10_cone.py` reproduces the diagnosis; readout invariance to
centering is verified numerically there and in `00_validate.py`.

## Materials and prompts

The paper does not publish base-model prompts, so every prompt format here
was developed on gpt2-small directly: formats were iterated against the
measured effect on a development set of items, with the metrics, grading,
capability gates, and intervention machinery frozen throughout — only
prompt text ever moved. The pools below were then extended with items the
format search never used, and the reported numbers come from the full
extended pools. Every pool passes once through `20_capability` before any
lens is consulted.

Scale: two-hop funnel ≈470 rows over 68 countries (the lang_capital and
city_language templates carry in-context shots; a build-time guard excludes
any item whose country or answer appears in a template's fixed text), 27
report categories × 4 few-shot formats (capability graded per
format×category, with a per-format shot-collision guard), 48 C2 words × 14
sentences × 5+5+1 attend/suppress/base phrasings, 52 C5a passages, 80
inject concepts × 6 report frames + 6 noun-expecting controls, 30 imagine
sentence pairs × 4 claim headers plus a second property category (past
tense), 3 demand shot-sets, and the C5b automatic-task suite split by
answer-token class. `analyze.py` prints the battery with Wilson CIs, exact
sign tests, cluster bootstraps (trials sharing a category/country/word are
clustered), and per-frame spreads; `results/analysis_summary.txt` is the
latest run. C3d additionally reports a 2×2 protocol grid (legacy
country-naming cues k=16 vs paper-faithful attribute cues k=25; norm-matched
vs natural-magnitude component swaps).

## Experiments

| file | measures |
|---|---|
| 00_validate | exact-readout + intervention semantics vs the reference implementation |
| 10_cone | the gauge diagnosis above |
| 20_capability | pre-lens capability filter for every task pool |
| 30_report | C1: report↔lens correlation; swap-to-report; injected "thought" |
| 31_c1c | C1 privilege: matched-norm J vs non-J probe-component swaps on the report |
| 40_modulation | C2: think-about-X / don't-think / base, rank-sensitive |
| 41_c2_privilege | C2 privilege, word form (negative result by design: mention contaminates) |
| 42_imagine | C2 privilege, property form (claim vs real stimulus, French + past tense, J-orth probe) |
| 43_demand | C2: demand-loading (hold a word during copying if it will be asked for) |
| 50_reasoning | C3: two-hop readout / intermediate swap / cross-function / probe split |
| 60_flexibility | C4: one argument swap redirects multiple functions |
| 70_selectivity | C5a: same-latent flexible-vs-automatic dissociation |
| 71_ablation | C5b: top-k centered ablation with protection + matched controls; retention split by answer-token class |
| 80–82 | structure: occupancy/band; lens vs logit lens; MLP gain |
