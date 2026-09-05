# tiny-jlens

Can the "privileged set" evidence pattern from *Verbalizable Representations Form a Global
Workspace in Language Models* (Gurnee et al. 2026) be instantiated in **gpt2-small** (124M),
using the J-lens artifact the authors released for it?

A re-implementation of the paper's five functional criteria, one module per criterion, written
against the paper text and Anthropic's reference implementation plus released prompt data
(`ref/jacobian-lens`) only. `PROTOCOL.md` is the record: for each criterion, the paper's text,
what we ran, what we found, and every deviation the base model forced.

## Layout

```
jl/                     the library
  model.py              model + lens loading, J-lens vectors, exact readout   <- read this first
  hooks.py              residual-stream edits as forward hooks
  interventions.py      swap, coordinate swap, clamp-to-clean, pursuit, loading, J-space ablation
  stats.py              Wilson intervals, sign test, medians, Spearman
  c1_report.py          criterion 1: verbal report
  c2_modulation.py      criterion 2: directed modulation
  c3_reasoning.py       criterion 3: internal reasoning
  c4_generalization.py  criterion 4: flexible generalization
  c5_selectivity.py     criterion 5: selectivity
  band.py               the structural statistics behind the choice of band
PROTOCOL.md             paper text <-> what we ran <-> deviations, with the results
results/<criterion>/    results.json, prompts.json, summary.txt from the runs behind PROTOCOL.md
lenses/gpt2-small/      the released lens: fit config and convergence (the .pt is not in git)
ref/jacobian-lens/      Anthropic's reference implementation (installed editable; the backend)
```

Each criterion module is self-contained: the paper's materials and the base-model prompt frame,
the experiments, and the summary that `PROTOCOL.md` quotes.

## Running

From the repository root, one command per criterion:

```
python -m jl.c1_report          # about two minutes on one GPU
python -m jl.c2_modulation      # about two minutes
python -m jl.c3_reasoning       # a few minutes
python -m jl.c4_generalization  # a few minutes
python -m jl.c5_selectivity     # a few minutes
```

Each writes `results/<criterion>/{results.json, prompts.json, summary.txt}` and prints the
summary. Two secondary analyses quoted in `PROTOCOL.md` are subcommands of their criterion:

```
python -m jl.c3_reasoning swap_ops       # E3 under both swap operations
python -m jl.c4_generalization gating    # E2 swap rates by gating class
python -m jl.band [stats|cka|mlp_gain]   # the band statistics (CPU is fine)
```

Setup: `pip install -e ref/jacobian-lens`, plus torch, transformers, datasets and scipy. The
lens file itself is a large binary and is not in git: download the authors' released gpt2-small
lens (Neuronpedia, `neuronpedia/jacobian-lens`) to
`lenses/_hf/gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt`, or point `JLENS_LENS` at
it. `DEVICE` overrides the device, which otherwise is the GPU when there is one.

## Conventions

Inherited from Anthropic's `jlens`; the details are in `jl/model.py`.

* **layer L** = residual stream at the *output* of block L (L = 0..11). The lens is the released
  gpt2-small lens (fit with Anthropic's code on wikitext-103; see `lenses/gpt2-small/config.yaml`),
  giving J_L for L = 0..10 with target layer 11.
* **readout** at layer L = `lm_head(ln_f(J_L h))`, identical to `JacobianLens.apply`.
* **J-lens vector** of token t at layer L: the rows of W_U J_L, as in the paper (v_t = J_Lᵀ w_t).
* **centering**: v_t ← v_t − mean_t' v_t'. Raw GPT-2 J-lens vectors have mean pairwise cosine
  0.99 (a cone inherited from the unembedding); centered, 0.00. Readouts are unchanged.
* **workspace band** = layers 7–9. Lens top-1 autocorrelation rises above the shuffled null from
  layer 6, and top-1 agreement with the output stays below 15% through layer 9; the paper's other
  band signatures (a kurtosis rise, CKA blocks) do not appear in gpt2-small. The evidence, and
  how arguable the choice is, are in the appendix of `PROTOCOL.md`.
* **ranks** are 1-indexed. **prompts** get `<|endoftext|>` prepended.

## The cone is gauge (read this before the geometry)

Every geometric operation in the code runs on the **centered** dictionary. This is not a tuning
knob; it is a canonical-gauge choice:

- The exact lens readout is `logit_t(h) = ⟨v_t, h⟩/σ(Jh) + β_t`. Replace every dictionary vector
  by `v_t + u` (any fixed `u`): every logit shifts by the same per-position constant
  `⟨u, h⟩/σ(Jh)`, and softmax kills shared shifts. **No readout the lens produces can distinguish
  the dictionary from any of its translates**, so raw-dictionary geometry (cosines, spans,
  projections, pseudoinverse coordinates) is not a property of the lens until a gauge is fixed.
  The invariance group is translations only — a shared rescale `a·v_t` is *not* softmax-invisible
  (β_t and σ do not transform with it) — and centering `u = −v̄` is the canonical translate: the
  minimum-total-norm representative of the gauge class.
- At GPT-2 scale one gauge component dominates: the vocabulary-mean vector `v̄ = Jᵀū` is 97–99%
  of the dictionary's second moment (cos = 1.000 with the top principal axis at every layer), and
  its logit profile is constant across the vocabulary to ~2% — softmax-invisible, pure gauge.
  This is the previously-reported "cone".
- In the centered gauge the dictionary is healthy (mean |cos| 0.07–0.10) and the operations the
  cone blocked come alive: gradient pursuit recovers a real share of activation variance, top-k
  ablation acquires a dose axis, and the paper's §3.1 projection swap works on GPT-2 two-hops.
- Interventions along *differences* of lens vectors (the coordinate swap moves along `v_s − v_t`,
  where `v̄` cancels) point along a gauge-invariant direction already — which is why swaps always
  worked on GPT-2 while decompositions failed. (The pseudoinverse coordinate *read* still changes
  with the gauge, so raw and centered swaps agree closely but not identically.)
