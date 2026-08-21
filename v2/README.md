# v2 — the five workspace criteria in gpt2-small, rebuilt from the paper

Re-implementation, one criterion at a time, of the functional tests in Gurnee et al. (2026),
*Verbalizable Representations Form a Global Workspace in Language Models*, on
`openai-community/gpt2` (124M, base model), written against the paper text and Anthropic's
reference implementation + released prompt data (`ref/jacobian-lens`) only.

```
jl/                  the library
  model.py           model + lens loading, J-lens vectors, exact readout     <- read this first
  hooks.py           residual-stream edits as forward hooks
  interventions.py   swap, clamp-to-clean, pursuit
c1_report/           criterion 1: verbal report
  PROTOCOL.md        paper text <-> what we ran <-> deviations, with results
  prompts.py         the paper's categories/candidates; the prompt frame
  run.py             the whole criterion in one run -> results/{results.json, prompts.json, summary.txt}
  figure.py          -> figures/criterion1.png
c2_modulation/       criterion 2: directed modulation  (same layout)
  PROTOCOL.md        four threads: instructed hold-in-mind, math, task-demand, privileging
  prompts.py         the paper's phrasings/carriers/categories; the copy frame; imagine materials
  run.py             -> results/{results.json, prompts.json, summary.txt}
  figure.py          -> figures/criterion2.png
c3_reasoning/        criterion 3: internal reasoning
  PROTOCOL.md        paper text <-> what we ran <-> deviations, with results
  prompts.py         two-hop country families (few-shot frame) + probe cues
  swaps.py           the coordinate swap (Fig 4C), local to c3 (c1 did not need it)
  run.py             the whole criterion in one run -> results/{results.json, prompts.json, summary.txt}
  figure.py          -> figures/criterion3.png
c4_generalization/   criterion 4: flexible generalization  (same layout)
  PROTOCOL.md        paper text <-> what we ran <-> deviations, with results
  prompts.py         the paper's flexible-generalization.json (verbatim) + the 2-shot frames
  swaps.py           the subtract-and-add swap with alpha (§3.4), coordinate swap, loading; local to c4
  run.py             the whole criterion in one run -> results/{results.json, prompts.json, summary.txt}
  figure.py          -> figures/criterion4.png
```

`python c1_report/run.py && python c1_report/figure.py` (about two minutes on one GPU).
`python c2_modulation/run.py && python c2_modulation/figure.py` (about two minutes).
`python c3_reasoning/run.py && python c3_reasoning/figure.py` (a few minutes on one GPU).
`python c4_generalization/run.py && python c4_generalization/figure.py` (a few minutes on one GPU).

## Conventions (inherited from Anthropic's `jlens`; details in `jl/model.py`)

* **layer L** = residual stream at the *output* of block L (L = 0..11).  The lens is the
  released gpt2-small lens (Neuronpedia, `neuronpedia/jacobian-lens`, fit with Anthropic's code
  on wikitext-103; file in `lenses/_hf/gpt2-small/`), giving J_L for L = 0..10 with target layer 11.
* **readout** at layer L = `lm_head(ln_f(J_L h))`, identical to `JacobianLens.apply`.
* **J-lens vector** of token t at layer L: the rows of W_U J_L, as in the paper (v_t = J_Lᵀ w_t).
* **centering**: v_t ← v_t − mean_t' v_t'.  Raw GPT-2 J-lens vectors have mean pairwise cosine
  0.99 (a cone inherited from the unembedding); centered, 0.00.  Readouts are unchanged.
* **workspace band** = layers 7–9.  **ranks** are 1-indexed.  **prompts** get `<|endoftext|>`
  prepended.
