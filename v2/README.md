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
```

`python c1_report/run.py && python c1_report/figure.py` (about two minutes on one GPU).
`python c2_modulation/run.py && python c2_modulation/figure.py` (about two minutes).

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
