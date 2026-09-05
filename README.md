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
results/<criterion>/    results.json, prompts.json, summary.txt from the runs behind PROTOCOL.md
lenses/gpt2-small/      the released lens: fit config and convergence (the .pt is not in git)
ref/jacobian-lens/      Anthropic's reference implementation (installed editable; the backend)
```

Each criterion module is self-contained: the paper's materials and the base-model prompt frame,
the experiments, and the summary that `PROTOCOL.md` quotes.

## Running

From the repository root, one command per criterion:

```
python -m jl.c1_report
python -m jl.c2_modulation
python -m jl.c3_reasoning
python -m jl.c4_generalization
python -m jl.c5_selectivity
```

Each writes `results/<criterion>/{results.json, prompts.json, summary.txt}` and prints the
summary. Two secondary analyses can be run with:

```
python -m jl.c3_reasoning swap_ops       # E3 under both swap operations
python -m jl.c4_generalization gating    # E2 swap rates by gating class
python -m jl.band [stats|cka|mlp_gain]   # the band statistics
```

Setup: `pip install -e ref/jacobian-lens`, plus torch, transformers, datasets and scipy. The
lens file itself is a large binary and is not in git: download the authors' released gpt2-small
lens (Neuronpedia, `neuronpedia/jacobian-lens`) to
`lenses/_hf/gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt`, or point `JLENS_LENS` at
it. `DEVICE` overrides the device, which otherwise is the GPU when there is one.
