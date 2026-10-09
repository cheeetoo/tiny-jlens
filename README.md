# tiny-jlens

Can the "privileged set" evidence pattern from *Verbalizable Representations Form a Global
Workspace in Language Models* (Gurnee et al. 2026) be instantiated in **gpt2-small** (124M),
using the J-lens artifact the authors released for it?

A re-implementation of the paper's five functional criteria, one module per criterion, written
against the paper text and Anthropic's reference implementation plus released prompt data
(`ref/jacobian-lens`) only.

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
  followups.py          follow-up experiments: protection rule, ignition, lists, neurons,
                        lens variants, band sensitivity, linear check, injected thought, seeds
  control.py            the two tests of top-down control (injected thought, directed modulation) as
                        the paper runs them, on GPT-2 and on Gemma 3 270m and 1b (base and instruct),
                        each Gemma model with its Neuronpedia lens; also the arithmetic and
                        paired-question parts of directed modulation
  introspect_probs.py   the injected-thought test in probabilities (GPT-2)
results/<criterion>/    results.json, prompts.json, summary.txt from each criterion's run
results/followups/      one JSON per follow-up experiment
results/control/        jl.control's results, one directory per model
ref/paper-data/         the data behind the paper's interactive figures (Claude's numbers, per trial
                        where released); its README describes each file
post/                   the post's figures: fig1/ (Figure 1), sketches/ (the structure figures) and
                        figures/; and lesswrong/ (the post's interactive embeds)
commentary/README.md    a sourced summary of the invited commentary and other reactions, with links
                        (local copies of the paper and commentary texts are gitignored)
lenses/gpt2-small/      the released lens: fit config and convergence (the .pt is not in git)
ref/jacobian-lens/      Anthropic's reference implementation (installed editable; the backend)
```

Each criterion module is self-contained: the paper's materials and the base-model prompt frame,
the experiments, and the summary.

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
summary. The follow-ups run with `python -m jl.followups [name ...]` and `python -m jl.introspect_probs`.
Other analyses:

```
python -m jl.c3_reasoning swap_ops       # E3 under both swap operations
python -m jl.c4_generalization gating    # E2 swap rates by gating class
python -m jl.c5_selectivity language     # the language test only, merged into results.json
python -m jl.control --model gpt2 introspect modulation
NCARRIERS=20 python -m jl.control --model gpt2 modulation_grid   # directed modulation on the paper's prompt, every layer kept
                                                                # (also modulation_readout, modulation_copy)
NCARRIERS=20 python -m jl.control --model gemma-1b-it modulation_grid modulation_copy clauses
                                                                # likewise gemma-270m, gemma-270m-it, gemma-1b;
                                                                # FAMILIES=math GRID_TAG=_math for the math family alone,
                                                                # LENS_FROM=<model> / LENS_CENTER=1 for the lens controls
python -m jl.band [stats|cka|fig28|mlp_gain]   # the band statistics (--model gemma-1b etc. for Gemma)
```

On a GPU, `GRID_BATCH=32` runs `modulation_grid` in batches (the default, 1, is
the per-trial loop the existing results were made with; the two agree to within a few places of rank). The figures are built from the
repository root with `python post/fig1/build.py` (needs headless Chromium) and
`PYTHONPATH=. python post/figures/<name>.py`.

Setup: `pip install -r requirements.txt` (Python 3.13; it installs `ref/jacobian-lens` editable). (A working
environment is in `.venv/`: `source .venv/bin/activate`.) The
lens file itself is a large binary and is not in git: download the authors' released gpt2-small
lens (Neuronpedia, `neuronpedia/jacobian-lens`) to
`lenses/_hf/gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt`, or point `JLENS_LENS` at
it. `DEVICE` overrides the device. Otherwise `jl.model` and `jl.control` use CUDA when there is one
(`jl.control` then MPS, then the CPU).
