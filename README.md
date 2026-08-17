# tiny-jlens

Can the "privileged set" evidence pattern from *Verbalizable Representations Form a
Global Workspace in Language Models* (Gurnee et al. 2026) be instantiated in tiny
models — SmolLM2-135M, GPT-2?

- **docs/REPORT.md** — the main report (verdicts, ladder, what transfers and what doesn't)
- **docs/BRIEF.md** — pre-registration (criteria, verdict bars, concession rules; frozen day 1)
- **docs/CONFIRMATORY.md** — frozen confirmatory protocol
- **docs/LAB-LOG.md** — complete audit trail: every run, bug, amendment, and negative result
- **docs/LW-DRAFT.md** — draft external write-up
- **src/tinyjlens/** — corpus, lens ops, interventions, prompt/pool utilities
- **scripts/** — fitting, band analysis, the C1–C5 experiment batteries, confirmatory runner
- **runs/** — all result JSONs and logs (lens .pt files gitignored; refit with
  `scripts/fit_lens.py` or download prefit lenses from `neuronpedia/jacobian-lens`)
- **paper.md / eleos-commentary.txt** — source materials

Setup: `pip install -e <clone of github.com/anthropics/jacobian-lens>` plus
transformers/datasets/scipy; see scripts for usage.
