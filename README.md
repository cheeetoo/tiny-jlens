# tiny-jlens

Can the "privileged set" evidence pattern from *Verbalizable Representations
Form a Global Workspace in Language Models* (Gurnee et al. 2026) be
instantiated in tiny models? Phase 1 (days 1–2): SmolLM2-135M + a scale
ladder. Phase 2 (current): a clean rewrite focused on **GPT-2**, including
the cone/gauge analysis of the J-lens dictionary at small scale.

- **gpt2/** — the current work. Start with `gpt2/PLAN.md`, then
  `gpt2/REPORT.md` (results), `gpt2/results/CONE.md` (the cone-is-gauge
  finding), `gpt2/CONFIRMED.md` (frozen confirmatory protocol),
  `gpt2/LOG.md` (every run and dead end).
  - `core.py` — model+lens loading, exact readouts, lens vectors,
    interventions (validated to 0.0 against the reference `apply()`)
  - `pools.py` / `pools_confirm.py` — task materials (exploration / held-out)
  - `experiments/` — numbered, one file per experiment
  - `results/` — JSONs + run logs
- **old/** — phase 1, kept verbatim as the audit trail (pre-registration
  BRIEF.md, confirmatory protocol, lab log, report, code, runs).
- **paper.md / eleos-commentary.txt** — source materials.
- **ref/jacobian-lens** — Anthropic's reference implementation (installed
  editable; fitting + readout backend).
- **lenses/** — prefit lenses (gpt2-small is the authors' release; medium
  and large are fitted here with the same recipe; .pt files gitignored).

Setup: `pip install -e ref/jacobian-lens` plus torch/transformers/datasets/
scipy/langid. Every experiment is `python gpt2/experiments/NN_name.py
[phase] [model]` from `gpt2/`.
