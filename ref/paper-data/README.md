# The paper's released figure data

Downloaded from the paper's page, where each interactive figure loads its data from
`https://transformer-circuits.pub/2026/workspace/data/<figure>/<file>.json`.

- `flex-gen-appendix.json` — `data/flex-gen-systematic/appendix.json`, the per-trial grids behind
  Fig. 68 (flexible generalization, Claude Sonnet 4.5): for each category and function, a 4x4 grid
  whose rows are the argument in the prompt and whose columns are the argument swapped in; each
  off-diagonal cell has the top-1 token after the swap (`t`) and the target answer's rank (`rk`).
  Used by `jl.c4_generalization` to score Claude on the same swaps as GPT-2.
- `flex-gen-systematic.json` — `data/flex-gen-systematic/systematic.json`, the data behind Fig. 19
  (flexible generalization, Claude Sonnet 4.5): per function, the top-1 counts out of 12 swaps at
  α=1 and α=2 (`hit1`, `hit2`), the clean workspace loading (`csum`: cos at the argument + cos at
  the readout, with its std over arguments), and the swap effect at α=1 (`dlpc`: Δ log-prob of the
  target answer minus Δ log-prob of the spontaneous answer, with its s.e.m.); `r` is their Pearson
  correlation. Used by `jl.c4_generalization` for Claude's loading-effect correlation.
- `modulation-lines.json` — `data/modulation-lines/lines.json`, the data behind Fig. 10 (directed
  modulation): for Haiku 4.5, Sonnet 4.5, and Opus 4.5, the hit rate (a tracked token at J-lens
  top 1 at any layer and position) under "think about X" and "ignore X", per task family.
- `modulation-readout.json` — `data/modulation-readout/modulation.json`, the data behind Fig. 9
  (directed modulation, Sonnet 4.5): three examples, each with the full prompt (`display_prompt`),
  its tokens, and the top 9 J-lens tokens at six layers (13, 15, 17, 19, 21, 24 of 0 to 24) at every
  token. **This is where the paper's prompt template for directed modulation comes from**
  (`Write "{sentence}" {instruction} Don't write anything else.`); the released protocol has none.
  Used by `jl.control` (frames `paper` and `human`) and `post/figures/dm_readout.py`.
- `modulation-prompts.json` — `data/modulation-prompts/data.json`, the data behind Fig. 65: for each
  task family and model, the hit rate of every instruction phrasing (`dots`, with its template `t`),
  the mean of each group (focus, mention, negated-think, dismissal), and the rate with no
  instruction (`none`). The rates are multiples of 1/440 (categories) and 1/480 (math), so every
  phrasing is run on all 20 carrier sentences. The math family has two focus phrasings that the
  released protocol leaves out. Used by `post/figures/dm_rates.py`.
- `top-down-summoning.json` and `top-down-summoning-appendix.json` — `data/top-down-summoning/main.json`
  and `appendix.json`, the data behind Figs. 11 and 66 (the paired-question test, Sonnet 4.5): for
  each passage and question, the passage's tokens with the label's best lens rank where it is in
  the top 10 (`r`), and the number of such tokens (`nHit` of `nTot`). The passage ends with a
  closing quote, so the paper's prompt puts it in quotation marks.
- `metacog-alarm.json` — `data/metacog-alarm/data.json`, the data behind Fig. 46 (think about or
  don't think about a named concept, base model against post-trained model, 40 concepts): the share
  of trials with the concept's word, a "fail" word, or "damn" in the lens top 5 over layers 38 to 92.
  Its example has the same prompt template as Fig. 9.
- `dual-task-simple.json` — `data/dual-task-simple/data.json`, the data behind the two-tasks-at-once
  figure in the paper's appendix on competition.
- `modulation-probe.json` — `data/modulation-probe/data.json`, the data behind Fig. 67 (the
  "imagine" privilege check for directed modulation, Sonnet 4.5, band L38–88).
- `verbal-report.json` — `data/verbal-report/data.json`, the data behind Fig. 6 (verbal report,
  Sonnet 4.5): the Sonnet example, and per category the Spearman correlation between lens and
  output scores of the 10 candidates at layers 54, 75, and 92 (`quant.rhos`), plus the swap pairs.
- `verbal-introspection.json` — `data/verbal-introspection/data.json`, the data behind Fig. 7 (the
  injected thought, Sonnet 4.5, 100 concepts, prefill ending `about the word "`): the example
  readouts for "lightning", and `curve`, the median and quartiles (`q25`, `q50`, `q75`) of the
  injected concept's reciprocal rank at the open quote (`slot`) and at the other positions of the
  assistant turn (`other`), at strengths `s` from 0 to 0.025. Values are rounded to 3 decimals.
  Used by `post/figures/introspect.py`. (Not to be confused with
  `ref/jacobian-lens/data/experiments/verbal-introspection.json`, the released protocol.)
