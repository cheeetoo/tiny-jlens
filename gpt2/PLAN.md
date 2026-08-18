# gpt2/: the clean rewrite

*Started 2026-08-17 (day 3). Everything from days 1–2 is in `old/` (kept as the
audit trail; its BRIEF.md §1–4 pre-registration still governs what the criteria
mean and what counts as passing). This directory is a fresh, minimal
implementation focused on one question:*

**Does GPT-2 have (at least hints of) a privileged set of verbalizable
representations, in the sense of Gurnee et al. (2026) / the Eleos commentary?**

Models: gpt2 (124M, primary; using the lens the paper's authors released),
gpt2-medium (355M) and gpt2-large (774M) as escalation (lenses fitted here with
the authors' pipeline, identical recipe — fits running in `../lenses/`).

## Rules carried over (short form)

- The five criteria C1–C5 and what "Shown / Hints / Not shown" mean are frozen
  in `old/docs/BRIEF.md`. Controls are not optional. Explore freely, then
  freeze bars + held-out items before any number becomes a headline
  (`CONFIRMED.md` will hold the frozen protocol when we get there).
- Capability filter first, symmetrically, before any lens measurement.
- Every adaptation for a base model (few-shot forms etc.) is documented.
- Report the search: failed variants land in `LOG.md`.

## Layer policy (the "don't discount late layers" lesson)

No pre-committed workspace band. Every readout experiment measures **all
fitted layers (L0–L10 on gpt2-small)** and reports the per-layer profile; task
content at tiny scale lives in the last few layers, partly inside what
band-metrics call "motor". For interventions we still exclude the single final
fitted layer only when the point of the experiment requires the effect to be
non-trivial (a swap applied where J ≈ readout is a direct logit edit); each
intervention experiment reports a layer-window analysis instead of assuming a
band.

## The cone, and the gauge fix (work stream 2 — see results/CONE.md)

Finding (results/cone_gpt2.json): at GPT-2 scale the raw J-lens dictionary is
97–99% one shared vector — exactly the vocabulary mean v̄ = Jᵀū — whose logit
profile across the vocabulary is constant to within ~2%. Because the softmax
readout is *exactly* invariant to adding any fixed vector to every dictionary
row (and to positive rescaling), the dictionary is only defined up to that
gauge; v̄ is pure gauge, invisible to every readout the lens ever produces.

Decision (method, fixed now, before any new experiment): **all geometric
operations — sparse decomposition, occupancy, top-k ablation, matched-norm
J/non-J splits, pseudoinverse coordinate reads — use the centered dictionary
(v_t − v̄), the canonical gauge representative.** Readouts are unchanged
(provable invariance, verified numerically). Interventions are run in both
flavors during exploration; the confirmatory protocol will fix one per
experiment before freezing, and the report shows raw-dictionary results
alongside (appendix) wherever they differ.

This is presented as a correction of gauge-dependence, not a new method: any
quantity that changes under a transformation that provably cannot change any
lens readout was not a property of the lens to begin with. At Claude scale
v̄ is small and the distinction is negligible; at d=768 with 50k tokens it is
97–99% of the dictionary.

## Experiments (files in experiments/, numbered)

- 00 validate — exact match vs reference `apply()`; vector identity; gauge
  invariance; smoke swap. DONE, all pass.
- 10 cone — diagnosis + justification. DONE (see results/CONE.md).
- 20 capability — few-shot/completion task survey on gpt2 / medium / large;
  produces the capability-filtered pools everything else uses.
- 30 report (C1) — category-report correlation + report swap + injected
  thought (base-model form).
- 40 modulation (C2) — think/don't-think rank protocol (LW-study form);
  demand-vs-no-demand maintenance.
- 50 reasoning (C3) — two-hop readout + swap + cross-function anti-smuggling
  + clamp-mediation privilege (decomposition-free) + probe-split privilege
  (centered decomposition).
- 60 flexibility (C4) — one swap, many functions (grid breadth needs
  medium/large).
- 70 selectivity (C5) — same-latent flexible-vs-automatic design; top-k
  ablation battery with the centered dictionary (the treated C5b).
- 80 structure — occupancy (centered), per-layer content geography, capacity.

## Endpoint

A short report (REPORT.md) with the per-criterion table for GPT-2 (each number
labeled by model size), the cone/gauge section, and the honest boundary:
what shows, what only hints, what does not appear. Old SmolLM2 results remain
as supporting evidence in old/.

**Status (2026-08-18): REACHED.** All three confirmatory suites complete
(124M / 355M / 774M, held-out, frozen bars). Verdicts: all five criteria
≥ Hints at every size; C2 Shown at all three; C1 Shown at 355M; ≥3-Shown
unmet everywhere. REPORT.md is final; LOG.md carries the full search
history including every post-hoc diagnostic.
