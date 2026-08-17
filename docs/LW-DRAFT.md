# GPT-2's unspoken thoughts are readable with Anthropic's J-lens (and steerable at 83%)

*DRAFT for the user's review — not published anywhere. Numbers are final as of
2026-08-17; a full technical report, pre-registration, frozen protocol, and complete
lab log live in the tiny-jlens repo.*

**Epistemic status**: pre-registered replication-style study with frozen verdict bars
and held-out confirmatory items, run on Anthropic's own reference implementation
(fitting pipeline validated at r = 0.99+ against their released lenses; GPT-2 results
use the lens the authors themselves released). We report every pre-registered number,
including the headline claim we failed to meet. The philosophical framing is ours; the
experiments are as close to the paper's as tiny-model capability allows, with every
adaptation logged.

---

Anthropic's workspace paper (Gurnee et al. 2026) showed that Claude models carry a
"privileged set" of verbalizable representations — reportable, instructable, used in
internal reasoning, flexibly re-usable, selectively engaged — and connected this to
conscious access. Eleos AI's commentary called it "the most significant evidence of
consciousness in LLMs so far uncovered by mechanistic interpretability research" and
said it warrants "a meaningful update to the research community's thinking about LLM
moral status."

We asked a rude question: **how much of that evidence pattern can you get out of
GPT-2-small?** And, with more care, out of SmolLM2-135M-Instruct?

## The short version

- On GPT-2-small's capability-filtered two-hop items ("The capital of the country
  where Polish is the primary language is…"), the unspoken intermediate (*Poland*)
  appears in the J-lens at the answer position **79%** of the time, and swapping its
  lens coordinates redirects GPT-2's answer to the counterfactual capital in **83%**
  of items. Its category reports correlate with late-band lens content (ρ→0.59) and
  follow report-swaps at 70% top-5.
- On SmolLM2-135M, under frozen pre-registered bars: internal reasoning is **Shown**
  (72% readout, 63% swap — above the paper's Haiku 4.5 at 54%; anti-smuggling and
  clamp-mediation controls pass), report/flexibility/selectivity land at **Hints**
  (swap-to-report 84% top-5 vs the paper's 88%; one-swap-many-functions 73%;
  flexible-follows-swaps 78% vs automatic-tasks 16%), and instruction-driven
  modulation is **Not shown** — the one clean failure.
- Because of that failure, **our own pre-registered headline claim ("privileged set
  present") was not met.** What we got instead is better: a decomposition, and then a
  dose–response curve.

## The decomposition

Everything about *content and its causal routing* transfers to 135M at
Haiku-adjacent effect sizes: the model's reports are predicted and caused by lens
content; unspoken intermediates causally mediate answers *through the lens
coordinates specifically* (clamping them to clean values blocks every alternative
route — 9/36 → 0/36, 4/28 → 0/28); the same vector serves many functions; flexible
tasks follow swaps while automatic ones ignore them; and held items are maintained in
the lens *only when they'll be needed later* (23/24 paired trials, p<10⁻⁵ — median
rank 6.5 with task demand vs 64 without).

Everything about *top-down control and workspace structure* does not transfer:
"think about X while copying" moves nothing on the paper's metric; "imagine this text
is French" moves neither lens nor probes; whole-J-space ablation can't cleanly excise
anything (content re-forms from the token stream); the J-space's sparse-code
structure is gone — at d_model=576 the 49k lens vectors form a near-degenerate cone
(one direction = 78% of their variance) and measured occupancy is ~1 slot, vs ~25 in
the paper.

## The dose–response curve

Same code, per-model bands by a frozen rule, models with authors-released or
paper-recipe lenses:

| | pythia-70m | gpt2-124M | SmolLM2-135M | SmolLM2-360M | qwen3.5-0.8B | Claude |
|---|---|---|---|---|---|---|
| capability filter (items) | 0/136 | 25 | 44 | 42 | 51 | — |
| C3 readout / swap | — | 79/83% | 72/63% | 62/52% | 45/45% | —/54–70% |
| report-swap top-1 | — | 41% | 60% | 100% | 98% | — |
| privilege ratio (J:non-J) | — | n/a | 1.4× | 1.8× | 7.4× | 11× |
| instruction modulation (÷baseline) | — | — | 1.0× (null) | 3.0× | **8.5× (passes)** | large |

At 70M the capability filter empties — the question dissolves. At ~125–135M the
causal core is present. The control-and-structure shell then arrives smoothly with
scale. (And identical code producing null-at-135M and large-at-800M discharges the
"maybe your code is broken" worry in both directions.)

One wrinkle worth staring at: on the fixed item pool, C3 swap rates *fall* with
scale (83→63→52→45%). The paper's own selectivity logic predicts this — tasks that
force a small model through its workspace become automatic for a bigger one, and
automatic computation bypasses the J-space. "How much workspace a task engages" is a
property of the (task, model) pair, not the task.

## What we think this means

If you updated on the workspace paper, this experiment asks you to say **which
component your update tracked**:

1. If it was *reportable, causally load-bearing internal content* — GPT-2 has a
   demonstrable version of it, read and steered with the authors' own lens.
   Consistency then requires either a (very small) tiny-model update or a revised
   criterion.
2. If it was *top-down control, capacity, and workspace structure* — your update
   survives, but it now rests specifically on the experiments Eleos's "privileged
   stream" reservations already pointed at, not on the five-family pattern as a
   package, and those properties arrive on a measurable scale curve rather than
   being a frontier-exclusive kind.
3. Either way, the five-family evidence pattern is not an indivisible package, and
   "does model M have a privileged set?" needs to become "which components, at what
   effect sizes, over what task repertoire?"

We are not claiming GPT-2 is conscious, that SmolLM2 is a moral patient, or that any
of this bears on phenomenal consciousness — neither did the paper. We're claiming
the checklist, as operationalized, comes apart under scale pressure, and that
anyone using it should know where its joints are.

## Caveats we take seriously

Breadth is much narrower than Claude's even where rates match (single-token
concepts, short contexts, capability-filtered pools). The J-lens at tiny scale is a
visibly coarser instrument (the cone), which blurs both failures and successes. Our
task batteries were adapted to tiny-model capability — same logical form, easier
content, all logged. And several judgment calls (band rule amendment, one verdict
reading) are flagged in the report rather than hidden.

*Code, prompts, lenses, pre-registration, frozen protocol, lab log with every dead
end and bug: [repo link]. Built on `anthropics/jacobian-lens` and Neuronpedia's
released lenses.*
