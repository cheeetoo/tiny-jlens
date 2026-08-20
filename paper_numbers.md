# Paper (Gurnee et al. 2026) — every Claude number usable opposite a gpt2-small measurement

Source: /tiny-jlens/paper.md (line refs). Default model **Sonnet 4.5**; key results corroborated on Haiku 4.5 / Opus 4.5; some analyses Opus 4.6 (line 378). Layers reported as 25 evenly-spaced, reindexed 0–100 as percentages (378). Workspace band ≈ L38–92 (647); motor = final few layers.

## C1 — Verbal report

| Metric | Claude number | Conditions | Line | Commensurability note |
|---|---|---|---|---|
| Report↔lens Spearman | **No number in text** — "typically highly correlated … increases towards the end of the workspace" | 10 candidate answers × 14 categories, colon position immediately before the name, three workspace layers (Fig 6) | 390–396 | gpt2's per-layer Spearman is the same quantity; paper value is figure-only. Compare shape (rising to top of workspace), not value. |
| Report swap success | **88% top-5** | "think of a {category}" swap; target chosen at random within category, **excluding targets already in model's output top-10**; swap at all token positions; success = target in top-5 outputs. α not stated (α=1 natural reading; the coordinate-patch definition allows optional α, 364) | 396, 412 | gpt2 must quote α=1 (or label α). Same top-10 exclusion, same top-5 metric. |
| Concept-vector J-share | median **6–7%** of concept vector's variance (k=16 GP) | concept vector = activation before response to "Tell me about {concept}", minus mean over **100 other concepts** | 406–410 | gpt2 C1c probes must match construction & k to compare share. |
| Privilege split (report channel) | J-component **59%** vs non-J **5%** top-5 (J "approaching the 88%" of pure lens vectors) | components substituted for lens vectors, **every perturbation rescaled to the same magnitude** (matched-norm IS the paper's protocol here, unlike C3d); k=16 | 412 | gpt2 C1c matched-norm is protocol-correct for the report channel. |
| Privilege clamp | non-J effect falls **to zero** (swap) / "nearly to zero" (injection) | J-coordinates clamped to clean-pass values at every position and layer | 422 | gpt2 clamp→0 is directly comparable. |
| Injected thought | "**majority of trials**"; median reciprocal rank vs strength over **n=100 concepts** (Fig 7, figure-only). Baseline report rate **0.54** appears in Fig 34b (broadcast-head ablation experiment) | single J-lens vector injected across user turn; report elicited by introspection prompt; position control = every other assistant-turn position | 398–402, 901 | gpt2's base-model port (no user/assistant turns) is an analog, not a replication; no clean rate to compare except 0.54-at-best-strength. |

## C2 — Directed modulation

| Metric | Claude number | Conditions | Line | Note |
|---|---|---|---|---|
| Focus/ignore/baseline rates | **No numbers in text** (Fig 10/65 figure-only). Baseline ≈ 0; focus "substantial fraction", increases with model size; ignore < focus but > 0 (white-bear); **mention ≈ focus** (mention primes "almost as strongly"); **don't-think ≈ mention** (suppresses less than ignore); **ignore < mention on every model** | metric = fraction of trials target reaches **J-lens top-1 at any (layer, position)** while copying; 5–8 phrasings per condition, averaged; 3 task families (category instance, math expression, line width) | 440–444, 1832–1838 | gpt2's median-best-rank metric is different; to compare, also compute paper's top-1-at-any-(layer,position) rate. The ordering (focus > mention ≈ dont > ignore > base≈0) IS comparable. |
| Line-width task | precision 14–56% across focus phrasings (Opus 4.5) | top-1 numeric readouts matching true width | 1836–1840 | no gpt2 analog implemented. |
| Implicit demand (paired question) | label appears at **3 stimulus positions** (example); systematic version figure-only | same stimulus under next-word vs name-the-property question; count of positions with label in lens top tokens | 448–452, 1842–1848 | repo's 43_demand names the word — different design, no paper analog. The paper analog would be a property-naming paired-question port. |
| Imagine/property privilege | instruction moves lens "by **several standard deviations**"; J-orthogonalized probe stays "**within roughly one SD**"; real stimulus moves probe **3–6 SD** | 4 categories (Python/tense/POS/caps), n=24 stimuli each; z-score against no-instruction baseline, mean over stimulus positions and workspace layers, ±1 SEM over stimuli; probe = mean-difference real pos vs neg, J-component (k=25 GP) projected out | 1850–1858 | gpt2 must use z-in-baseline-SD units (not t) and k=25 to compare. Also reverse dissociation exists: real past-tense drives probe +6 SD while "past" stays far down lens. |

## C3 — Internal reasoning

| Metric | Claude number | Conditions | Line | Note |
|---|---|---|---|---|
| Intermediate readout | No pooled % in main text ("we first confirm that the intermediate appears…"). Quantitative comparison: J-lens pass@k AUC best of 3 lenses on all 6 distributions; margin over logit lens "**modest on multihop and association**" | 50 two-hop prompts (e.g. spider/legs); intermediate at position immediately preceding answer; multihop items built so trigger+answer rarely co-occur without intermediate | 460–476, 1328, 1335–1339 | gpt2 C3a's % has no direct Claude number; the honest comparison is J-vs-logit margin (modest on multihop for Claude too). |
| Intermediate swap | **Haiku 4.5: 54%, Sonnet 4.5: 70%, Opus 4.5: 70%** top-1 | 50 two-hop prompts, swap target random within same category, clamped coordinate swap at every position (α=1 per Fig 54 convention); success = target-appropriate answer at top-1 | 476, 1345–1349 | gpt2 must quote α=1, all positions. Layer band unspecified by paper. |
| Anti-smuggling | intermediate swap takes effect a median **~17% (of depth) earlier** than answer swap | swap at different layer ranges; compare effect depth | 480 | not implemented in repo (repo's crossfn is a different, repo-invented control). |
| Probe privilege split | full-probe? no — **raw J-lens token swap 60%**, J-component **61%**, non-J **28%**, non-J with J-clamp **6%** | n=90 two-hop prompts; probe = mean activation over prompts implying same intermediate via different surface cues & different questions, minus mean over all intermediates; k=25 GP; **natural magnitudes** (no norm-matching stated); clamp set = J-component tokens + naming token | 484–496 | gpt2's "full" column is a full-probe swap — paper's first column is the raw token swap; align columns. k=25, natural magnitudes for protocol match. |
| Probe J-share | "typically roughly **10–15%** of the probe's variance" | k=25 GP on the intermediate probes | 488 | direct comparison to gpt2 probe J-share (40–48% legacy naming cues → cue-construction matters). |
| Multi-step | 21→42→49 surface in computation order, separating ~L71 | Opus 4.5, single example | 498 | qualitative. |

## C4 — Flexible generalization

| Metric | Claude number | Conditions | Line | Note |
|---|---|---|---|---|
| Overall swap | **76/192 = 40%** (α=1); **101/192 = 53%** (α=2) | 4 categories (countries, months, animals, numbers) × 4 functions × 12 ordered pairs; identical swap regardless of prompt; clamped at every position; success = target answer top-1 | 510 | gpt2 α=1 comparison; note paper has NO top-10 pre-exclusion here. |
| Countries cell | **42/48 off-diagonal reach rank 1** ("almost perfectly") | α=1 | 1864–1866 | gpt2's countries-only number sits opposite 42/48 = 88%. |
| Months / animals / numbers | months "succeed **partially**", animals "**rarely**", numbers **0/48** at α=1 (failures typically move target up ranking; α=2 often rescues) | same | 1864 | gpt2's one-category breadth: honest statement is countries-vs-countries; only numbers are zero for Claude. |
| Loading→success | "Workspace loading of the source argument predicts swap success well" (Fig 19 scatter, figure-only) | loading = cosine(residual, lens vector) averaged over argument+readout positions | 512–514 | gpt2's max-over-positions loading differs; and gpt2 finds no within-category prediction. |

## C5 — Selectivity

| Metric | Claude number | Conditions | Line | Note |
|---|---|---|---|---|
| Same-latent dissociation | report + flexible flip "**on essentially every trial**"; continuation + anomaly "**largely unmoved**" | **n=8 passages**, 4 conditions (continuation, anomaly detection, explicit report, flexible computation ×3 function types); swap across question tokens only; language present in lens at comparable rates in all conditions | 522–533 | gpt2 n=12–19 vs paper n=8 — gpt2 actually has more trials; gpt2 lacks anomaly condition. |
| Character-count variant | linewrap: count absent from lens, swap no effect; report: answer follows swap; first-letter: highest lens presence, follows swap | n=11 passages; swap maps 40s→60s lens counts | 537–547 | no gpt2 analog. |
| Ablation protocol | k=**10** top-activated lens vectors projected out per position; **protect clean top-10**; doses = layer-band widths (light/medium/heavy) | multihop eval: near-ceiling → "near zero" under heavy. Pretraining top-1 match: figure-only ("substantially less" perturbed than multihop) | 555–563 | gpt2 doses by k at fixed band = transposed protocol; compare at k=10. No 80% bar exists in paper. |
| Task battery | shallow (MMLU, odd-one-out, SQuAD, sentiment, CoLA) "**essentially unaffected** … at or near baseline" even heavy; inferential (Caesar, analogy, summarization, TriviaQA, multihop, translation, sonnets) "**well below unablated Haiku 4.5**"; GSM8K with CoT much more robust than direct | 14 tasks, normalized to unablated Sonnet 4.5 | 569–577 | figure-only levels; qualitative pattern is the comparison. |
| Experiential reports | ablation (k=10, **L38–54** only) reduces experiential-language score "dramatically" on Sonnet/Opus 4.5/4.6; **matched-norm controls near baseline**; Haiku 4.5: coherence breaks first. Story quality only modestly reduced | LLM-graded 3-rubric score; controls calibrated to same per-position magnitude (random dir, non-J shrink, top-10 SAE dampening, J-stripped activation bank) | 583–598, 2122–2157 | figure-only; gpt2 has no experiential-report analog (could be a hillclimb target?). Control battery is the model for fixing gpt2's C5b controls. |

## Structure

| Metric | Claude number | Conditions | Line | Note |
|---|---|---|---|---|
| Workspace band | ≈ **L38–92** (of 0–100 reindexed) | CKA blocks + 4 metrics (top-k next-token acc; excess kurtosis; top-1 autocorrelation vs shuffled null; effective dimensionality) | 611–647 | motor metric (Fig 28a) = fraction of positions where **any top-k lens token matches model top-1**. gpt2's 0.5-on-top-1 rule differs; L10 = 0.53 under paper metric at k=10. |
| Ignition | sharp commitment from ~L38; J-space transition narrower & earlier than non-J | 16 country pairs × 40 carrier sentences, embedding mixture sweep | 649–713, 1882–1916 | no gpt2 analog implemented. |
| Occupancy | plateau ≈ **25** (median) across workspace band; near zero first third | K where marginal reconstruction improvement falls below a **same-size random control set** | 719–727 | gpt2's 2–3 uses top-64-candidate GP vs 4096-random control — control not size-matched; fix before headline comparison. |
| Excess variance | "never exceeding **10%**" | top-K lens vectors at K=median occupancy, **in excess of same-size random control** | 729–737 | gpt2's 3–18% has NO control subtraction — subtract random-control share before comparing. |
| List capacity | unrelated lists: ~**6** words present (pooled over band); **1–2 at single layer** (L79); related lists: whole 80-word family present | present = best rank over band in top 25, measured at commas | 753–761, 1918–1922 | no gpt2 analog implemented. |
| MLP gain | J-lens vectors ≈ **10×** at workspace peak; neuron output-weight rows ≈ 1 throughout; rises from 1 at onset, falls at motor | gain of MLP block ℓ+1 on 2,000 unit directions, normalized to isotropic-random median | 773–813 | gpt2's 1.05–1.32× is the same quantity (N=512, centered) — genuine quantitative gap, not metric mismatch. |
| Broadcast heads | top-1% J heads: recall@25 falls 0.86→**0.67**; behavior top-1 changes only 5% vs 2% control; injected-thought report 0.54→**0.09** under head ablation | OV gain + label preservation, workspace layers | 823–903 | no gpt2 analog implemented. |
| SAE features in J-space | peak ≈ **15%** of features pass kurtosis baseline (excl. motor); low-κ features = syntax/bookkeeping | κ = lens-projection excess kurtosis | 1936–1950 | no gpt2 SAE available in repo. |
| Logit-lens comparison | causal: J-lens ablation ≈ **2×** the KL of logit-lens directions on multihop; swap flips more across 3 scales; readout: J-lens best on all 6 distributions but margin "**modest on multihop**" | pass@k AUC; KL; α=1 swaps | 1335–1359 | gpt2's L9-two-hop parity is consistent with "modest multihop margin" — cite that, not general superiority. |

## No commensurable Claude number exists for

- C1a Spearman **value** (figure-only) — compare the rising-shape, not a number.
- C2 focus/ignore/mention/don't-think **rates** (figure-only) — the ordering is the comparable object.
- C2 demand-loading with a named word (43_demand) — repo-invented variant; paper's implicit-demand design is different and its systematic numbers are figure-only.
- C1d inject rates vs strength (figure-only; only 0.54 at Fig 34b).
- C3a pooled "intermediate in lens top-10" % (never stated for the 50-prompt set).
- C3c both-questions-flip cross-function rate (repo-invented design; paper's anti-smuggling = 17%-earlier depth timing, unimplemented in repo).
- C5b per-task retention **numbers** and any 80% threshold (paper is qualitative; bar is repo's own).
- Experiential-report scores (figure-only; no gpt2 analog anyway).
- Steering strength units (paper's α un-unitized; repo's mean-resid-norm rule is its own).
- Median-rank versions of any C2 metric (paper uses top-1-at-any-(layer,position) rates).
