# Log (day 3+, gpt2/ rewrite)

Chronological; [E] exploration, [C] confirmatory (frozen first). Numbers here
are exploration unless marked.

## 2026-08-17 (day 3)

- Reorganized repo: days 1–2 → `old/`; new minimal codebase in `gpt2/`.
- Launched gpt2-medium + gpt2-large lens fits (Neuronpedia recipe, n=1000,
  sequential background job; medium ~21s/prompt).
- `core.py` written; **00_validate**: readout == reference `apply()` exactly
  (0.0 max diff); lens-vector identity holds (2e-4); centering shifts every
  logit by a per-position constant (1e-4 spread) — gauge invariance confirmed
  numerically. Smoke swap Poland→Spain flips Warsaw→Madrid (raw and centered).
- **10_cone** [E→settled]: the cone axis IS the vocabulary-mean vector
  (cos = 1.000 every layer), carrying 97–99% of dictionary second moment; its
  logit profile is constant across the vocab to ~2% (softmax-invisible ⇒ pure
  gauge). Centered: |cos| 0.07–0.10, no dominant PC, gradient pursuit k=16
  recovers 3–16% of activation variance (raw: 0.0–0.6%). Provenance: v̄ = Jᵀū;
  GPT-2's unembedding is 83% mean; J amplifies ū 2.6–6.9× vs 1–1.7× random.
  → Method decision in PLAN.md: geometry in the centered gauge everywhere.
- **20_capability v1** [E] on gpt2-small: two-hop 42/183; found (i) one-hop
  "The capital of Poland is" fails in *prose register* (model continues " now
  home to...") while the model demonstrably knows the fact — quiz-register
  formats needed for base GPT-2; (ii) my Spanish/French stopword classifier
  misattributed genuinely-Spanish continuations; (iii) report few-shot shots
  leaked into answers.
- Format bake-offs [E]:
  - one-hop capital: "The capital of {c} is" 0/26 → "France's capital is
    Paris. {c}'s capital is" 22/26; continent "on the continent of" 5/26 →
    "{c} is a country on the continent of" 15/26.
  - two-hop: lang_capital "…where {arg} is spoken is the city of" 6→11/22;
    lang_continent "…is a country on the continent of" 6→10/22;
    city_language "In the country whose capital city is {arg}…" 20→21/28.
    city_continent ≤5/28 in all forms (capability wall at 124M); riddles 0/5
    in all forms ("A spider has" → " been") — dropped at small, retest larger.
  - report format: QA few-shot induces category echo + shot bleed; colon-list
    5-shot (disjoint shot categories) best: 7/16 top-1-valid, 9/16
    member-in-top-5 (with member-list additions: USA, pear, tea…). planet/
    tree/bird/city/river echo the category — small-model ceiling.
- Capability caution flagged: city_language two-hop passes 21/28 while its
  firsthop (capital→country) is ~0 — GPT-2 may shortcut Warsaw→Polish without
  routing through Poland. The C3 readout + cross-function control adjudicates.
- 20_capability v2 running on all three sizes (background).

- **50_reasoning readout** [E] gpt2-small, 52 capability-passing items:
  content is a late cliff — intermediate median rank 338 (L8) → 15 (L9) → 8
  (L10); null country stays ~900; top-10 rate 0% below L9. Family split is
  the story:
  - lang_capital: 8/11 top-10, median best rank 2, and the intermediate is
    NOT an output candidate (output rank median 98, 0/11 in output top-10) —
    genuinely held-not-spoken. The clean flagship family.
  - lang_continent / city_continent / paper_language-capital: lens hits but
    the country is ALSO in the model's output top-10 (naming-the-country is a
    live continuation) — flagged contaminated; added `unspoken` flag
    (int not in output top-10) to the readout as a formal criterion.
  - city_language: intermediate absent from lens (0/21) + firsthop ≈ 0 →
    the model shortcuts capital→language by association without computing
    the country. A real negative control: the lens distinguishes items that
    require the intermediate from items that bypass it.

- **40_modulation** [E] gpt2-small: think 14 / dont 26 / base 657 (36/36
  think<base; 24/36 think<dont; non-blurt trials 69/117/925 — ordering
  holds). Effect at L9–10. Extreme white-bear: dont≈think (mention installs).
- **80_structure band** [E]: lens-vs-model top-1 agreement maxes at 0.27
  (L10) — NO fitted layer is motor by the old ≥0.5 rule; output conversion
  happens inside the unfitted final block. L9–10 swaps are not logit edits.
- **50 swap sweep** [E] n=35 (target-top10 guard; continent families
  excluded by it — only ~5 continents): coordinate swap 63–77% top-1 at α=1
  (raw ≈ centered, as gauge theory predicts); **projection swap 0% raw,
  46–69% centered** — the centered gauge rescues the paper's §3.1 method
  from complete failure. α=2 no better. Effects present without L10
  ("mid" L5–9: 66–71%). Family split: lang_capital 10/11 (91%);
  city_language 13/18 (72% — mid-rank content is causally load-bearing);
  paper_language-capital 4/4.
- **50 crossfn** [E]: both-flip 14/27, any-flip 22/27; best pairing
  lang_capital+city_language 8/10 both. Failures are near-misses (incl.
  " Southeast" for Thailand-continent — semantically right, graded wrong;
  pre-declare variant later). Anti-smuggling passes; city_language is
  country-mediated after all.
- **50 probe** [E] n=45: matched-norm component swaps — full 69%, J 58%,
  nonJ 27%, nonJ+clamp 4% (paper: 60/61/28/6). Probe J-share median 37%
  (paper 10–15%; raw gauge day-2: 0.16%). The cone-blocked privilege
  control now REPLICATES THE PAPER'S PATTERN at GPT-2 scale under the
  centered gauge.
- **30 corr** [E]: report↔lens Spearman rises monotonically to +0.61 at L10
  over 9 usable categories (paper's signature shape).
- Capability v2, medium/large: two-hop pool 52→94/96 (continent families
  come alive); report 8/16 top1-valid, 12–14/16 member-top5; C4 non-country
  categories still mostly dead at large (as in the paper's own weak cells).
