# Round 11, 2026-09-30: two directed-modulation figures in Appendix C.2

- **New Figure 6** (`figures/fig6_modulation.png`, from `post/figures/dm_rates.py`): the paper's
  Fig. 65 for all five models, one row per model and one column per task family, narrow for the
  reading column.
- **New Figure 7** (`figures/fig7_modulation_frames.png`, from `post/figures/dm_gpt2_frames.py`):
  GPT-2 in each of its six prompt frames, the paper's score only, with how much of the sentence
  GPT-2 predicts itself in each title.
- **The band-statistics figure is now Figure 8** (`fig6_band_stats.png` renamed `fig8_band_stats.png`;
  `band_stats.py`, the reference in "The structure", its caption, and "(Figure 8b)" in D updated).
- **C.2, new sentences** after "...88% to 100% of the tokens after the first.": the story's hits come
  from GPT-2 being about to write the word (a member among its own top 10 next tokens on 23% to 35%
  of instructed trials, against 5% to 8% on the paper's prompt; with tokens where a member is in its
  top 25 left out, 0.2% or less).
- `post/NOTES.md`: the figure list. `post/DM_SETUP.md` uses the new figures. Removed from
  `post/figures/dm/`: `dm_prompts*.png`, `dm_lines_narrative.png`, `dm_gpt2_frames.png` (replaced
  by Figures 6 and 7); `dm_rates.py` no longer takes a GPT-2 frame argument.

# Round 10, 2026-09-30: the post updated for directed modulation on the paper's prompt

Backups of `post.md`, `post.html`, `CHANGES.md`, and the Figure 1 files from just before this round
are in the session scratchpad (`round10_backup/`). The numbers and their sources are in
`post/DM_SETUP.md` (round 9, below).

## Every edit to post.md

- **How each test went, directed modulation bullet.** "almost never has a citrus fruit on top of
  its J-lens, and the wording matters more than what the instruction asks: "don't think about
  citrus fruits" raises them more than "think about citrus fruits"" became "... (0.5% of trials),
  and what the instruction asks makes no difference: "think about citrus fruits" does no more than
  "ignore citrus fruits"". Qwen: "25% of "think about" trials, and under 1% of "don't think about"
  trials" became "23% of "think about" trials, 4% of "ignore" trials, and 1% of "don't think
  about" trials".
- **The directed modulation section.** Five paragraphs became four, restructured to lead with the
  three parts:
  1. The three things the paper tests, and that GPT-2 can only be tested on the first. This
     replaces "The definition has three parts..." and the old fourth paragraph (Claude's math and
     paired-question results, "We couldn't test either in GPT-2").
  2. Claude: now gives the bare-mention rate (66% to 87%), says "don't think about" leaves Claude
     at that rate (it said "brings it down less"), and "(orange, lemon)" became "(orange, lime)"
     (lemon isn't a tracked member).
  3. GPT-2: on the paper's prompt as plain text, 0.5% whatever the instruction; "think about"
     ranks the members no higher than a bare mention and "ignore" no lower; five other framings,
     where "ignore" does rank the category below a bare mention, and in all six "don't think
     about" ranks it above "think about". Gone: the 72% and
     65% figures and the per-frame sentence about "ignore" and "think about", which were for the
     old main frame.
  4. Qwen: 23% / 16% / 4% / 1%, "the same as with no instruction". The sentences on the
     instruction order, the 15 of 24 sums, and the 2 of 7 passage questions moved to Appendix E
     ("Qwen doesn't show the other two parts either (Appendix E)").
- **Why the tests are cheap, directed modulation paragraph.** "Mentioning citrus fruits does that,
  and so does "don't think about citrus fruits", since text that talks about not thinking about
  something tends to go on about it." became "Mentioning citrus fruits does that, whatever is said
  about them: on the paper's prompt, "ignore citrus fruits" raises them as much as a bare
  mention." New sentence after
  the Qwen sentence: "Told to think about the category and left to write its own reply, it often
  brings the category into the text (Appendix E)."
- **Limitations, last bullet.** "...may be a failure of our prompts rather than of the model."
  became "...may come from how we gave it the prompt. We gave it the paper's prompt as plain text,
  and five other framings didn't show the pattern either."
- **B.2.** The materials sentence (tracked members `orange`, `lime`, `mandarin`; the answer as a
  digit or a number word; the two extra math phrasings). New: how members become tracked tokens;
  the paper's prompt and where it comes from; the main frame is that prompt as plain text with all
  20 sentences; the old main frame is now the first of five other frames, which are named. The
  earlier version's example was `Think about lemon`, which was never run; it is `Think about
  orange`. The arithmetic line now describes the three capability prompts.
- **C.2.** New main table (the paper's prompt, 20 sentences). New: the hit rate over other layers;
  Claude's mention and "don't think about" rates. The paired-comparison table has a new first row,
  and every frame is rerun with all 20 sentences. New paragraphs: what holds across
  frames, the wording effect, and that the story frame isn't a copying task. The arithmetic
  paragraph has the worked-example checks and the paper's-prompt numbers.
- **E.** The directed modulation table and paired comparisons are the paper's prompt with 20
  sentences. New paragraphs "Other prompts" (the old numbers, and the instruction-before order)
  and "Qwen's own replies". Math: 0 hits in 2,100 "think about" trials on the paper's prompt. The
  band paragraph now reads layers 9 to 21 and every lens layer from the same run.

Not edited: the TL;DR ("about a quarter of Claude's rate" still holds: 23% against 93% to 97%),
the Figure 1 caption, and "What this means".

## Figure 1

`fig1/template.html`, rebuilt: the directed modulation card's second line `"don't think about":
↑↑` became `"ignore": no effect`; "25%" became "23%"; the thought bubble "orange? lemon?" became
"orange? lime?".

## Code

- `jl/control.py`: new `arithmetic` experiment (GPT-2's capability check on the math problems in
  three plain-text formats; `results/control/gpt2/arithmetic.json`). `modulation_copy` also saves
  the instruction-tuned model's own greedy replies.

# Round 9, 2026-09-30: directed modulation on the paper's own prompt

`post.md` is **not edited** in this round. Everything is in `post/DM_SETUP.md` (open
`post/DM_SETUP.html`), which has the setup knob by knob, every difference from the paper, the
results, and a draft of the section for the post.

## Why

The code said the paper doesn't release its prompt template for directed modulation. It is in the
data behind the paper's Fig. 9 (`data/modulation-readout/modulation.json` on the paper's page):
`Write "{sentence}" {instruction} Don't write anything else.`, with the sentence as the forced
reply. We had run Qwen on `Write the following sentence: "{sentence}" {instruction}` and GPT-2 on
frames of our own, with 10 of the paper's 20 sentences, and without the two math-only focus
phrasings of its Fig. 65.

## What the reruns say

- The verdicts don't change. GPT-2: a member on top of the lens on 0.5% of trials in every
  condition (at most 1.4% over any layers). Qwen3.5-0.8B: 22.6% think about, 15.8% mention, 0.8%
  don't think about, 4.2% ignore, 0.7% no instruction (the post has 25.0 / 17.3 / 0.6 / 3.5 / 0.0
  from the old prompt and 10 sentences).
- GPT-2's rank comparisons change with the prompt: on the paper's, "think about" ranks the category
  above a bare mention in 30% of pairs (72% in the old `copy` frame), and "ignore" in 50%. "Don't
  think about" still beats "think about" (65%).
- Qwen math on the paper's prompt, the 15 problems it can do: 0 hits in 2,100 "think about" trials
  and 2 in 1,800 with a bare mention (all 20 sentences).
- New checks: GPT-2 can't do the sums with worked examples either; the story frame isn't a copying
  task for GPT-2 (it predicts 28% of the sentence's tokens); Qwen's own reply under "think about"
  is exactly the sentence on 25 of 66 trials.

## Code, data, and files

- `jl/control.py`: frames `paper` (Qwen) and `human` (GPT-2) in `mod_text`; new experiments
  `modulation_grid` (every lens layer and position kept, all 20 sentences, 26 phrasings),
  `modulation_readout` (the paper's Fig. 9 for our models), and `modulation_copy` (does the model
  copy by itself). The existing `modulation` experiment and its result files are untouched.
- New results: `results/control/{gpt2,qwen}/modulation_grid_*.{json,npz}`,
  `modulation_readout_*.json`, `modulation_copy.json`. Qwen's grid was run one family at a time
  (`FAMILIES=topic`, then `FAMILIES=math`) and the two files joined in order.
- `ref/paper-data/`: `modulation-readout.json`, `modulation-prompts.json`,
  `top-down-summoning{,-appendix}.json`, `metacog-alarm.json`, `dual-task-simple.json`,
  `modulation-probe.json`, and their README entries.
- `post/figures/dm_data.py` (loading and scoring), `dm_readout.py`, `dm_rates.py`, `dm_knobs.py`
  (figures, written to `post/figures/dm/`), and `dm_tables.py` (every table in `DM_SETUP.md`).

# Round 8, 2026-09-30: no "like the paper" framing; Qwen's injected-thought results removed

- Following the paper's method is the default, so the text no longer points it out: "on the
  paper's (own) measure" is gone from the TL;DR, the summary bullet, the section title, the first
  GPT-2 sentence, and "What this means"; "Like the paper, we score only the bare token" is gone
  (the "The dog" sentence now just says counting ` dog` too gives 41 of 57, against 11 for `dog`);
  "as in the paper's protocol" is gone from the Figure 3 caption, B.1, and C.1. The C.1 scoring
  note still says the protocol only specifies the report and we use the same token elsewhere.
- Qwen's injected-thought results are removed: the C.1 paragraph and the Appendix E paragraph.
  "What we did" and the start of Appendix E now say Qwen was run for directed modulation. Qwen's
  verbal-report and layer-statistics checks in E stay. The Qwen injection code and result files are
  untouched.

# Round 7, 2026-09-30: score the injected thought as the paper does (bare token only); drop the Q/A frame

Backups of `results/control/` and the post files from before this round are in the session
scratchpad (`round7_backup/`).

## Why

The released protocol scores "the rank of `surface`" (the bare token, e.g. `lightning`). We had
counted the best rank over the word's forms (`dog`, ` dog`, `Dog`, ` Dog`). Switched to the bare
token at the report and at every other position; the any-form ranks are still saved
(`report_forms`, `other_forms`).

## Code and runs

- `jl/control.py`: `introspect` and `introspect_privilege` score the bare token and also save the
  any-form ranks. New `--vectors` and `--intro-frames` options for `introspect`.
- Rerun, fine grid: GPT-2 centered vector, one-line and transcript frames, bare and word-initial
  token, both replies (8 files); the one-line raw-surface files for both replies; the privilege check.
- Not rerun: the Q/A (`interview`) frame, now dropped from the post (its files are left over from
  round 4, any-form scoring); the other raw-vector files (any-form; the post's "at most 3 of 57"
  still holds, since the bare token can only rank lower); Qwen (stopped for time; one file,
  `qwen/introspect_chat_raw_surface.json`, was rewritten with the new scoring before it stopped;
  the post notes its numbers count any form, which can only overstate them).

## What changed in the numbers

- Report: unchanged (top-5 counts and medians identical at every strength; 3 of 1,425 ranks
  differ, all at rank > 800).
- Other positions, main setup: pooled median at strength 0.2 is rank ~320 (was ~40); averaged per
  concept, ~95. Claude ~60. At the best strength (0.44): ~54 pooled, ~13 averaged.
- Top 1 after "The" at 0.44: 11 of 57 bare (41 any form); after "an injected": 5 (38).
- Clean (top 5 at the report and nowhere else): up to 39 (was 15).
- The per-position story ("the word comes out where the reply names the thought") doesn't hold for
  the bare token (median ranks at 0.2: report 1, then 45 to 1,778 elsewhere). Replaced in "Why the
  tests are cheap" by: the bare token fits mainly after the open quote (with nothing injected, rank
  ~2,000 there and ~37,000 elsewhere), so the control mostly checks that the bare form isn't said
  where it wouldn't fit anyway.
- The band-width argument for pooling is gone (under bare scoring Claude's 15-17x sits between
  GPT-2's averaged ~10x and pooled 25x+); not needed, since both summaries put GPT-2 below Claude
  where its report first reaches the top.
- 8-setup table: every setup other than the main one has a weaker report (top 5 at most 40 of 57,
  and none reaches median #1). With the ` dog` vector the word is rarely elsewhere as a bare token.
- Privilege: J-space part 44% at k = 2 (was 49%) and 39% at k = 16 (was 47%); rest of the random
  split at k = 16, 30% (was 32%). Everything else the same. The conclusions hold.

## Every edit to post.md

- Verbal report section: "around rank 40" → "around rank 300 (Claude: around 60)"; the "The dog"
  sentence now says we score the bare token like the paper, and that counting any form the word is
  #1 after "The" for 41 of 57 at the best strength, against 11 for the bare token.
- Figure 3 (redrawn) caption: both lines count only the bare token; the pooling-band sentence is gone.
- Why the tests are cheap: the injected-thought paragraph rewritten as above.
- B.1: the score is the bare token's rank; pooling paragraph without the band-width argument; two
  frames, not three.
- C.1: new main table (pooled and per-concept-averaged other positions; "after The" bare / any form;
  clean), per-position ranks for the bare token, the scoring paragraph flipped (we score the bare
  token; any form differs only at the other positions), 8-row setup table, privilege table and
  paragraph, Qwen note.
- Figure 5 (redrawn, transposed): columns are the two questions, rows the reply × vector.
- E: note that the Qwen counts include any form.

# Round 6, 2026-09-30: the privilege check for the injected thought

New experiment: `python -m jl.control --model gpt2 introspect_privilege` (`jl/control.py`,
`introspect_privilege`) → `results/control/gpt2/introspect_privilege.json`. The paper's Fig. 8
(right) in the main setup: concept vectors from `Tell me about {concept}.` minus the mean of the other
100 concepts, split by pursuit (k = 2 and 16), injected in place of the J-lens vector; plus the
clamp, a random direction, and the random-dictionary split from c1_report. Strengths 0.01 to 5.12.
Peak report top 5 (of 57): J-lens 52, J-space part 28 (k=2) / 27 (k=16), full 27, rest 4 / 3,
clamped 0 / 0, random 0, random part 0 / 4, rest of random split 25 / 18. Claude (Fig. 8 data): 89,
79, 57, 12, 3, 0%. The J-space part peaks at the top of the grid (k=2 is flat at 21-28 from 0.24;
k=16 still creeping up).

Edits to `post.md`: one sentence in the verbal-report section ("The paper's privilege check comes
out as in Claude too…"); one sentence ending the verbal-report paragraph of "Why the tests are
cheap" ("The same check on the injected thought behaves the same way"); a method paragraph in B.1;
a table and a short paragraph in C.1.

# Round 5, 2026-09-30: a shorter main text for the injected thought, and an appendix figure

## Every edit

- **Figure 3.** Each panel against its own per-layer strength, as in the paper (GPT-2 up to 0.5);
  no "after The" line; one-line panel titles, with the sources moved to the caption. The "counted
  as 55 layers, as if Claude had 100" framing is gone everywhere: the text now says only that the
  strength is per layer and the paper's band presumably has many more layers.
- **Verbal report section.** The GPT-2 paragraph drops the "rank ~11 at the best strength"
  sentence. The "Two things qualify this" paragraph is now shorter: the result depends on the setup
  (reply, token, question; details in C.1), GPT-2 still puts the word at #1 after "The" for 41/57 at
  its best strength, which pooling hides, and a closing line: "We take all this to show that a model
  like GPT-2 can pass the test, not that GPT-2 passes it in general."
- **How each test went bullet.** Ends "On the paper's measure GPT-2 can do this about as well as
  Claude, though how well depends on the exact setup." (the "The dog…" sentence is gone).
- **Limitations.** The setup bullet is one sentence.
- **C.1.** The six-version table and the separate word-initial-token paragraph are replaced by one
  12-row table (3 questions × 2 replies × 2 tokens) and the new **Figure 5**
  (`post/figures/introspect_setups.py` → `fig5_introspect_setups.png`), the same 12 setups as
  small multiples. New paragraph on scoring: we count any form of the word (`dog`, ` dog`, `Dog`,
  ` Dog`); the paper scores the bare token at the report. At the report this changes nothing (top-5
  counts and median rank identical at every strength; 3 of 1,425 ranks differ, all at strength ≤
  0.005 and rank > 800). At the other positions, bare-only would put the pooled median at rank ~320
  instead of ~40 at strength 0.2 (~54 instead of ~11 at 0.44), so our choice is the stricter one.
  Source for the bare-only numbers: the scratch run `fine2_word.json`, which stored both scorings
  (the repo's result files store only the any-form ranks).
- **E.** Adds that we only tried the paper's chat prompt with Qwen, and the question's wording
  mattered a lot for GPT-2.
- **Figure 6.** The band-statistics figure (was Figure 5) is renumbered, since the new figure comes
  before it: `fig6_band_stats.png`, `band_stats.py`, and the four references in `post.md`.

`five_tests_draft.md` has the same verbal-report section as `post.md` again.

# Round 4, 2026-09-29: the injected thought passes on the paper's measure

Backups of `post.md`, `post.html`, `five_tests_draft.md`, `CHANGES.md`, and `fig1.png` from just
before this round are in the session scratchpad (`*_before_round4.*`).

## Why

The post said GPT-2 fails the injected-thought test because the word comes out at other points of
the reply ("an injected ___", "The ___") as well as at the report. Two things were wrong with the
comparison to Claude.

1. **The reply.** The released protocol (`verbal-introspection.json`) has two prefills, `default`
   ('... The thought is about "') and `word` ('... about the word "'). The paper's Fig. 7 uses
   `word`. We had only run `default`. With `word`, GPT-2's report is much stronger: in the one-line
   frame the word is in the top 5 at the report for 52 of 57 concepts at the best strength (was 40),
   and the median concept is the top prediction from strength 0.2 on.
2. **The control.** The paper's position control is a median (with quartiles) over "every other
   position in the assistant turn". Its released Fig. 7 data (now `ref/paper-data/verbal-introspection.json`,
   from `data/verbal-introspection/data.json`) has only those pooled numbers. Pooled the same way,
   GPT-2's other positions stay well below its report (rank ~40 where the report first reaches the
   top; Claude ~60). The per-position leaks the post described are still there (after "The" top 1
   for 41/57 at the best strength), but the pooled median can't see a leak at one or two of ~15
   positions, and we can't check Claude position by position. The width of Claude's band fits
   pooling (upper/lower quartile 15-17x; GPT-2 pooled 15-24x, per-concept mean ~3x).

Also: the per-layer strengths differ tenfold (GPT-2's report reaches the top at 0.2, Claude's at
0.02), but we add the vector at 3 layers and the paper at every layer of its band, 38-92% of
Claude's depth (the paper numbers layers by percent of depth; Claude's layer count isn't public),
presumably many more layers. The post says this in one sentence and doesn't try to equate the two.

What we checked and dropped: an explanation that the word wins where the next token is most open
(high clean entropy). Spearman over the 16 positions was only 0.2. What holds instead: at strength
0.2 the word's median rank is 1 at the report, 2 after "The", 3 after "an injected", 7 after "about
the", and 13 to 282 at the other twelve positions.

## Code and results

- `jl/control.py`: `--prefill {default,word}` (the released prefills; `word` runs are saved with a
  `_word` suffix) and `--strengths {paper,fine}` (fine: 24 steps from 0.005 to 1.0, a superset of the
  paper's). `gpt2_intro_frames(prefill)` replaces the fixed `GPT2_INTRO_FRAMES` dict (kept as an
  alias for the default prefill).
- New: `results/control/gpt2/introspect_{researcher,transcript,interview}_{raw,centered}_{surface,space}_word.json`
  (fine grid), `results/control/qwen/introspect_chat_*_word.json` (paper grid).
- Rerun on the fine grid: the 12 GPT-2 `introspect_*.json` files with the default reply, so all six
  prompt versions in C.1 share one grid. The fine grid contains the old strengths, and the numbers
  at those strengths are unchanged.
- Every injected-thought number in the post was recomputed from these files (a check script is in
  the session scratchpad, `verify_c1.py`).
- `ref/paper-data/verbal-introspection.json` and its README entry.

## Figures

- **Figure 3 is new** (`post/figures/introspect.py` → `fig3_introspect.png`, replacing the superseded
  Qwen3-1.7B version): the right panel of the paper's Fig. 7 for Claude (the paper's data) and GPT-2
  (main frame, strengths up to 0.5), each against its own per-layer strength, with the same two
  series as the paper's (the report and the other positions, pooled).
- **Figure 1**: the verbal-report card's "injected thought (an extra test)" mark is now ✓.

## Every edit to post.md

- **TL;DR.** "Neither small model can report an injected thought." became "On the paper's own
  measure, it also reports a thought injected into its activations." The last bullet now names one
  test neither small model passes (directed modulation at Claude's strength), not two.
- **How each test went, verbal report bullet.** Retitled "Verbal report works, and so does the
  injected thought on the paper's measure". Says GPT-2 passes on the paper's measure with a
  one-line question, and also says the word at a few other points ("The dog…"), which the measure
  doesn't pick up and we can't check in Claude. The Qwen sentence is gone.
- **Verbal report section.** Same new title. The injected-thought paragraph is now three paragraphs:
  the test and Claude's result with the paper's quote; GPT-2's result (52/57, rank ~40 elsewhere,
  one sentence on per-layer strength vs. Claude's many more layers, rank ~11 elsewhere at GPT-2's
  best strength); and two
  qualifications (prompt and token dependence; per-position leaks that pooling hides). Qwen is no
  longer mentioned here. Figure 3 and its caption follow.
- **Why the tests are cheap.** "Directed modulation and the injected thought are different" became
  "Directed modulation is different". The "Neither GPT-2 nor Qwen3.5-0.8B passes it" ending of the
  verbal-report paragraph became a new paragraph: the J-lens vector makes the word the topic, and it
  comes out where the reply could say what was injected (the per-position ranks), so the pooled
  control checks that the word doesn't come out at most points, not that it waits to be asked. The
  "So two of the paper's tests aren't passed..." paragraph now names only directed modulation.
- **What this means.** Four of five, "and on the paper's measure also reports an injected thought";
  the exception is directed modulation only. "and perhaps from directed modulation and the
  injected-thought test" became "and perhaps from directed modulation".
- **Limitations.** Two new bullets: the result depends on the prompt and token (all versions in
  C.1), and Claude's released data is pooled, so no per-position comparison. "GPT-2's failures on the
  two tests that need instructions" became "GPT-2's failure on directed modulation".
- **B.1.** Main frame now ends `about the word "`; each frame run with both replies; the fine
  strength grid; how the other positions are pooled and why.
- **C.1.** New main-frame table (strengths 0 to 1.0: report median RR, top 1, top 5; other positions
  pooled; top 1 after "an injected" and "The"; clean); Claude's Fig. 7 numbers; per-position median
  ranks at 0.2; a table of all six prompt versions; the word-initial token; raw vectors; Qwen with
  either reply.
- **E.** One sentence: with the Fig. 7 reply, Qwen reaches the top 5 for none of 75 concepts.

`five_tests_draft.md` got the same verbal-report edits and new sources. `NOTES.md` has a short
section on this round; its older verdicts on the injected thought are marked as out of date.

# Round 3, 2026-09-28 (late): flexible generalization's two "smaller results"

Line numbers are for the current `post.md`. Backups of `post.md`, `post.html`, and
`five_tests_draft.md` from just before this round are in the session scratchpad
(`post_before_round3.*`, `five_tests_draft_before_round3.md`).

## Why

The post said two results from the paper's Fig. 19 "don't hold in GPT-2": doubling the swap
strength helps Claude but not GPT-2, and workspace loading doesn't predict which swaps work.
Neither holds up against the paper's Fig. 19 data (`ref/paper-data/flex-gen-systematic.json`,
new; the paper's page loads it from `data/flex-gen-systematic/systematic.json`).

1. **Doubling.** Claude's 76 to 101 of 192 is over all 192 swaps. By category it goes countries
   42 to 35 of 48 (worse), months 24 to 35, animals 10 to 20, numbers 0 to 11. Most of the gain is
   on prompts GPT-2 can't answer. On the functions GPT-2 can do, both models move the same way:
   countries down (GPT-2 11 to 9), "the month after" up (Claude 0 to 6 of 12, GPT-2 0 to 2). Claude's
   α=2 on our 46 testable swaps can't be computed (the paper releases per-swap results for α=1
   only); from the per-function counts it lies between 13 and 28, against 16 at α=1.
2. **Loading.** The paper's y-axis is a continuous swap effect (Δ log-prob of the target answer
   minus Δ log-prob of the spontaneous answer), per function, over all 12 swaps. On that measure
   GPT-2 correlates at Pearson 0.60 / Spearman 0.64 across the 16 functions (Claude 0.91 / 0.92);
   0.60 / 0.44 without the country functions (Claude 0.92 / 0.88); Spearman 0.69 over the 64
   prompts. The post's −0.11 was a different measure (top-1 success rate over the 25 prompts with
   testable swaps), and it's still reported.

We read "spontaneous answer" as the model's own top answer before the swap; the paper doesn't
define it further.

## New results (`results/c4_generalization/`, rerun; every existing number is unchanged)

- Mean swap effect, coordinate swap, all 192 swaps: 3.57 at α=1, 6.39 at α=2 (countries 9.4 and
  16.3). Doubling pushes the target further; it doesn't raise top-1 because GPT-2 outputs the
  swapped-in argument itself more often (30 to 55 of 185, already in the post).
- Top-5 instead of top-1 (Claude's ranks at α=1 are in the Fig. 68 grid): all 192 swaps, GPT-2
  100 (α=2: 94) and Claude 105, with 49 of GPT-2's already in its top 5 before the swap. On the 32
  testable swaps whose answer starts outside GPT-2's top 5: GPT-2 23 (α=2: 24), Claude 17. The
  difference is "the month after", 8 of 11 against 4 of 11.

## Every edit to post.md

- **Line 115** (flexible generalization). "Two smaller results from the paper don't hold in GPT-2.
  Doubling the swap strength helps Claude (from 40% to 53%) but not GPT-2, and how strongly the
  argument is present in the J-lens beforehand doesn't predict which swaps work." became four
  sentences with no numbers (they're in C.4): the two results "look much the same in GPT-2";
  doubling helps Claude mostly on functions GPT-2 can't answer, and on the ones it can it hurts the
  country functions and helps "the month after" in both; loading predicts how far a swap moves the
  answer in both, more weakly in GPT-2. The sentence defining the 46 swaps was also rewritten
  ("We count only the 46 swaps where GPT-2 knows the answer the swap should produce, and wasn't
  already giving it.").
- **Line 585** (C.4). "This is why doubling the strength doesn't help" became "doesn't raise the
  top-1 rate", after a new first part giving the swap effect at α=1 and α=2, and a closing
  sentence with Claude's per-category counts at α=2.
- **Line 587** (C.4, new paragraph). The top-5 result.
- **Line 589** (C.4). "This order matches the paper" was wrong: the paper's category order is
  countries > months > animals > numbers, ours countries ≈ animals > months > numbers. Now says the
  ends match. Adds the loading vs swap effect correlations; keeps the −0.11, labeled as top-1.

`five_tests_draft.md` line 45 had the same paragraph as line 115 and got the same edit.
`post.html` is rebuilt (`build.sh`); apart from the edited paragraphs, only pandoc's boilerplate CSS
changed.

## Code, data, and other files

- `jl/c4_generalization.py`: each swap row records `effect` (the paper's Fig. 19 swap effect) for
  both operations and both strengths. The summary adds the mean effect at α=1 and α=2, the top-5
  scores with Claude's from the grid ranks (`claude_ranks()`, new; `claude_grid()` now uses it),
  and the loading vs swap effect correlations for GPT-2 and Claude.
- `jl/stats.py`: `pearson` (new).
- `ref/paper-data/flex-gen-systematic.json` (new) and its entry in `ref/paper-data/README.md`.

## Open decision

- **Line 171** ("Why the tests are cheap") says swaps on "the month after" and "the number after"
  fail in both models, and explains it by months and numbers living partly on circles. That's true
  at top-1 and α=1. But at α=2 Claude gets 6 of 12 month-after swaps, and at top-5 GPT-2 gets 8 of
  11 and Claude 4 of 11. "The number after" still fails in both (Claude 0 at α=2; GPT-2 2 of 8 at
  top-5). You may want to narrow the circle argument to numbers, or note it's about the top answer.
  Not edited.

# Round 2, 2026-09-28 (evening): judging each test by the paper's own definitions

Line numbers in this section are for the current `post.md`. The afternoon section below uses the
line numbers of the earlier version (everything after line 33 has moved down by 2 to 21 lines).
A backup of the post just before this round is in the session scratchpad
(`post_before_definitions.md`); `git diff post/post.md` still shows everything since the last commit.

## Why

The paper defines "workspace-like" by five one- or two-sentence properties (§1, "We define a subset of
vector representations as workspace-like if it satisfies the following properties"). Chalmers's
commentary quotes the same definition as the operative one. Nothing in the paper gives a stricter
version; the stricter conditions in the commentaries (Eleos's modules / bottleneck / broadcast /
selection, Chalmers's recurrence and ignition) are about the architecture, which the post already
treats as "the structure". So each test is now judged clause by clause: passed if every clause we
could test holds, with untestable clauses named. Experiments the paper runs beyond its definitions
(the injected thought) are reported but don't set the verdict.

Clause by clause (GPT-2 unless noted):

| property | clause | result |
|---|---|---|
| Verbal report | names what's in the workspace | lens-output Spearman 0.44-0.57 over the band (Claude 0.42-0.84, from the paper's data) |
| | swap changes the answer | 38 of 38 |
| | (injected thought: beyond the definition) | fails (position control) |
| Directed modulation | hold a concept in mind | fails (<1% hits; "don't think about" > "think about") |
| | mental calculation | untestable (0 of 24 sums) |
| | pulled in when the task requires it | untestable (can't answer the questions) |
| Internal reasoning | intermediate in the lens, swap redirects | 90% at layer 9; 71% of 984 swaps |
| Flexible generalization | one vector, many functions | 10 of 46 matched swaps (Claude 16); countries 9 of 15 (Claude 14) |
| Selectivity | small subset | 3-6% more variance than random |
| | needed for only some behavior | two-hop 100 to 69%, copying 98% |
| | not routine processing | ordinary text 82% unchanged (Claude 87%); language swap never moves the continuation |

Verdicts: verbal report, internal reasoning, flexible generalization, selectivity pass; directed
modulation fails. This changed the Figure 1 verbal report badge (PARTLY to WORKS) and the
selectivity heading ("mostly works" to "works").

## New results (all run this round)

1. **Claude's directed modulation rates were missing and are much higher than the post implied.**
   The paper's Fig. 10 data (`ref/paper-data/modulation-lines.json`, new): "think about" hits on
   93 / 95 / 97% of category trials and 73 / 91 / 96% of math trials (Haiku 4.5 / Sonnet 4.5 /
   Opus 4.5); "ignore" 21 / 52 / 47% and 14 / 42 / 41%. The post had said "a substantial fraction"
   and called Qwen's 25% "what Claude does".
2. **Qwen3.5-0.8B does not show the other two parts of directed modulation**
   (`results/control/qwen/dm_clauses.json`, new; `python -m jl.control --model qwen clauses`).
   It gets 15 of 24 sums right when asked directly (all the single-operation ones), but the answer to
   those 15 is on top of its J-lens in 0 of 750 "think about" trials while copying. It answers 2 of
   the 7 passage questions (British, sad), and on those the label never reaches the band lens top 10.
3. **The gap is not the band.** Qwen with layers 9-22 instead of 15-22
   (`results/control/qwen_band9-22/modulation_after.json`, first 3 sentences): focus 25% vs 23%,
   mention 21% vs 18%, math under 1%.
4. **Verbal report's first clause, with Claude's number.** The paper's Fig. 6 data
   (`ref/paper-data/verbal-report.json`, new): mean lens-output Spearman 0.42 / 0.61 / 0.84 at
   Sonnet's layers 54 / 75 / 92. GPT-2's is 0.44 / 0.47 / 0.57 at layers 7 / 8 / 9.

The takeaway shifts: GPT-2's directed modulation failure isn't only a missing-instruction-following
failure. Instruction following (at 0.8B) gets the direction of the effect, not its size, and not the
arithmetic or pulled-in parts. So there are now two tests no small model passes: directed modulation
at Claude's strength, and the injected thought.

## Every edit to post.md

### Sections you have rewritten: port these by hand

- **TL;DR bullet 2 (line 6).** Added "and judged each against the paper's own definition of the
  property". Replaced "It fails directed modulation, which asks the model to follow an instruction;
  a small instruction-tuned model, Qwen3.5-0.8B, passes it." with "It fails directed modulation,
  which asks the model to follow an instruction. A small instruction-tuned model, Qwen3.5-0.8B, gets
  that test's direction right, but at about a quarter of Claude's rate, and not for sums worked out
  in its head."
- **TL;DR bullet 3 (line 7).** "...and perhaps on the injected-thought test." became "...and
  perhaps on the two tests neither small model passes: directed modulation at Claude's strength, and
  the injected thought."
- **Figure 1** (`fig1/template.html`, rebuilt). Verbal report badge PARTLY became WORKS; its note
  "✗ reporting an injected thought" became "✗ injected thought (an extra test)". Directed modulation
  note "✓ an instruction-tuned 0.8B model follows it" became "~ instruction-tuned 0.8B: 25% (Claude:
  93–97%)".
- **Figure 1 caption (line 11).** Added "against Claude's rate from the paper's released data" after
  the Qwen sentence.
- **What we did.** Not edited. If you want the rule stated there too, one sentence would do: "We
  judge each test against the paper's own definition of the property, and count it as passed when
  GPT-2 shows every part of the definition we could test."

### Sections you haven't rewritten yet

- **How each test went, line 33 (new).** "We judge each test against the paper's own one- or
  two-sentence definition of the property, quoted in the next section."
- **Line 35 (verbal report bullet).** "The paper also injects a concept..." became "The paper also
  runs a test that goes beyond its definition of verbal report: it injects a concept...".
- **Line 39 (directed modulation bullet).** Heading "fails in GPT-2, but not in a small
  instruction-tuned model" became "fails in GPT-2, and is weak in a small instruction-tuned model".
  "Qwen3.5-0.8B follows the instruction, as Claude does. A citrus fruit is on top..." became "follows
  the instruction in the right direction: a citrus fruit is on top of its J-lens on 25% of 'think
  about' trials, and under 1% of 'don't think about' trials. But Claude is at 93% to 97%, and an
  answer Qwen works out in its head never comes out on top of its J-lens."
- **The five tests, line 73 (new paragraph).** States the rule: definitions quoted at the start of
  each section; passed = every testable part holds; experiments beyond the definitions (the injected
  thought) are reported but don't decide.
- **Definition blockquotes (new), verbatim from the paper:** verbal report line 77, directed
  modulation 87, internal reasoning 101, flexible generalization 111, selectivity 119.
- **Verbal report, line 79.** Added at the end: "The first part of the definition holds too. Just
  before the answer, the candidates' order in the J-lens tracks their order in the output (a Spearman
  correlation of 0.44 to 0.57 across GPT-2's band, against 0.42 to 0.84 across Claude's)."
- **Verbal report, line 83.** "The paper also adds a concept's J-lens vector..." became "The paper
  then goes beyond the definition, to test whether the J-lens also finds thoughts the model isn't
  about to say but could report if asked. It adds a concept's J-lens vector..."
- **Directed modulation, lines 85-97 (rewritten).** Heading as in line 39. New line 89 names the
  three parts. Line 91 (Claude) now gives the rates: 93% to 97% with "think about", a bare mention
  almost as high, 21% to 52% with "ignore", "don't think about" lowering it less (white bear). Line 93
  (GPT-2) unchanged. Line 95 (new) gives Claude's math rate (73% to 96%), describes the paired
  question test, and says both are untestable in GPT-2 (this replaces the old last paragraph). Line 97
  (Qwen) now says it gets the direction right but is much weaker than Claude, that reading more
  layers barely changes it, that the before-order result holds as before, and that it doesn't show the
  other two parts (15 of 24 sums, answer never on top; 2 of 7 passage questions, label never in the
  top 10). It ends: "So following instructions gets a small model the direction of the effect, but
  not its size, and not the rest of the property." The old ending ("So this test seems to need a
  model that follows instructions, and not much more: Qwen3.5-0.8B is about six times the size of
  GPT-2.") is gone.
- **Selectivity heading (line 117).** "Selectivity mostly works" became "Selectivity works". Body
  unchanged; the fragility paragraph stays as a caveat.
- **Why cheap, line 163.** "Directed modulation adds a third: following instructions." became
  "Directed modulation and the injected thought are different, and we come back to them below."
- **Line 167.** "The injected-thought test is the part of verbal report that isn't built in, and
  neither..." became "The injected-thought test goes beyond the paper's definition of verbal report,
  and it isn't built in. Neither GPT-2 nor Qwen3.5-0.8B passes it."
- **Line 175 (directed modulation paragraph).** Dropped "and what Qwen3.5-0.8B adds is what
  instruction following does" and the ending "So the test separates models that follow instructions
  from ones that don't, and a 0.8B model already follows them well enough to pass." New ending:
  "...and Qwen3.5-0.8B shows that pattern. But that isn't all the test asks. Claude has the category
  on top of its J-lens on nearly every trial, and the answer to a sum it was told to work out on
  most, and Qwen does neither, whichever of its layers we read."
- **Line 177.** "The one part of the tests that neither small model passes is reporting an injected
  thought only when asked. It is also the one part that isn't built into the lens, into computing in
  several steps, or into following instructions..." became "So two of the paper's tests aren't
  passed by either small model: directed modulation at Claude's strength, and reporting an injected
  thought only when asked. Neither is built into the lens or into computing in several steps, and
  following instructions isn't enough for them at 0.8B, so they may be the most informative. Our two
  small models fail the injected thought in different ways, though."
- **What this means, line 185.** "has most of these, and a 0.8B instruction-tuned model has directed
  modulation too, so for language models they are weak indicators. The exception is reporting an
  injected thought, which neither small model does." became "has four of the five, so for language
  models those four are weak indicators. The exceptions are directed modulation, which a 0.8B
  instruction-tuned model shows only weakly, and reporting an injected thought, which neither small
  model does."
- **Line 187.** "Most of the functional tests ... since GPT-2 small, or a 0.8B instruction-tuned
  model, passes them. ... and perhaps from the injected-thought test." became "Four of the five
  functional tests ... since GPT-2 small passes them. ... and perhaps from directed modulation and the
  injected-thought test."
- **Limitations, line 203.** Added "We ran the arithmetic and paired-question tests on
  Qwen3.5-0.8B instead (Appendix E)."
- **Line 204.** Added "Its directed modulation results barely change when we read more of its layers,
  but a better lens might raise them."
- **Appendix C.1, line 418 (new).** Claude's correlations from the paper's data (0.42, 0.61, 0.84).
- **Appendix C.2, line 463 (new).** Claude's Fig. 10 rates for think / ignore, categories / math,
  three models, and the at-most-2% baseline.
- **Appendix E, line 719 (new).** The other two parts of the definition on Qwen: arithmetic (15 of
  24; 0 hits in 750 focus trials; median best rank 65 vs 50 mention and 396 none) and the paired
  questions (2 of 7 answered; labels ranks 4-139 for the rest, below the foil for 2; never in the
  lens top 10 on the 2 answered).
- **Appendix E, line 721 (new).** The band check (9-22 vs 15-22).

## Code, data, and other files

- `jl/control.py`: new `clauses` experiment (arithmetic check and paired-question test on the
  instruction-tuned model) and CLI entry; docstring lists it and the band-check command.
- `results/control/qwen/dm_clauses.json` and `results/control/qwen_band9-22/modulation_after.json`
  (new; the latter overwrote nothing, that folder only had injected-thought files).
- `ref/paper-data/modulation-lines.json` and `verbal-report.json` (new), documented in its README.
- `README.md`: the `clauses` command, and what `control.py` and `ref/paper-data` now cover.
- `five_tests_draft.md`: regenerated to match the post's five-test section; sources list updated.
- `fig1/fig1.png`, `fig1/fig1.html`, `post.html`: rebuilt.

## Open decision

- The injected thought is now "an extra test, beyond the definition" for verbal report. That is the
  paper's own framing ("Next, we test whether the lens also captures thoughts that the model is not
  about to immediately verbalize"), but the Eleos commentary lists it under Report. If you'd rather
  count it, the verbal report badge goes back to PARTLY and nothing else changes.

---

# Changes, 2026-09-28 (afternoon session)

Nothing is committed. The exact record of every edit to the post is `git diff post/post.md`
(`git diff --word-diff post/post.md` is easier to read). This file groups the edits by section and
says why each was made. Sections you have already rewritten in your own draft (TL;DR, Background,
What we did, The structure) are listed first, with old and new text, since those need porting by hand.

## What changed in the results

1. **Flexible generalization.** The old 35% (19 of 54) counted 8 swaps where the target's answer
   was already GPT-2's top output before the swap (GPT-2 says "seven" to every month-number prompt,
   "forty" to every squaring prompt). Now scored only on swaps that could change the answer, and
   compared with Claude on the same swaps, from the paper's released per-swap grid: GPT-2 10 of 46
   (22%), Claude 16 of 46 (35%), and the two match function by function. The main operation is now
   the coordinate swap, which the paper's Fig. 68 names (subtract-and-add is also given: 11 of 46).
   Under the coordinate swap, France to China no longer gives Beijing and Yuan (it gives Shanghai
   and Yen), so the example is now Canada to France (Toronto to Paris, English to French).
2. **Directed modulation.** Rerun in the paper's own form (the instruction names a category like
   "citrus fruits", every member is tracked, a hit is a member at lens top 1), with all 24 released
   phrasings. GPT-2: hits under 1% in every condition, and "don't think about" raises the category
   more than "think about" in all five prompt frames we tried. Qwen3.5-0.8B (instruction-tuned) on
   the paper's chat prompt: hits 17% with a mention, 25% with "think about", 3.5% with "ignore",
   0.6% with "don't think about", the pattern the paper reports for Claude.
3. **Injected thought.** Rerun with the paper's released protocol (its strengths, the rank at the
   open quote, the other reply positions as the control). GPT-2 looks like Claude at the report (the
   word in its top 5 for 40 of 57 concepts; Claude 89%), but the word comes out at other points of
   the reply as often or more. Qwen3.5-0.8B shows no effect at all in 12 configurations (raw or
   centered vector, bare or word-initial token, three bands).
4. **Internal reasoning.** Fixed an off-by-one: "outside the top 10" was coded as rank >= 10. Now
   984 swaps instead of 999; rates barely move (70.6%; privilege 94.1% / 14.0%).
5. **Selectivity, language test.** Now as in the paper: the swap covers only the question tokens
   after the passage (before, it covered the passage). GPT-2 names the swapped-in language on 12 of
   21 trials and continues in it on 0 of 24. It can't do the paper's author task.
6. **Why cheap, selectivity.** The argument that the skip rule protects copying and recall was
   wrong at k = 1 (without the rule, one-hop and two-hop both lose about 20 points, and copying
   doesn't change). Replaced with an argument the data supports.
7. **Qwen3-1.7B is replaced by Qwen3.5-0.8B everywhere**, run faithfully (every band layer, the
   paper's strengths and metric, the category form, all phrasings).

## Sections you have rewritten: port these by hand

### TL;DR (`post.md` lines 6-7)

Bullet 2, old: "It passes verbal report, internal reasoning, flexible generalization, and
selectivity about as well as Claude does, and it partly passes directed modulation. It has almost
none of the structure."
New: "It passes verbal report, internal reasoning, flexible generalization, and selectivity about as
well as Claude does. It fails directed modulation, which asks the model to follow an instruction; a
small instruction-tuned model, Qwen3.5-0.8B, passes it. Neither small model can report an injected
thought. GPT-2 has almost none of the structure."

Bullet 3: "We think these functional tests are cheap" became "We think most of these functional
tests are cheap", and "The case for a workspace in Claude has to rest on the structure" became "...
has to rest on the structure, and perhaps on the injected-thought test."

For your version: "and partly passes directed modulation" should change along the same lines, and
so should "the case for Claude's would have to rely on the structure". Your first bullet still has
the typo "five functional tests for a global workspace paper".

### Figure 1 (`post/fig1/fig1.png`, built from `post/fig1/template.html` by `post/fig1/build.py`)

- Verbal report: badge WORKS became PARTLY, with a new line "✗ reporting an injected thought".
- Flexible generalization: the example is now Canada to France (Toronto to Paris, English to French,
  Canada to Europe), all real coordinate-swap outputs. It was France to China (Paris, French, Franc
  to Beijing, Chinese, Yuan), which only holds under the subtract-and-add swap, and "Franc" was
  GPT-2's wrong answer for France's currency.
- Directed modulation: badge PARTLY became FAILS; the prompt now names "citrus fruits"; the lines are
  "✗ a citrus fruit on top: <1%", "✗ "don't think about": ↑↑", and "✓ an instruction-tuned 0.8B
  model follows it".
- Caption (line 11): added "The last line of the directed modulation card is Qwen3.5-0.8B, a small
  instruction-tuned model (Appendix E)."
- The old image is not kept in the repo; `git show HEAD:post/fig1/fig1.png > old.png` recovers it.

### What we did (lines 51 and 63)

- End of the paragraph after the "Name a sport" block, added: "For the two tests that ask the model
  to follow an instruction, we also ran the paper's own chat prompts on a small instruction-tuned
  model, Qwen3.5-0.8B, with its J-lens from Neuronpedia (Appendix E)."
- End of the centering paragraph, added: "Qwen3.5-0.8B's are much less aligned (an average cosine of
  about 0.1)."

### The structure (lines 129 and 137)

- Persistence paragraph: "Qwen3-1.7B's persistence is short-lived in the same way" became
  "Qwen3.5-0.8B's ...". Your version doesn't mention Qwen, so nothing to port.
- MLP paragraph: "In GPT-2 it amplifies them only 1.2 to 1.5 times as much" became "In GPT-2's band it
  amplifies them only 1.2 to 1.4 times as much" (layers 7-9 are 1.23, 1.32, 1.36; the 1.54 is layer
  10). Your version says "1.2-1.5x in the band" and needs the same fix. The overview's structure
  bullet (line 38) got the same fix.
- From last night, still open in your version: "drop the top 5 vectors" should be "top 5 principal
  components"; "only slightly" and "do not appear at all" contradict each other (the appendix says
  no block either way); "Claude holds about six across its 55" (55 is percent of depth, not layers);
  typos "and and" and ",,".

### Background

No changes.

## Sections you haven't rewritten yet

### How each test went (lines 33-38)

All five test bullets were rewritten with the new results, and their headlines changed:
- "Verbal report works, but not the injected thought."
- "Flexible generalization works where GPT-2 can do the task." (was: "... works"; the example is now
  Canada to France; "35%, against Claude's 40%" became "22% of them work, against 35% for Claude on
  the same swaps")
- "Selectivity works." (copying "98%" became "39 of 40"; added the language-test sentence)
- "Directed modulation fails in GPT-2, but not in a small instruction-tuned model." (was: "partly
  works"; now about citrus fruits, the paper's form, with the Qwen numbers)
- Internal reasoning is unchanged. The structure bullet has only the 1.2-1.4 fix.

### The five tests in more detail (lines 69-113)

All five subsections are replaced. The same text, with a list of where every number comes from, is
in `post/five_tests_draft.md`, which replaces last night's version. The headings are now claims, as
in your structure section:
- **Verbal report works, but the injected thought doesn't.** The injected-thought paragraph is new
  (the paper's protocol). Figure 3 is no longer shown; it was built from Qwen3-1.7B and the old
  metric, and `post/figures/fig3_introspect.png` is now stale.
- **Directed modulation fails in GPT-2, but not in a small instruction-tuned model.** New: the
  paper's category form and hit measure, the four extra prompt frames, the Qwen3.5-0.8B result, and
  the note that GPT-2 can't answer the paired questions Also the caveat that with the instruction placed before the request, Qwen's
  "think about" no longer beats a bare mention (its downward control holds in both orders).
- **Internal reasoning works.** 999 became 984; privilege 94% / 14% (was 15%).
- **Flexible generalization works on the swaps GPT-2 can do.** New scoring and the Claude
  comparison on the same swaps.
- **Selectivity mostly works.** No table; four paragraphs plus one on the "small part" clause. The
  language paragraph uses the new question-token swap. The skip-rule paragraph is gone.

### Why the tests are cheap (lines 145-165)

- Intro: added "Directed modulation adds a third: following instructions."
- Verbal report paragraph, last sentence: "and GPT-2 fails it" became "and neither GPT-2 nor
  Qwen3.5-0.8B passes it".
- Flexible generalization paragraph: "The failures fit too." became "The failures fit too, and they
  are the same in both models. Swaps on "the month after" and "the number after" fail in GPT-2 and
  in Claude, and Claude's fail on every function of a number word."
- Selectivity paragraph: rewritten without the skip-rule argument. It now says the word to be copied
  is far down the J-lens at the ablated layer (median rank 211), while the one- and two-hop answers
  are the top J-lens token and protected by the rule in both.
- Directed modulation paragraph: rewritten. GPT-2 shows what next-word prediction does, and Qwen
  shows what instruction following adds; the test separates instruction-tuned models from base
  models, and a 0.8B model passes.
- New paragraph before "The structure GPT-2 does have...": the injected thought is the one part of
  the tests neither small model passes, and may be the most informative; the two models fail it
  differently.

### What this means (lines 167-175)

- "A 124M base model from 2019 mostly has these, so ..." now also mentions the 0.8B model and the
  injected-thought exception.
- "The five functional tests, as the paper runs them, don't tell apart ..." became "Most of the
  functional tests ..."; "the evidence has to come from the structure" gained "and perhaps from the
  injected-thought test".
- Citation: Goldstein and Kirk-Giannini (2024, arXiv) became (2026, the published JCS version, DOI
  10.53765/20512201.33.7.061, per Chalmers's bibliography).

### Limitations (lines 177-190)

- The injected-thought bullet now says the fixed reply is the paper's released protocol.
- The "couldn't run" bullet adds the paired-question test and the language-switch detection task.
- The Qwen bullet is replaced: one small model with a band from its layer statistics, and GPT-2's
  failures on the instruction tests may be failures of our prompts.
- "We used one model and one lens, and didn't fit our own lens" became "Almost everything is on one
  model with one lens, and we didn't fit our own lenses."

### Appendix

- **A.1:** the subtract-and-add swap is now "verbal report; also given for flexible generalization";
  the coordinate swap is now "internal reasoning, flexible generalization, the language test".
- **A.2:** the two-hop row is now 984 trials (same percentages).
- **A.3:** the two-hop privilege row is now 984 trials, 94% / 14% and 95% / 1.4%.
- **A.4:** band 8-10's two-hop swap is 67% (was 66%).
- **B.1:** the injected-thought protocol is rewritten (the paper's strengths, the three frames, raw
  and centered vectors, bare and word-initial tokens).
- **B.2:** rewritten for the category form and the five frames; the earlier named-word version is
  described in one line; the paired-question capability prompt is added.
- **B.5:** the language test is rewritten (three tasks, swap over the question tokens).
- **C.1:** the injected-thought table is replaced (the paper's metric at its strengths), plus the
  other configurations and Qwen.
- **C.2:** rewritten. There is a new main table (the category form) and a table of paired
  comparisons in each frame. The earlier named-word tables are kept as "the earlier version". The
  old "main-text version" line is removed, since the new table replaces it. Also added: GPT-2 can't
  answer the paired questions (ranks 69 to 3,043, below the foil on 5 of 7).
- **C.3:** 43/48 and 36/43 became 41/48 and 36/41; the swap is now 984 trials (70.6%), with 70.9%
  of 611 for capital to language, a median rank of 90 to 1, and a note on target selection. The
  subtract-and-add rate is 85.7% (was 85.5%). The depth table has new values (31 items, was 32),
  and there is a note that the paper swaps over layer ranges. The privilege table is new (984
  trials).
- **C.4:** rewritten. There is a new table with Claude on the same swaps, per-function results,
  the note on the earlier inflated version, the Canada to France example, and the France to China
  note. The loading correlation is now -0.11 over 25 prompts (was -0.38 over 31).
- **C.5:** the language table is replaced (question-token swap); the old result is kept in one line.
- **E:** rewritten for Qwen3.5-0.8B (model, lens, band, layer stats, verbal report, injected
  thought, directed modulation) in both instruction orders).

## Code and results

- `jl/control.py` (new): the injected thought and directed modulation as the paper runs them, on
  GPT-2 (several base-model frames) and on Qwen3.5-0.8B; also Qwen's verbal report and layer stats,
  and the paired-question capability check. Results are in `results/control/{gpt2,qwen,qwen_band14-22,qwen_band9-22}/`.
- `jl/c4_generalization.py`: the summary scores the testable swaps, adds Claude on the same swaps
  from `ref/paper-data/flex-gen-appendix.json` (new; the paper's released Fig. 68 data), and makes
  the coordinate swap primary. The case study runs both operations, and the floor check uses the
  coordinate swap. Rerun: `results/c4_generalization/`.
- `jl/c3_reasoning.py`: the top-10 off-by-one is fixed in every place. Rerun, including `swap_ops`.
- `jl/c5_selectivity.py`: new `language_tasks` (question-token swap; three tasks) and a
  `python -m jl.c5_selectivity language` mode that merges it into the existing results. The old
  passage-swap result is kept as `language_passage`. The summary handles string keys after JSON.
- `jl/followups.py`: the same off-by-one in `variants`; `variants` and `bands` are rerun.
- `post/figures/fig2_centering.png`: rebuilt from the rerun `variants.json` (no visible change).
- `README.md` and `jl/__init__.py`: list the new module and commands.
- Superseded but not deleted: `jl/qwen_control.py`, `results/qwen_control/`, `results/introspect_probs/`,
  and `post/figures/introspect.py` / `fig3_introspect.png` (Qwen3-1.7B and the old metric).

## Open decisions for you

- The title "GPT-2 Small Mostly Has a Functional Global Workspace" still fits (GPT-2 passes four of
  the five tests by the paper's definitions). (Superseded in round 2: directed modulation does not
  simply come with instruction tuning; see the top of this file.)
- Hoel's post names "someone at Prime Intellect", not Elie Bakouch (the link presumably does).
- `five_tests_draft.md` duplicates the post's five-test section; delete it once you've used it.
