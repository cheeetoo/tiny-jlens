# Notes on the draft

## Reading and rebuilding

- `post/post.md` is the source. `post/post.html` is a local preview made from it, with the figures inline. After editing the Markdown, run `post/build.sh` (needs pandoc) and open `post/post.html` in a browser.
- Figure 1 is `post/fig1/fig1.html`. Render it to `fig1.png` with:

  ```
  cd post/fig1 && /Applications/Chromium.app/Contents/MacOS/Chromium --headless=new --disable-gpu --hide-scrollbars \
    --force-device-scale-factor=2 --window-size=1200,965 --screenshot=fig1.png "file://$PWD/fig1.html"
  ```

- Figures 2 to 4 come from `post/figures/centering.py`, `introspect.py`, and `structure.py` (run them from the repo root with `.venv/bin/python`). I renamed the outputs so the numbers match the order in the post: `fig3_introspect.png` and `fig4_structure.png`. The old `fig3_structure.png` and `fig4_introspect.png` are deleted.
- `paper.md` (local only) is the paper. `commentary/` (local only, except `README.md`) has the commentaries. `commentary/README.md` is a sourced summary of what's online.

## What changed in this pass

- The TL;DR is three bullets, with the interpretation in one: the tests are cheap, so passing them should count for little. It links Rob Long's post.
- New order: TL;DR, Figure 1, background, "How each test went" (a reader can stop there), a short setup, the tests in detail, the structure, why the tests are cheap, what this means, limitations, appendix.
- Background:
  - The J-lens gets three plain sentences.
  - The structural signatures are motivated with the Eleos "privileged set" point.
  - The review summary is cut. One quote is left (Eleos), followed by the paper's "Future work could investigate..." line and "We ran the tests on GPT-2 small."
- Setup:
  - It is one paragraph plus the Name-a-sport example, with no list of interventions and no mention of the earlier replications.
  - Centering is one paragraph.
  - The band is one short paragraph.
- Qwen is folded into verbal report and directed modulation (details in Appendix E), not given its own section.
- The structure section explains kurtosis, occupancy (2 to 5 J-lens vectors in GPT-2, against about 25 in Claude), and how we chose the heads.
- Limitations are plain sentences.
- Leech's "near-analytic" point is now cited in "Why the tests are cheap".
- The appendix has no file or repo references. The experiments table and the "Reproducing" section are gone.
- Length: the body is about 6,100 words (it was about 8,000) and the appendix about 6,200. If it needs to be shorter:
  - The per-test detail sections could each lose a paragraph.
  - The heads subsection could go to the appendix.

## Verdicts

- **Verbal report: works.** The swap works on all 38 trials. The injected-thought version fails. That is said in the same bullet, not framed as "the harder version".
- **Internal reasoning: works.** 71% against Claude's 70%. See the caveat below.
- **Flexible generalization: works.** You were right to doubt "partly". GPT-2 gets 35% on the swaps whose target function it can compute, and Claude gets 40% on the full grid. Both fail one kind of function: Claude fails number words (0 of 48), and GPT-2 fails the two successor functions. Doubling the strength hurts GPT-2 because the swap then makes it output the swapped-in argument itself. The post mentions this in one sentence and doesn't lean on it.
- **Selectivity: partly.** The pattern matches the paper (two-hop breaks, copying survives). But the ablation also changes 29% of ordinary next-token predictions (random: about 19%) and wipes out one-hop recall at three layers. Part of the pattern comes from the rule that skips the model's likely next tokens.
- **Directed modulation: partly.**
  - "Think about" beats a bare mention in 76% of pairs, but the effect is small.
  - "Ignore" doesn't lower the word. I frame that as a difference from Claude rather than a clear failure. The criterion as stated asks only for bringing a concept in, and there is the white-bear point.
  - Qwen3-1.7B passes in both directions.
- **Structure: "almost none".** You said "none of the structure is there at all". I kept "almost" because GPT-2 does have a limited capacity, the category-block effect in lists, and heads that copy J-lens directions. The post gives the simple explanations for each.

## Checks in this pass

- **Internal reasoning comparison.**
  - The paper's 70% (Sonnet 4.5 and Opus 4.5) is on its own 50 two-hop questions of many kinds.
  - GPT-2 answers only 9 of the paper's 90 released two-hop questions. Four of those 9 are all of the set's language-to-capital questions.
  - We wrote 53 more questions of that kind, and GPT-2's 71% is on those.
  - So the two rates come from different questions. The post says this in the "How each test went" bullet and again in the section.
  - The kind of question is one the paper uses, so it isn't unfaithful. But it isn't a like-for-like comparison, and a red-teamer will push on it.
- **Injected-thought numbers.** I recomputed them all from `results/introspect_probs/`.
  - At the strongest setting, the word is GPT-2's top prediction at the report for 41 of 57 concepts, and at the start of the reply for 50 of 57.
  - At the two strongest settings, it is more likely at the start than at the report for 53 and 55 of 57.
  - The start of the reply means the positions after "Model", "Model:", and "Model: Yes". Using only the position after "Model:" (the first word of the reply), it is still top-1 for 49 of 57 at the strongest setting. So the speaker label isn't doing the work.
  - Qwen: the word is never top-1 at the start of the reply, and top-1 at the report for at most 6 of 74. At most 8 of 74 get a 10% chance at the report at any one strength, and 11 across all strengths. I fixed the post, which had said "at most 8 ever".
- **Quotes.** I checked every quote against `paper.md` and the commentary texts: Eleos, "Future work could investigate...", "by construction, we should expect...", "when the model's introspective report is being elicited", "several of the key functional properties...", Leech's "near-analytic", and Nanda's "notably less clean than the paper's". I made the Nanda and Hoel sentences more precise.

## Caveats on the injected-thought result (for the red-team)

- It uses a prefilled transcript, not free generation. We read next-token predictions at each position of a fixed reply. We never ran the free-generation version (sample a reply and see whether GPT-2 says the word early). A KV-cached version was written before the Wi-Fi cut out, but it never finished.
- Positions inside the answer sentence ("an ___", "an injected ___", "The ___") are ambiguous, because saying the word there could count as naming the thought early. The post uses only the start of the reply. Appendix C.1 has the in-sentence positions too.
- Strength is a multiple of the layer's mean residual norm on the prompt. At 0.5 and 1.0 the injection is large.
- The paper injects over the whole user turn and samples replies. We inject over the researcher's line.
- Qwen's band and strengths weren't tuned, so it might report more at other settings.

## For the figures Claude

- **Figure 1** was rebuilt from your notes as a starting point:
  - no title;
  - green WORKS, gold PARTLY, and gray MISSING tags;
  - the structure shown as three small panels (band, ignition, MLP gain).
  The Claude panels are stylized from the paper, not data.
- **Figure 2 (centering)** should shrink to (a) the schematic and (b) raw vs. centered cosine bars. Panel (c) duplicates the table in A.2.
- **Figure 3 (injected thought)** is fine. It could add the in-sentence positions as a lighter line.
- **Figure 4 (structure).** You wanted the paper's CKA figure next to GPT-2's.
  - GPT-2's plain linear CKA is 0.94 or higher for every pair of layers 0 to 10, because one principal component dominates even after centering.
  - So the GPT-2 panel has to use CKA after removing the top 5 components, or mean canonical correlation, and the caption should say so. Otherwise it looks like one big block.

## Open questions for you

- **Title.** Alternatives: "GPT-2 Small Passes Most of the Global Workspace Tests" and "The Global Workspace Tests Are Cheap: GPT-2 Small Passes Most of Them".
- **Quote.** I kept the Eleos quote. The other option is Dehaene and Naccache calling the paper "a landmark in consciousness research".
- **Structure.** "Almost none" or "none"?
- **Citations I didn't read directly:** Herzog et al. 2007, Butlin et al. 2026, and Goldstein and Kirk-Giannini 2024. They came from secondary sources, so check them before publishing.
- **Hoel.** His point rests on Bakouch's plots, which Hoel himself calls early and not his own.
- **Eleos.** Do you want to send the draft to them (Rob Long) before posting? It engages with his "Merely functional" post and quotes their commentary.

## Things we didn't run that we probably should

1. The free-generation injected-thought test. This matters most, since it's the one functional result small models don't reproduce.
2. A scale ladder: GPT-2 medium and large, and Pythia. Which of the missing pieces (suppression, holding a thought without saying it, ignition, the band) appear with scale?
3. Another model and lens, such as Pythia-160m or 410m. tao-hpu found the J-lens beats the logit lens on Pythia-70m but not on OpenAI's GPT-2.
4. A toy model: a 2 to 4 layer transformer trained on a synthetic two-hop lookup. If a toy passes reasoning, generalization, and report, that's the cleanest "the tests are cheap" demonstration.
5. The Qwen checks with a tuned band and strengths, and the other three tests on Qwen.
6. A two-concept version of the paper's dual-task (competition) test.
7. KL divergence instead of top-1 agreement on ordinary text, and more than 16 paragraphs.

## Slack message draft

> New draft: we ran the tests from Anthropic's global workspace paper on GPT-2 small (124M, 2019). It passes verbal report, internal reasoning, and flexible generalization about as well as Claude, partly passes selectivity and directed modulation, and has almost none of the structure (no band of workspace layers, no ignition, little MLP amplification). Our take is that the five functional tests are cheap, so passing them is weak evidence for a workspace, and the case for Claude rests on the structural results. Draft: [link]. Comments welcome, especially on the injected-thought test and the internal-reasoning comparison.

## Tweet thread draft

> 1/ Anthropic's global workspace paper found that Claude's "J-space" passes five functional tests of a global workspace. We ran the same tests on GPT-2 small, a 124M model from 2019. It mostly passes them. [Figure 1]

> 2/ Verbal report, internal reasoning, and flexible generalization work about as well as in Claude. Swapping France for China in GPT-2's J-space turns "Paris" into "Beijing" on 71% of swaps (Claude: 70%, on harder questions).

> 3/ Selectivity and directed modulation partly work. Told to "ignore lemon", GPT-2 doesn't push "lemon" down the way Claude does. A small chat model, Qwen3-1.7B, does, so that part seems to come with instruction tuning.

> 4/ What GPT-2 lacks is the structure: no distinct band of workspace layers, no "ignition" (a France/Germany blend stays a blend), and MLPs amplify J-lens directions 1.2–1.5x, against ~10x in Claude.

> 5/ We think the functional tests are cheap. A functional global workspace would be a big deal, but a 124M base model mostly passes these tests, so passing them is weak evidence for one. The case for a workspace in Claude rests on the structure.

> 6/ Post: [link]. Code: [link].

## Re-run check (earlier session, all criteria on CPU)

- c1 verbal report: identical.
- c2 directed modulation: identical, apart from one median rank that is off by 1 (floating point).
- c3 internal reasoning: identical, apart from a few ranks that are off by 1.
- c4 flexible generalization: re-run with the corrected squaring frame ("thirty-six", not "thirty"). The capable-subset rate went from 22/57 (39%) to 19/54 (35%). The post uses the new numbers.
- c5 selectivity: the J-space ablation numbers are identical. The single-draw random controls differ from the old results, which were probably made on a GPU with different random draws, so the post reports five-seed ranges.
- Band statistics, CKA, and MLP gain: identical.
