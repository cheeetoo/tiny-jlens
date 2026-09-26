# Notes on the draft

## Files

- `post/post.md` is the post plus appendix.
- `post/fig1/fig1.html` is Figure 1. `post/fig1/fig1.png` is the render (Chromium headless, 2×).
- `post/figures/fig2_centering.png` is Figure 2 (script: `post/figures/centering.py`).
- `post/figures/fig3_structure.png` is Figure 3 (script: `post/figures/structure.py`).
- `paper.md` (local only, gitignored) is the paper, converted from the HTML. Interactive figures are missing, but the captions are there.
- `commentary/` (local only, except `README.md`) has the invited commentary, Leech, Chalmers, and Hoel. `commentary/README.md` is a sourced summary of everything I found online, with links. The repo is public, so the third-party texts are not pushed.
- `jl/followups.py` has the new experiments. Results are in `results/followups/`.

Re-render Figure 1 with:

```
/Applications/Chromium.app/Contents/MacOS/Chromium --headless=new --disable-gpu --hide-scrollbars \
  --force-device-scale-factor=2 --window-size=1200,985 --screenshot=post/fig1/fig1.png file://$PWD/post/fig1/fig1.html
```

## What I checked

- I re-ran every criterion module on CPU. c1 and c2 match the old results exactly (one rank differs by 1). c3, c4, and c5: see the bottom of this file.
- The matched-random control in the ablation is a single random draw. The old results were probably made on a GPU, which draws different random numbers, so the old and new values differ (for example, two-hop at one layer: 83% vs. 92%). I added a five-seed version (`python -m jl.followups randseeds`). The post now reports ranges.
- One flexible-generalization demo said "Six squared equals thirty". I fixed it to "thirty-six" before the c4 re-run.

## New experiments (all in `jl/followups.py`)

1. **Injected thought** (`introspect`). This is a base-model version of the paper's Fig 7. GPT-2 reports the injected word once the injection is strong enough, but at every strength it blurts the word out at other positions first. So it fails the harder verbal-report test. This changed the story: the tests GPT-2 fails are the ones that need control over the J-space.
2. **Protection rule** (`protect`). This is the ablation with and without the paper's "don't ablate the clean top-10 tokens" rule. At three layers, copying survives only because of the rule (65% vs. 25%). Nobody online has made this point, as far as the search found.
3. **Ignition** (`ignition`). An embedding mixture of two countries stays a near-linear blend at every layer. There is no ignition.
4. **Lists** (`lists`). The J-space holds about 1 unrelated word at a time (paper: about 6), and about 10 of an 80-word category (paper: nearly all of it).
5. **Broadcast heads** (`heads`). J-specific copying heads exist, including the IOI name movers 9.6, 9.9, and 10.0, but they occur at every depth (the most in block 3). Ablating the top 3 has a small effect.
6. **Neuron composition** (`neurons`). This rises smoothly with depth. There is no band.
7. **Raw vs. centered vs. logit lens** (`variants`). Centering matters for the subtract-and-add swap and the ablation, but not for the coordinate swap. The logit lens is worse than the J-lens on the two-hop swap (45% vs. 71%), so the J-lens does add something on GPT-2 small, even though it doesn't win on readout.
8. **Band sensitivity** (`bands`). The results hold for any band from layer 6 on.
9. **Linear check** (`linear`). The non-J remainder is almost orthogonal to v_t − v_s (median cosine 0.02). So the J-vs-non-J "privilege" test is close to built in.
10. **Random-control seeds** (`randseeds`).
11. **Category blocks** (`blocks`). This is the paper's Fig 31E/F. A new category pushes the old one out of GPT-2's J-space (21% of a block's words present during the block, 1% during the next), the same as in Claude. The explanation is cheap: the J-lens reads what the model expects to say, and in a list it expects the current category.
12. **Qwen3-1.7B control tests** (`jl/qwen_control.py`). The injected-thought test, directed modulation (ignore vs. mention), and the verbal-report swap, run on a small instruction-tuned model. **Qwen passes directed modulation in both directions**: focus > mention in 94% of pairs, and ignore < mention in 72%. So the "downward control" GPT-2 lacked comes with instruction-following at 1.7B. On the injected-thought test, Qwen doesn't blurt the way GPT-2 does, but it also barely reports the thought. That makes the injected-thought test the one functional result that neither small model reproduces. This is a quick check. The band and strengths are not tuned for Qwen, and the verbal-report swap is weak there (43%), which suggests the settings could be better.
13. **Injected thought in probabilities** (`jl/introspect_probs.py`). This checks that "in the top 10" isn't a low-probability tail. At strength 0.25, the word is GPT-2's top-1 prediction somewhere earlier in the line for 53 of 57 concepts (median peak probability 0.30), against a median of 0.009 at the report. Qwen almost never predicts it early, but it also rarely reports it.

## Calls I made that you might want to change

- **Verdicts.** Verbal report: works (the swap works, the injected-thought version fails). Internal reasoning: works. Selectivity: mostly. Flexible generalization: partly. Directed modulation: weak/mostly fails. You guessed three full passes, one partial, and one fail. I ended at two full passes, one mostly, one partly, and one mostly fails. "Mostly has a functional global workspace" still fits.
- **Title.** I kept yours. The alternatives are at the bottom of the post. My favorite is "GPT-2 Small Passes Most of the Global Workspace Tests", because it says what we found without claiming GPT-2 *has* anything.
- **The main argument.** The tests GPT-2 passes are the ones the J-lens and next-word prediction give for free. The ones it fails need control (hold without saying, suppress, summon) or real structure. The Qwen check then shows that "suppress on instruction" comes cheaply with instruction tuning. That leaves the injected-thought test (plus the structure) as the part of the paper's evidence that small models don't reproduce. I think this is a cleaner and more defensible message than "the tests are cheap" alone, because the failures back it up. You may want to make the Qwen result more prominent, since it closes the obvious objection that GPT-2 is just a base model.
- **Numbers in the body.** I tried to keep only the numbers that compare directly with the paper (for example 71% vs. 70%). Everything else is in the appendix. The directed-modulation and selectivity sections are still somewhat numbery.
- **Length.** The body is about 8,000 words and the appendix about 6,000. If you want it shorter, the easiest cuts are: (1) the five test sections, which repeat the "How each test went" bullets with more detail and could move to the appendix; (2) the structure subsections, which could collapse into Figure 3 plus one paragraph; (3) the Qwen section, which could go to the appendix.
- **Figure 1.** The mascot is a generic pixel robot, deliberately not the paper's creature. The colors are SLR teal, gold, and gray for works, partly, and weak.

## Things we didn't run that we probably should

1. **A scale ladder.** GPT-2 medium and large (the git history says earlier versions ran them), plus Pythia-70m through 1.4B, whose J-lenses are on Neuronpedia for 70m. Which of the failed pieces (suppression, holding without blurting, ignition, band) appear with scale? That is the most interesting follow-up.
2. **A self-fit lens, or a model without tied embeddings.** tao-hpu found the J-lens beats the logit lens on Pythia-70m and on a self-trained GPT-2 124M, but not on OpenAI's GPT-2. Rerunning our battery on Pythia-160m/410m would rule out "GPT-2's lens is weird".
3. **A toy model.** Train a 2–4 layer transformer on a synthetic two-hop lookup task, fit a J-lens, and run reasoning, generalization, and report. If a toy passes, that is the cleanest possible "the tests are cheap" demonstration.
4. **Competition / dual task.** The paper's appendix dual-task experiment (concept plus arithmetic) can't run on GPT-2, but a two-concept version could. This is the one capacity test that isn't built in.
5. **Selectivity without the protection rule on a big model.** The paper's Fig 22/24 battery with the rule removed. We can only do the GPT-2 version.
6. **More ablation battery items.** The text-prediction measure uses only 16 paragraphs. KL divergence would be a less noisy measure than top-1 match.
7. **A multimodal / non-verbal check** (Chalmers's prediction). Out of scope, but worth mentioning.

## Open questions for you

- Do you want the Qwen3-4B "not selective" result and the injected-thought failure more prominent? They are the two strongest "not cheap / fails" results.
- The "What this means" section cites the small-network / minimal-implementation literature. I haven't read Herzog 2007 or Butlin 2026 directly; the citations come from Michel's paper and the research summary. Worth checking before publishing.
- Should we contact the Neuronpedia / tao-hpu / Bandularatne folks? Our logit-lens comparison partly disagrees with "failed replication" framings: the J-lens loses on readout but wins on interventions.

## Re-run check (all criteria, CPU)

- c1 verbal report: identical.
- c2 directed modulation: identical, apart from one median rank that is off by 1 (floating point).
- c3 internal reasoning: identical, apart from a few ranks off by 1.
- c4 flexible generalization: re-run with the corrected squaring frame, so a few numbers changed. The capable-subset rate went from 22/57 (39%) to 19/54 (35%), and the capability gate from 20/64 to 19/64. The post uses the new numbers.
- c5 selectivity: the J-space ablation numbers are identical. The single-draw random controls differ from the old results, a lot at medium and heavy strength (for example, two-hop at medium: 0.31 old vs. 0.75 new). The post reports five-seed ranges instead. S1 variance shares changed in the first decimal place.
- band statistics, CKA, and MLP gain: identical.

The re-run overwrote `results/c*/`. `git diff results/` shows what changed. The old versions are in git.

## Draft tweet (goes with Figure 1)

> Anthropic's global workspace paper tests whether a model's "J-space" passes five functional tests of a global workspace. We ran them on GPT-2 small (124M, 2019). It passes most of them: verbal report and internal reasoning work about as well as in Claude. What it fails are the tests that need control (hold a thought without saying it, suppress on instruction) and the structural signatures. Our read: the functional tests are cheap. [link]
