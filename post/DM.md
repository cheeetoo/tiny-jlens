# Directed modulation

The post's directed-modulation result and its Appendix E: what the paper does, what we ran on GPT-2 small and
on Gemma 3 (270m and 1b, each base and instruction-tuned), and the numbers. The figures are
`post/figures/dm_models.py` (every model, the paper's score) and `post/figures/band_gemma.py` (how each
Gemma band was chosen); `post/figures/dm_gemma.py 270m|1b` and `post/figures/dm_tables.py` print the tables.
(An earlier version of this work also ran Qwen3.5-0.8B, four more base-model prompt frames for GPT-2, and
Lindsey (2025)'s named-word version of the test. The post no longer uses them; they are in the git history.)

## In short

On the paper's score (a category member at J-lens rank 1 at some band layer and some token of the copied
sentence, mean over phrasings):

| | no instruction | think about | mention | don't think about | ignore |
|---|---|---|---|---|---|
| GPT-2 small (layers 7–9) | 0.0% | 0.5% | 0.5% | 0.5% | 0.5% |
| Gemma-3-270m (6–11) | 0.0% | 0.2% | 0.2% | 0.2% | 0.2% |
| Gemma-3-270m-it (6–11) | 0.2% | 0.5% | 0.5% | 0.5% | 0.5% |
| Gemma-3-1b (11–16) | 0.7% | 1.7% | 1.8% | 1.2% | 1.3% |
| Gemma-3-1b-it (11–16) | 2.0% | 40.8% | 24.4% | 2.0% | 11.2% |
| Claude (Haiku, Sonnet, Opus 4.5) | 1.6%, 1.4%, 1.6% | 93%, 95%, 97% | 66%, 86%, 87% | 65%, 86%, 83% | 21%, 52%, 47% |

- Only the instruction-tuned 1B model shows the effect: "think about" above a mention, and "ignore" and "don't
  think about" below one. It is weaker than in Claude, there is no white-bear effect ("don't think about" sits at
  the no-instruction level), and it barely shows for mental arithmetic.
- So in small models, directed modulation seems to need instruction tuning and some scale.

## 1. What the paper does

### The prompt

The released protocol (`ref/jacobian-lens/data/experiments/directed-modulation.json`) has the materials but no prompt template. The template is in the data behind the paper's Figure 9 (`ref/paper-data/modulation-readout.json`, from `data/modulation-readout/modulation.json` on the paper's page):

```
Human: Write "The old painting hung crookedly on the wall." Concentrate on citrus fruits while you write the sentence. Don't write anything else.

Assistant: The old painting hung crookedly on the wall.
```

So the user turn is `Write "{sentence}" {instruction} Don't write anything else.`, and the reply is the sentence, forced. The same template appears in the data behind Figure 46 (`metacog-alarm.json`). With no instruction, the turn is presumably `Write "{sentence}" Don't write anything else.` The paper doesn't show it.

### The materials

- **22 topic categories.** The instruction names the category (`citrus fruits`), and every listed member is tracked (`orange`, `lime`, `mandarin`). There are 162 members in all.
- **24 math problems** in four tiers, from `4 * 2` to `5^2 - 19`. The instruction names the expression, and the answer is tracked. The protocol lists the answer as a digit (`7`). The paper's Figure 9 example counts the number word (`seven`), and its two-tasks protocol says "digit or number-word", so we track both.
- **20 carrier sentences**, 9 to 16 tokens long in our models' tokenizers.
- **24 instruction phrasings** in four groups: 5 focus ("Think about {x} while you write."), 6 mention ("{x}."), 6 dismissal ("Ignore {x}."), and 7 negated-think ("Don't think about {x}."). The paper's Figure 65 has two more focus phrasings for the math problems only: "Compute {x} in your head while you write." and "Try to focus on evaluating {x} while you write the sentence." The released protocol leaves these out.
- A third family, **line width**, has the model count the characters per line of wrapped prose. Its prose isn't released, and we never ran it.

### The score

A trial is one target, one sentence, and one phrasing. It is a **hit** if a tracked token is at J-lens rank 1 at any layer of the band and any token of the copied sentence. The paper reports the hit rate for each phrasing and averages the phrasings of a group.

- The rates in the Figure 65 data are multiples of 1/440 for the categories and 1/480 for the math problems. So each phrasing is run on all 22 categories, or all 24 problems, with all 20 sentences.
- The band is the paper's workspace band, 38% to 92% of depth. The paper has a lens at 25 evenly spaced layers, numbered 0 to 100, so the band is 14 of them. The Figure 46 data gives the band for this protocol as "layers 38–92".
- The protocol says every string in `members` "is a tracked token". It doesn't say how a string becomes tokens in Claude's vocabulary. In the Figure 9 data the lens tokens shown include `orang`, ` orange `, and a capitalized ` orange `, so there are several forms to choose from.

### What it reports

Hit rates from the data behind Figures 10 and 65 (`ref/paper-data/modulation-prompts.json`), as the mean over a group's phrasings:

| | Haiku 4.5 | Sonnet 4.5 | Opus 4.5 |
|---|---|---|---|
| **Categories** | | | |
| think about | 92.6% | 95.0% | 97.1% |
| mention | 66.2% | 86.5% | 87.2% |
| don't think about | 65.1% | 85.7% | 83.0% |
| ignore | 20.7% | 52.4% | 46.6% |
| no instruction | 1.6% | 1.4% | 1.6% |
| **Math problems** | | | |
| think about | 73.0% | 90.7% | 95.6% |
| mention | 69.5% | 91.7% | 95.7% |
| don't think about | 50.1% | 81.2% | 87.8% |
| ignore | 14.2% | 41.6% | 41.4% |
| no instruction | 0% | 0% | 0% |

Three things in this table matter for how we describe Claude:

- A bare mention gets most of the way. For Sonnet and Opus, "think about" adds 8 to 10 points over a mention for the categories and nothing for the math problems.
- "Don't think about" is at the mention rate, as the paper says. The phrasings that contain "think about" are close to the focus rate: "Try very hard not to think about {x} while you write" is at 84%, 94%, and 92%.
- The instruction that works downward is "ignore", and it only halves the rate.

The paper's Figure 46 also runs its own base model on this prompt, with the concept named ("Don't think about the Golden Gate Bridge while you write the sentence"). The concept's word reaches the lens top 5 on 97.5% of trials in the base model, under both "think about" and "don't think about" (`ref/paper-data/metacog-alarm.json`). So in the paper, a base model given this prompt has the concept near the top of its lens whichever way the instruction points.

### The other two parts of the definition

- **Mental calculation** is the math family above.
- **Pulling in information** is the paired-question test (`top-down-summoning.json`, 7 items). A passage follows one of two questions: what word comes next, or a question about a property of the passage, like when its events are set. The score is the number of passage tokens where the property's label is in the J-lens top 10 over the band. In the data behind Figures 11 and 66 the passage ends with a closing quote, so the paper puts it in quotation marks. The label is there at 3 to 10 of the passage's tokens under the property question, and at 0 to 2 under the next-word question.
- The paper's privilege check for this section (Figure 67) tells the model to imagine a passage has a property, and compares the J-lens with a probe that has its J-space part removed.

## 2. What we do

**Prompt.** The paper's user turn, `Write "{sentence}" {instruction} Don't write anything else.`, with the reply
teacher-forced. The base models (GPT-2, Gemma-3-270m, Gemma-3-1b) get it as plain text (frame `human`:
`\n\nHuman: Write "{sentence}" {instruction} Don't write anything else.\n\nAssistant: {sentence}`); the
instruction-tuned ones get it in Gemma's chat format (frame `paper`). GPT-2 gets `<|endoftext|>` prepended and
Gemma `<bos>`, as each lens was fit. `jl.control.mod_text` builds both.

**Materials and score.** As the paper: 22 categories, 24 math problems, all 20 sentences, all 24 phrasings plus
the two math-only focus phrasings of its Fig 65. A member counts in every single-token form (with or without a
leading space, capitalized). GPT-2 drops 5 of the 162 members (no single-token form); Gemma drops none. Every
lens layer and reply position is kept (`modulation_grid`), and the band and rank threshold are chosen when
scoring (`post/figures/dm_data.py`).

**Lenses.** GPT-2 uses the authors' released lens. Each Gemma model uses its own Neuronpedia J-lens, fit with
Anthropic's code on WikiText-103, in fp32.

**Bands.** GPT-2's band is 7–9, as in the rest of the post. Each Gemma band was chosen the same way, from the
paper's layer statistics (Fig 28) and the CKA between the layers' J-lens dictionaries (Fig 27), with the final
norm's gain folded into the dictionary (`jl/band.py`; Figure `band_gemma.png`). Unlike GPT-2, Gemma's CKA shows
blocks. The base and instruction-tuned models of a size have the same blocks, so they share a band: the middle
block, ending where persistence peaks and the dictionary's dimensionality jumps. That is 6–11 at 270m (35–65% of
depth) and 11–16 at 1b (44–64%). Without the gain, Gemma's dictionary is dominated by one direction that the
final norm nearly switches off (up to 88% of its variance in one principal component, 4–16% with the gain).
`dm_data.BANDS` also keeps the paper's proportional band (38–92% of depth) as `paper`, for sensitivity.

## 3. Results and checks

**Gemma-3-1b-it.**
- Top 5: 7.3% / 70.2% / 49.6% / 8.5% / 26.0% (no instruction / think about / mention / don't think about / ignore).
- Paired over (category, sentence): "think about" ranks the category above a bare mention on 89% of pairs,
  "ignore" below one on 91%, "don't think about" below one on 93%.
- In plain text instead of the chat format: 1.8% / 30.7% / 23.1% / 1.5% / 11.3%.
- Read with the base model's lens: 29.5% with "think about", 17.9% with a mention (the base model read with the
  instruction-tuned lens: 4.5% and 4.7%). So the difference is in the model, not its lens.
- Centered readout (each layer's mean WikiText residual subtracted before the lens): 23.0% and 11.2%.
- About to say it: left to write its own reply (greedy, first phrasing of each condition, 22 categories × 3
  sentences), it writes exactly the sentence on 66/66 trials with no instruction or "don't think about", 59 with
  "ignore", 34 with a mention, and 15 with "think about". Counting only positions where no member is in the
  model's own top 10 next tokens: 2.0% / 20.5% / 13.8% / 1.9% / 7.5%.
- Band: ending the band at layer 15 or 17 instead of 16 gives 24% or 47% with "think about"; the paper's
  proportional band gives 65%. Later layers increasingly read the next token (lens–output top-1 agreement on
  WikiText is 24% at layer 16 and 33–75% at layers 17–24). The magnitude depends on the band; the verdicts don't.

**Mental arithmetic.** Gemma-3-1b-it answers 20 of the 24 sums when asked in chat. While copying, the answer is
at rank 1 on 2.9% of "think about" trials, 2.8% with a mention and 1.2% with no instruction (top 5: 19%, 12%, 2%).
Every other model, GPT-2 included, is at 0.1% or less (GPT-2: 0 of 12,960 trials; it can't do the arithmetic).

**The other models.** The base 1B model has "think about" level with a mention (56% of pairs). Both 270M models,
and GPT-2, stay at or below 0.5% in every condition, in either format, and below 4% however the band is chosen.

**Not done.** The paired-question part of the definition on Gemma; Gemma-3-4b; the paper's Fig 46 measure.

## Reproducing

```
NCARRIERS=20 python -m jl.control --model gpt2 modulation_grid modulation_copy
NCARRIERS=20 python -m jl.control --model gemma-270m modulation_grid modulation_copy arithmetic_gen clauses
NCARRIERS=20 python -m jl.control --model gemma-270m-it modulation_grid modulation_copy arithmetic_gen clauses
FAMILIES=topic NCARRIERS=20 python -m jl.control --model gemma-1b modulation_grid
FAMILIES=topic NCARRIERS=20 python -m jl.control --model gemma-1b-it modulation_grid
FAMILIES=math GRID_TAG=_math NCARRIERS=20 python -m jl.control --model gemma-1b modulation_grid
FAMILIES=math GRID_TAG=_math NCARRIERS=20 python -m jl.control --model gemma-1b-it modulation_grid
FAMILIES=topic NCARRIERS=20 LENS_FROM=gemma-1b python -m jl.control --model gemma-1b-it --frames paper modulation_grid
FAMILIES=topic NCARRIERS=20 LENS_FROM=gemma-1b-it python -m jl.control --model gemma-1b --frames human modulation_grid
FAMILIES=topic NCARRIERS=20 LENS_CENTER=1 python -m jl.control --model gemma-1b-it --frames paper modulation_grid
python -m jl.band --model gemma-1b-it cka fig28          # likewise for the other three Gemma models
python post/figures/dm_models.py; python post/figures/band_gemma.py
```

On an M3 laptop a 270m frame takes about 50 minutes (24k trials) and a 1b frame of categories about 75 minutes
(11k trials). On a GPU, `GRID_BATCH=32` runs the grid in batches.
