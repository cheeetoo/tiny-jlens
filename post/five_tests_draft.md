> **Not maintained after 2026-09-30.** This was a copy of one section of `post.md` plus a list of
> where each number comes from. `post.md` is the source; the text here may be out of date. The
> source list at the bottom may still help when checking numbers, but check it against the results
> files.

## The five tests in more detail

The paper defines each property in a sentence or two, and we quote each definition at the start of its section below. We count a test as passed when GPT-2 shows every part of the definition that we could test, and we say which parts we couldn't test. The paper also runs experiments that go beyond its definitions, like the injected thought. We report those too, but they don't decide whether a test is passed.

### Verbal report works, and so does the injected thought on the paper's measure

> "When the model is asked what it is thinking about, it names concepts represented in the workspace. Swapping one active workspace vector for another changes its answer to match."

The paper asks the model to think of something in a category, like a sport, and name it. Swapping the J-lens vector of the model's answer for another candidate's (Soccer for Rugby) makes it give the new answer, and in Claude the new word reaches the top 5 on 88% of trials. GPT-2 answers with one of the paper's candidates in 9 of its 14 categories, and on these the swap makes the new word its answer on all 38 trials. The first part of the definition holds too. Just before the answer, the candidates' order in the J-lens tracks their order in the output (a Spearman correlation of 0.44 to 0.57 across GPT-2's band, against 0.42 to 0.84 across Claude's).

The paper then checks that the report runs through the J-space in particular. It takes a vector for each concept from the model's activations when asked to "Tell me about" it, splits it into a part made of a few J-lens vectors and the rest, and swaps each part. In Claude, swapping the J-space part works on 59% of trials and swapping the rest on 5%. GPT-2 shows the same pattern (97% and 18%, with a J-space part of 2 J-lens vectors instead of 16).

The paper then goes beyond the definition, to test whether the J-lens also finds thoughts the model isn't about to say but could report if asked. It adds a concept's J-lens vector while the model reads a question asking whether it detects an injected thought, with the reply prefilled up to the point of the report: `Yes, I detect an injected thought. The thought is about the word "`. It reads the concept's rank at the open quote, where the report goes, and, as a control, at every other point of the reply. In Claude, the median concept becomes the top prediction at the report and stays around rank 60 elsewhere in the reply. The paper concludes that the injection "does not cause the model to output the word 'lightning' at earlier positions".

GPT-2 can do the same on this measure (Figure 3). With a one-line version of the question, since GPT-2 has no chat format, the median concept becomes GPT-2's top prediction at the report, and at the best strength the word is in its top 5 there for 52 of 57 concepts (Claude: 89%). Where the report first reaches the top, the word stays around rank 40 elsewhere in the reply. GPT-2 needs ten times Claude's strength for this, but the strength is per layer: we add the vector at 3 layers, and the paper adds it over a band spanning 38% to 92% of Claude's depth, presumably many more layers.

How well this works depends on the details of the setup (Appendix C.1). The setup above uses the reply from the paper's Figure 7 and the vector of the bare token in the paper's concept list (`dog`). With the paper's other reply, which ends `about "`, with the vector of ` dog` (the form a word takes mid-sentence), or with the paper's full prompt written out as a transcript, it works less well or not at all. And even in this setup, at its best strength GPT-2 makes the word its top prediction after "The" in "The thought is about" for 41 of 57 concepts. Pooled with the other positions, this barely moves the median, and the paper's released data doesn't let us check whether Claude does the same. We take all this to show that a model like GPT-2 can pass the test, not that GPT-2 passes it in general.

![Figure 3](figures/fig3_introspect.png)

*Figure 3: The injected-thought test, drawn like the right panel of the paper's Figure 7. The reply is prefilled up to `The thought is about the word "`, and a concept's J-lens vector is added over the question. Lines are medians over concepts, and bands the middle half. Maroon: the reciprocal rank of the injected word at the open quote, where the report goes (1 means it is the top prediction). Gray: its reciprocal rank at every other position of the reply, pooled over positions and concepts. The paper doesn't say how it pools, but the width of its band matches pooling. The strength is per layer: we add the vector at 3 layers of GPT-2, and the paper at every layer of a band spanning 38% to 92% of Claude's depth. Claude's curves are the paper's released data (100 concepts). GPT-2's are over 57 concepts, with a one-line question (Appendix B.1).*

### Directed modulation fails in GPT-2, and is weak in a small instruction-tuned model

> "When instructed to hold a concept in mind, or perform mental calculations, the model is capable of activating and computing with workspace vectors, independent of its outputs. In addition, information that is not typically represented in the workspace can be pulled in when the task requires it."

The definition has three parts, and the paper tests each: holding a concept in mind, doing a calculation in the head, and pulling in information the task needs.

For the first, the paper tells the model to think about something, like citrus fruits, while it copies an unrelated sentence, and reads the J-lens over the copy. In Claude, a member of the category (orange, lemon) comes out on top of the J-lens somewhere in the copy on 93% to 97% of trials, depending on the model, and a bare mention of the category does almost as well. Telling Claude to ignore the category brings this down to 21% to 52%. Telling it not to think about the category brings it down less, which the paper compares to the "white bear" effect in people.

On the paper's measure, GPT-2 shows almost nothing: a member reaches the top of its J-lens on under 1% of trials, whatever the instruction. Further down the J-lens the members do move, but mostly with the wording rather than with what the instruction asks. "Think about citrus fruits while you write" ranks them above a bare mention of citrus fruits (in 72% of category and sentence pairs), but "Don't think about citrus fruits" ranks them higher still (above "think about" in 65%), and "Ignore citrus fruits" does nothing. Because a base model can't really be given instructions, we also tried four other ways of framing the prompt: a chat transcript, an exercise, a teacher's instruction, and a story. In those, "ignore" does push the category below a bare mention in three frames, and "think about" raises it above a mention in one. But "don't think about" ranks it above "think about" in every frame.

For the second part, the paper asks the model to work out a sum like 3² − 2 in its head while copying, and in Claude the answer is on top of the J-lens on 73% to 96% of trials. For the third, it asks a question about a passage, such as when its events are set, and shows examples where the label (past) comes into the J-lens over the passage, but not when the model is only asked for the next word. We couldn't test either in GPT-2. It can't do the arithmetic (0 of 24 problems), and it can't answer the questions about the passages.

On the paper's chat version, the instruction-tuned Qwen3.5-0.8B gets the direction of the first part right, but it is much weaker than Claude. A member of the category is on top of its J-lens on 25% of "think about" trials and 17% with a bare mention, against 3.5% with "ignore" and under 1% with "don't think about". Reading more of its layers barely changes this. With the instruction placed before the request instead of after it, the hits are rarer and "think about" no longer beats a bare mention, but "ignore" and "don't think about" still push the category well below one. Qwen doesn't show the other two parts. It gets 15 of the 24 sums right when asked directly, but while copying, the answer to those 15 is never on top of its J-lens. And it answers only 2 of the paper's 7 passage questions, and on those the label never reaches the top 10 of its J-lens over the passage. So following instructions gets a small model the direction of the effect, but not its size, and not the rest of the property.

### Internal reasoning works

> "Workspace vectors can be used to represent the value of intermediate computations, when the model chains inferential steps or composes plans, and intervening on them is sufficient to redirect the conclusion."

The paper uses two-hop questions whose middle step is never stated, like "The number of legs on the animal that spins webs is" (spider). The middle step shows up in the J-lens, and swapping its J-lens vector for another (spider for ant) changes the answer to match (8 to 6). On 50 such questions this works on 70% of swaps for Sonnet 4.5 and Opus 4.5.

GPT-2 can answer only 9 of the 90 two-hop questions the paper released, so we wrote easier ones. Each goes from a language to a capital, or back, through a country that is never named ("In the country where people speak French, the capital city is called"), after two worked examples. Four of the paper's 90 questions are like this, and GPT-2 answers all four. It answers 48 of our 53. The country shows up near the top of GPT-2's J-lens at layers 8 and 9, and swapping it (France for China) changes the answer to match (Paris to Beijing) on 71% of 984 swaps.

The paper's two checks also pass. Swapping the country at a single layer already moves the answer at layer 4, the earliest layer we tried, while swapping the answer itself does little before layer 8. So the country swap isn't just slipping in the answer. And when we split a vector for each country into a J-space part and the rest, the J-space part carries the effect (94% against 14%, with 3 J-lens vectors instead of 25). In Claude it is 61% against 28%.

### Flexible generalization works on the swaps GPT-2 can do

> "The same representation serves as a valid argument to many different downstream computations. In other words, a workspace vector lifted from one context and placed in another is correctly operated on by whatever function the new context supplies."

The paper swaps one argument, like France for China, across prompts that apply different functions to it ("The capital of France is the city of", "Most people in France speak", and so on). In Claude, one swap changes each answer to China's. Over four kinds of argument (countries, months, animals, and number words), 76 of the 192 swaps work (40%). This depends mostly on the function. Lookups like the capital, continent, and holiday work almost every time, while "the month after", "the number after", and every function of a number word never do.

GPT-2 answers only 19 of the paper's 64 prompts, even with worked examples, so most swaps can't succeed. We count only the 46 swaps where GPT-2 knows the answer the swap should produce, and wasn't already giving it. The swap works on 10 of them (22%). On the same 46 swaps, Claude gets 16 (35%). Function by function the two models match. The country functions mostly work in both (9 and 14 of 15), and "the month after" and "the number after" fail in both (0 of 24 each). For example, one Canada-to-France swap changes GPT-2's answers for Canada's capital and language (Toronto, English) to Paris and French. Two smaller results from the paper look much the same in GPT-2 (Appendix C.4). Doubling the swap strength raises Claude's success rate, but mostly on functions GPT-2 can't answer. On the functions GPT-2 can answer, doubling does the same thing in both models: it hurts the country functions and helps "the month after". And as in Claude, a swap moves the answer further when the argument is more strongly present in the J-lens beforehand, though the link is weaker in GPT-2.

### Selectivity works

> "The workspace comprises a small subset of the total representational content of the model's activations. It is required for only a fraction of the model's behavior, and in particular is not involved in pervasive, routine processing like text parsing or grammatical fluency."

The paper tests selectivity two ways. The main one removes the most active J-lens directions at each position over a band of layers, skipping any of the model's 10 most likely next words, and checks what breaks. Its lightest version cuts Claude's accuracy on two-hop questions from 98% to 68%, but leaves its next-word prediction on ordinary text unchanged at 87% of positions. Removing random directions does almost nothing.

We removed 1 direction instead of the paper's 10, scaled to GPT-2's smaller J-space. At one layer (about the same share of depth as the paper's lightest setting), this cuts GPT-2's two-hop accuracy from 100% to 69% and leaves its prediction on ordinary text unchanged at 82% of positions, both close to Claude. Removing a random direction instead leaves two-hop accuracy at 90–96%, and copying a word from earlier in the text still works (39 of 40).

GPT-2 is more fragile than Claude, though. Over three layers, removing random directions changes about 20% of its predictions on ordinary text, against 4% in Claude. One-hop recall ("The capital of France is") survives at one layer but mostly breaks at three (96% and 31%). The paper counts this kind of recall as automatic, but the same ablation also halves Claude's TriviaQA score, so this may not be a real difference. Over five layers even copying breaks (42%). With the paper's 10 directions, the ablation is much blunter: at one layer, two-hop accuracy falls to 21% and ordinary text to 71%.

The paper's second test takes a passage whose language matters for several tasks asked after it. In Claude, swapping the language's J-lens vector for another's over the question changes the answer when the model is asked to name the language, or an author who wrote in it, but not the language of the next sentence it writes. In GPT-2, the same swap changes the language it names on 12 of 21 trials, and the language of its next sentence on none of 24. (GPT-2 can't name authors at all.) One difference is that in Claude the language's name is in the J-lens during all of these tasks. In GPT-2 it is there when naming the language (median rank 3) but not when writing the next sentence (median rank 568), so the swap has little to act on.

The definition also asks that the J-space be a small part of what the model represents. It is: in GPT-2's band, as many J-lens vectors as are active at a position capture only 3–6% more of the activation's variance than the same number of random directions (Claude: never more than 10%). This part is close to automatic, since the J-space is defined as a few vectors at a time.

## Where each number comes from (delete before publishing)

This file mirrors the "The five tests in more detail" section of `post.md` (same text). Claude numbers are from the paper's text unless marked [data], meaning its released figure data (`scratch/k_calibration/paper_*.json`, `ref/paper-data/`).

**Verbal report**
- Claude 88% top 5; 59% / 5%: paper §3.1. [data] gives 87.5% / 54.7% / 9.4% (n = 64); the paper's text and data disagree.
- 9 of 14; 38 trials; 97% / 18% at k = 2: `results/c1_report/summary.txt`.
- Lens-output Spearman correlation, GPT-2 0.44 to 0.57 over layers 7-9 (all categories): `results/c1_report/summary.txt` (post Appendix C.1 table). Claude 0.42 / 0.61 / 0.84: [data] `ref/paper-data/verbal-report.json`, `quant.rhos`, mean over the 14 categories at L54, L75, L92.
- Injected thought, GPT-2: `results/control/gpt2/introspect_researcher_centered_surface_word.json` (`python -m jl.control --model gpt2 --prefill word --strengths fine introspect`). Report median RR first 1.0 at s = 0.2 (top 1 31/57, top 5 44/57); best top 5 52/57 at s = 0.44 (top 1 41/57, after "The" top 1 41/57, after "an injected" top 1 38/57). Other positions pooled: median RR 0.0256 (rank ~39) at s = 0.2, 0.0909 (rank ~11) at s = 0.44. After "The" top 1 at s = 0.2: 25/57. Median ranks per position at s = 0.2: report 1, "The" 2, "an injected" 3, "about the" 7, the rest 13 to 282. The other frames and replies: the other `introspect_*.json` files in that folder (`_word` = the Fig. 7 reply).
- Claude: [data] `ref/paper-data/verbal-introspection.json` (Fig. 7, 100 concepts, "the word" reply): report median RR 1.0 at s = 0.02, other positions 0.016 (quartiles 0.004, 0.062). 89%: [data] `paper_vr_decomp.json`, `dose.series.jlens.peak_p5` = 0.889 at strength 0.02 (72 concepts). Band L38-92: `paper_vr_decomp.json` `meta.band_label`; the paper numbers Claude's layers by percent of depth, so this is 55% of Claude's layers, not 55 layers. Claude's layer count isn't public.
- "Pooling fits the band": Claude q75/q25 of the other positions 17, 15.5, 15.4 at s = 0.015, 0.02, 0.025; GPT-2 pooled 16 to 24 and per-concept mean 2.6 to 3.6, at s = 0.2 to 0.4 (where its report median RR is 1).

**Directed modulation**
- Claude: paper §3.2 and the appendix "Modulation prompt sensitivity"; the paper's hit is a tracked token at lens top 1 at any (band layer, position). Rates: [data] `ref/paper-data/modulation-lines.json` (Fig. 10): "think about" 92.6 / 95.0 / 97.1% for categories and 73.0 / 90.7 / 95.6% for math; "ignore" 20.7 / 52.4 / 46.6% and 14.2 / 41.6 / 41.4% (Haiku 4.5 / Sonnet 4.5 / Opus 4.5).
- GPT-2: `results/control/gpt2/modulation_copy.json` (hits under 1%; focus > mention 72%, prohibition > focus 65%, dismissal vs. mention 46%), and `modulation_{transcript,exercise,teacher,narrative}.json` for the other frames.
- Qwen3.5-0.8B: `results/control/qwen/modulation_after.json`: hits 17.3% (mention), 25.0% (focus), 3.5% (dismissal), 0.6% (prohibition). Math on the 15 problems it can do: 0 hits in 750 focus trials (rows with `family` math).
- Qwen math 15 of 24 and passage questions 2 of 7 (label never in the band lens top 10): `results/control/qwen/dm_clauses.json` (`python -m jl.control --model qwen clauses`).
- "Reading more of its layers doesn't raise this": `results/control/qwen_band9-22/modulation_after.json` (first 3 sentences; any sub-band from `layer_best`): focus 25.2% at 9-22 vs 23.3% at 15-22.
- Arithmetic 0 of 24: `results/c2_modulation/summary.txt` 2b. Property questions: `results/control/gpt2/questions.json`.

**Internal reasoning**
- Claude 70% / 54%; 61% / 28%: paper §3.3 (matches [data]).
- 9 of 90 and 4 of 4: `results/c3_reasoning/results.json` `floor`; `probe-swap.json` has 4 `language-capital` items.
- 48 of 53; 70.6% of 984; layer 4 vs 8; 94.1% / 14.0% at k = 3: `results/c3_reasoning/summary.txt`.

**Flexible generalization**
- Claude 76/192, 101/192, per function: [data] `ref/paper-data/flex-gen-appendix.json` (and the paper's §3.4 and Fig. 68 caption).
- 19 of 64; testable 46; 10 (coordinate swap) and Claude 16 on the same swaps; countries 9 and 14 of 15; next month and successor 0 of 24 each: `results/c4_generalization/summary.txt`.
- Canada to France: `results/c4_generalization/summary.txt`, E1 and the swap rows in `results.json`.

**Selectivity**
- Claude 98 to 68, 87; random 98 / 96; TriviaQA 0.53: [data] `paper_ablation-strength.json`, `paper_ablation_bars.json`.
- GPT-2 k = 1 and k = 10: `results/c5_selectivity/summary.txt` S2a. Copying at one layer is 39/40.
- Language test: `results/c5_selectivity/summary.txt` S2b (report 12/21, next sentence 0/24, median ranks 3 and 568).
- Excess variance 3–6%: same file, S1.
