# External commentary on Gurnee et al. (2026), "Verbalizable Representations Form a Global Workspace in Language Models"

Compiled 2026-09-24 for a GPT-2-small replication write-up.

**How to read the provenance tags**
- **[READ]**: I pulled the full text (LessWrong GraphQL API, direct HTML, PDF, or the GitHub API) and read the relevant parts. Anything in quotation marks under a [READ] tag is copied verbatim from that text.
- **[FETCH-SUMMARY]**: I only saw the page through a summarising fetch tool, so paraphrase may have crept in. Quotes under this tag are not verified.
- **[SNIPPET]**: I only saw a search-result snippet, or a description in another source (named). Not verified.
- **[NO ACCESS]**: I tried and could not get the page (paywall, 403, or JavaScript-only).

All sources are linked below. Local copies of some of them are kept out of the repository.

---

## 0. URLs for the commentaries you already have

| Item                                                                                                                 | URL                                                                                                                             | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Combined invited commentary PDF (Dehaene & Naccache; Eleos; Nanda)                                                   | https://www-cdn.anthropic.com/files/4zrzovbb/website/cc4be2488d65e54a6ed06492f8968398ddc18ebe.pdf                               | 53 pages. Linked as "Read their commentaries" at the bottom of Anthropic's blog post (https://www.anthropic.com/research/global-workspace). Page ranges: Dehaene & Naccache pp. 2–15, Eleos pp. 16–32, Nanda pp. 33–53. [READ]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Dehaene & Naccache, "Does Claude possess a conscious global workspace?" (standalone)                                 | https://unicog.org/wp_2025/wp-content/uploads/2026/07/Dehaene-and-Naccache-Workspace-commentary-on-Gurnee-Lindsey-June-2026.pdf | Resolves (HTTP 200).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Eleos (Butlin, Shiller, Plunkett, Long), "Consciousness and cognitive access in LLMs" (standalone, dated 2026-07-06) | https://eleosai.org/papers/eleos_gwt_commentary_20260706.pdf                                                                    | Resolves (HTTP 200). Note: the Anthropic blog lists Derek Shiller under Rethink Priorities. The Digital Minds newsletter (Sep 2026) says he has since joined Eleos.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Neel Nanda review (LessWrong version, same text plus comments)                                                       | https://www.lesswrong.com/posts/zFJ3ZdQwrTWE9jT5S/a-review-of-anthropic-s-global-workspace-paper                                | 2026-07-06. [READ]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **Gavin Leech, "On J-space" ("A very hot take written in 2 hours")**                                                 | **https://www.paradigm3.org/research/jspace/**                                                                                  | Posted **7 July 2026** on the **Paradigm 3** site, not gleech.org, his Substack, or LessWrong. I checked his LW account (`technicalities`) and his Substack archive and found no crosspost. [READ] Confirmed as the text you have: it contains "Main objection#: Half of this looks like a selection effect…", "roughly half of the advertised workspace properties are near-analytic consequences of how the J-space is found", and a "Mild effect sizes" section. Sarah Constantin's link roundup of 2026-09-23 points to the same URL and summarises it as "Gavin Leech does not think the J-space actually is an AI's 'global workspace.' he thinks the argument is circular" (https://www.lesswrong.com/posts/cAfAayy9k2H25iBHx?commentId=iWbmjNPyLZXgdsAbe). |

---

## 1. Critiques of the global-workspace and consciousness framing (priority 1)

### 1.1 David J. Chalmers, "Is the Jacobian space a global workspace?"
- URL: https://philpapers.org/rec/CHAITJ-2 (PDF: https://philpapers.org/archive/CHAITJ-2.pdf)
- Date: added to PhilPapers 2026-08-14. It thanks Jack Lindsey and cites the R-lens post of 2026-08-05.
- [READ] (full 16-page PDF)
- **Summary.**
  - Chalmers argues the J-space's links to the global workspace and to access consciousness "have been somewhat overstated."
  - **(a) Functional criteria.** His first objection is that the five "workspace-like" properties are generic properties of conscious access, not properties specific to global workspace theory: "these five properties reflect the generic characterization of conscious access and not any of the further distinctive claims of global workspace theory". He proposes "access-like" as the more accurate label.
  - He separates a "substantial" workspace (a causal, dedicated, privileged, unified system) from a "minimal" one, which is just a set of representations that are selectively broadcast. On this he writes that J-space vectors "are selected precisely because these are the ones that have a strong disposition to influence output tokens. We might say: they are in the J-space because they are accessible, or disposed to be broadcast. This is the mark of a minimal workspace."
  - He calls the union-of-cones J-space a "gerrymandered object". He notes the logit lens reproduces many of the effects and that the R-lens probably reproduces them too. His footnote 6 makes a K-space point: showing that J-space vectors beat non-J vectors cannot show the J-space is privileged, because many other "K-spaces" might beat their complements in the same way.
  - He conjectures that the effects come from **outputability rather than verbalizability**. He predicts that J-space-like effects will appear in image-only or audio-only generative models.
  - He also separates verbalizability from reportability with a Freudian example: thoughts that make someone say "mother" are verbalizable without being reportable.
  - **(b) Structural properties.**
    - Broadcast: "The J-space does not seem to broadcast information in this sense… best characterized not as broadcast in the classical sense… but as influence."
    - Broadcast heads: J-lens vectors beat only five other populations, and "merely beating five other populations is not enough to establish that J-space is privileged here."
    - Capacity: "The J-space as designed involves sparse combinations of 25 or so J-lens vectors, so it has a built-in limited capacity."
    - Ignition is degreed rather than all-or-none. He cites Dehaene & Naccache's "Ignition remains to be fully demonstrated."
    - Overall: "the J-space shows at best limited versions of integration, capacity limits, and ignition, and it does not seem to involve recurrence or modules at all."
  - **(c) Small models:** not discussed directly. The "minimal workspace" argument implies that the functional properties alone would not tell a small model apart from a large one.

### 1.2 Erik Hoel, "Anthropic runs like Wile E. Coyote into the brick wall of consciousness research"
- URL: https://www.theintrinsicperspective.com/p/anthropic-runs-like-wile-e-coyote
- Date: 2026-07-13
- [READ]
- **Summary.**
  - Hoel says the paper is built on reportability. He argues that a global workspace theory stripped down to reportability is "scientifically trivial". He calls this failure mode "strict dependency": the evidence used to infer consciousness (report) and the theory's prediction come from the same source.
  - **(a)** "In order to make the definition work, they drop most requirements for a 'global workspace' outside of things that naturally go along with reportability (e.g., how could reportability not involve internal reasoning, or generalization?)."
  - He gives deflationary readings of two properties. "The deflationary version of their definition of ignition is just when the model commits to an interpretation." "The deflationary version of their definition of broadcast is just that information with high reportability is more likely to be used by later layers." He uses the paper's own statement that the logit lens captures "much of the workspace-like structure" as support.
  - **(b)** He treats the sensory/workspace/motor split in the CKA plot as the one part that is not baked in by construction. He then argues it is:
    - post hoc, quoting the paper's admission that the late boundary "was somewhat post-hoc"; and
    - possibly falsified on open models, citing Elie Bakouch's J-lens CKA explorer over about 38 open models (https://eliebak.com/viz/jspace-open): "for open-source language models, there is no sharp blocky tripartite structure".
  - He notes that Anthropic shows a sharp block structure only for Sonnet 4.5. He also quotes the paper: "in some models the transition is more gradual, sometimes containing sub-blocks, and… the observed sharpness is exaggerated by layer subsampling."
  - **(c)** His "substitution argument" uses open models as behavioural stand-ins that lack the structure.
  - Notable comment by Amarda Shehu (2026-07-13): "The global workspace framing adds no predictive content. Every result the paper reports follows directly from representational efficiency. The one property that I would want to see (which would distinguish the workspace), is sharp nonlinear ignition. This is the one they cannot show."

### 1.3 Gavin Leech, "On J-space" (Paradigm 3). You already have this.
- URL: https://www.paradigm3.org/research/jspace/ · 2026-07-07 · [READ]
- For cross-reference, his central argument, verbatim: "Any direction selected by that statistic must (a) shift verbal output under intervention…; (b) be transmitted by downstream weights, since a large average Jacobian just is the product of transmissibility downstream; (c) be context-general, since context-specific effects would cancel out under the averaging step."
- He concludes that reportability, flexible generalisation and selectivity (in the small-variance-share sense) "could just follow from the J-lens selection procedure", and that "MLP-gain and broadcast-head are presented as discovered structures."

### 1.4 Robert Long (Eleos), X thread plus Substack post "'Merely' functional is still a big deal"
- X thread: https://x.com/rgblong/status/2074300916455670201 (2026-07-06/07). [SNIPPET] for the page itself. The text is [READ] via Zvi's quotation in "No Space Like J-Space".
- Eleos TL;DR thread: https://x.com/rgblong/status/2074211176280690906 [SNIPPET]
- Substack: https://experiencemachines.substack.com/p/merely-functional-is-still-a-big (2026-08-30) [READ]
- **Summary.** Long's complaint is about communication. Anthropic stresses that it makes "only" functional claims and then "help[s] themselves to an extremely non-trivial functional claim". In his words: "basically all of the objections I've seen to that paper, and the main ones considered in the reviews, are about its core *functional* claims."
- He quotes Chalmers's line about limited integration, capacity and ignition, and says Eleos "made related points."
- In the X thread (via Zvi) he endorses Nanda's caution: "it's easy to read too much into post-hoc analysis of results like this."
- He also backed thebes's suggestion that capacity limits deserve more probing ("thought it was light on exploring hard capacity limits of this kind").

### 1.5 Anil Seth, "Once again we are told AI may be conscious – I study consciousness, and I have my doubts" (Guardian)
- URL: https://www.theguardian.com/commentisfree/2026/jul/15/ai-consciousness-anthropic-claude-dawkins · 2026-07-15 · [READ]
- **Summary.**
  - Seth calls the research "impressive" and "valuable" because it looks for mechanistic signatures instead of relying on behaviour.
  - Against the framing: "Anthropic's findings fall short of what global workspace theory typically requires (for example, there is no recurrent activity in Claude…)."
  - His main objection is biological naturalism: "the information processing unfolding inside Claude is no more likely to result in consciousness than a simulation of a weather system is likely to generate a real hurricane."
  - He says nothing about small models or the J-lens construction.

### 1.6 LessWrong: comments on the authors' post "A global workspace in language models" (wesg)
- URL: https://www.lesswrong.com/posts/3PaLrzxagpbnNtPLT/a-global-workspace-in-language-models · post 2026-07-06, score 369, 67+ comments · [READ] (all comments pulled through the API)
- Individual comments are at `…/3PaLrzxagpbnNtPLT?commentId=<id>`.
  - **Lucius Bushnaq** (`EwojCnTw23xnLuFr5`, 07-07, score about 91): "it isn't a subspace of the residual stream, or any kind of vector space"; it is "sparse non-negative linear combinations of 'J-lens vectors'" found by gradient pursuit. Replying to Measure, he adds: "I would not expect it to pick up on thoughts that aren't tightly bound to a relatively small number of tokens."
  - **Linda Linsefors** (several comments, 07-08/09) separates two things. One is a real result: "a new method for finding meaningful linear directions in the residual stream". The other is not shown: "What they haven't found is some separation between the conscious-like and the unconscious-like information channels… Taken as a linear space, the J-space is just the residual stream." On the selective ablation effects: "that could be the difference between using internal states that are aligned with single tokens or not. Or having other redundancy or not."
  - **Jason R Brown** (`w4MuNjkdTwAyzLAyc`, 07-07): "one would expect the J-lens (from its construction) to act as a sort of buffer for tokens that might need to be emitted soon… it looks like some mix of potential-next-token buffering as well as something higher-order, and leaning hard on the workspace analogy seems premature." He also points out that the post, unlike the paper, never mentions that the logit lens captures much of the structure.
  - **Measure** (07-07): "Would this technique have found non-verbal thoughts if they existed, or did it only find words because it was only looking for words?"
  - **Archimedes** quotes the paper's figure that the J-space component carries "a median of only 6–7% of the concept vector's variance". He also quotes the finding that high-κ SAE features beat J-lens vectors on MLP gain in early workspace layers.
  - **Petropolitan** estimates the signal-to-noise ratio at about 1:15 even with carefully chosen layers.
  - **Unnamed commenter** (`WShLDuuivbscNifLX`, 2026-07-29; username not returned by the API) makes the cheap-implementation argument directly. It imagines "a distributed control computer equipped with an arbiter" with one shared cache slot: "Limited capacity, competition among modules, nonlinear ignition, maintenance through recurrent feedback, global broadcasting, global availability—not one feature is missing… Is it conscious? Or has it merely achieved functional coordination?"
  - **phoenix** (`c4dNnEwARCxLBm9YG`, 07-07). This is the only comment about GPT-2 small:
    - "on GPT-2-small I've found that the workspace internals fall below median frequency once that prior is removed (though this did not appear on GPT-2-large)". The final-layer bias correlates with log token frequency at about 0.67.
    - "Broadcasting appears to be almost entirely a read-side property… a direction is a hub because it is widely read, not because it is widely written". Read-side connectivity is about 1.3–1.6× an isotropic baseline, write-side about 1.1×.
    - "Being in the workspace and being wired into the network are two independent axes… I don't think the workspace is the 'integrated' part of the model so much as it's the nameable part."
    - These are unreviewed claims from their own tool ("ATLAS"). Nelith Bandularatne and Ameya Panchal later partly confirmed the frequency point (see 2.3 and 2.10).
  - **Supportive side:**
    - **Yair Halberstadt**: "it fits the model of a global workspace super well… I would have given maybe a 2% chance of seeing all these outcomes."
    - **Kaj Sotala** contrasts the results with Golden Gate Claude.
    - **Neel Nanda** replies that characterising the space is a real contribution even if people had priors.
  - **volpe** (`LP74N9imekrbtMPdB`) notes that Neuronpedia's demo text says Qwen does not think of ocean creatures when told not to, while the paper reports imperfect suppression. **Johnny Lin** (Neuronpedia) answers that Neuronpedia only shows the top 8 readouts per layer and position.

### 1.7 LessWrong shortforms and other short takes
- **Raul Cavalcante** (shortform, 2026-07-07), https://www.lesswrong.com/posts/HiPNvyRAneLyELuP6?commentId=xuyydyNvBk6mZYsQ4 [READ]: "the j-lens estimates hidden-state impact on expected logits over plausible continuations. the 'don't think about the golden gate bridge' result can just be that a plausible assistant completion being: 'ok, i won't think about the golden gate, damn.'" This is a construction-based deflation of directed modulation.
- **Joseph Miller** (shortform, 07-07), https://www.lesswrong.com/posts/DDtEnmGhNdJYpEfaG?commentId=MitdkBEsnrpgMi3jG [READ], written before he had read the paper: "The J-Lens is improvement of the logit lens. It's cool, but not a major advance." **Alex Gibson** replies (`KgyfnS7bigckQEsqa`) that the J-lens answers a different question, about future tokens, but agrees the paper is "definitely overhyped."
- **Francisco Ferreira da Silva, with Stefan Heimersheim** (shortform, 2026-07-27), https://www.lesswrong.com/posts/sQu3RNTBtkJbrNhzS?commentId=cJaTfbpqajutQSeSL [READ]. They applied their feature-direction test (fit a p-norm to the response function under perturbation) to J-lens directions at layer 36 of Qwen3.6-27B. They found p < 2, meaning the directions are "not aligned with the model's features, and not privileged by its error correction mechanism". They add that this lowers their trust both in the J-lens and in their own method.
- **Ethan Garcia**, "When (and when not) LLMs can verbalize awareness of J-Space concept injections" (2026-08-12), https://www.lesswrong.com/posts/uvgw5FTS2RXgFFFii [READ]. This tests the **verbal report** criterion on Qwen3.6-27B.
  - The paper's injection-report demo pre-fills "Yes, I detect an injected thought…".
  - Without that pre-fill, with the model asked to report first, there were "exactly zero reports of intervention awareness" across 1,560 injections.
  - With the answer first, 322 reports appeared, and every one came after an already-steered wrong answer. This is "consistent with autoregressive self-conditioning."
- **Hacker News**, "A global workspace in language models" (2026-07-06, 467 points, 201 comments), https://news.ycombinator.com/item?id=48808002 [READ, top-level comments only]. Representative sceptical comments:
  - snaking0776: the J-space is "basically the expectation of how much a final logits output would change as a result of a small change in a particular layer… more… like showing there exists an abstract reasoning subspace."
  - narmiouh: the J-space "could be simply a transmission channel and j-lens is a reader you added… without the j-space being a cognitive global workspace."
  - Several comments call the consciousness framing PR.

### 1.8 Zvi Mowshowitz, "No Space Like J-Space"
- https://thezvi.substack.com/p/no-space-like-j-space, crossposted at https://www.lesswrong.com/posts/EnxHPxJT4Xin5cTsX/no-space-like-j-space · 2026-07-07 · [READ]
- Largely positive: "a major advance". He argues that the result should shift beliefs toward LLM consciousness by conservation of expected evidence.
- Useful for you mainly because it quotes other people:
  - **antra**: "J-space by design can *only* surface narratable material directly… representations that only subtly affect token distributions… will not get a higher J-lens rank". Antra also notes that Jacobians averaged over 128-token sequences miss long-context circuits.
  - **thebes**: a proposed capacity-limit test. If the J-space is full of "BUT/actually" during a pre-fill the model disagrees with, it should do worse on J-space-heavy tasks.
  - **janus**, Dehaene's Bluesky post, and Rob Long's thread (1.4).

### 1.9 The top-10 protection rule in the ablation experiments
- I found **no source that argues selectivity is an artifact of the rule that exempts tokens in the clean top-10 from ablation.** The closest material:
  - **Peter Flo (Harvard), `jspace-4b`, pre-registered ablation study on Qwen3-4B** (see 2.5). It implements the exclusion rule exactly (`exempt_ids` = clean top-10, including inside the selectivity control; see `src/jspace/calibrate.py`). Even so, 34–42% of teacher-forced top-1 next-token predictions changed under J-ablation, against 16–27% for norm-matched random ablation. So at 4B the rule does not produce the paper's "ordinary prediction left largely intact". Their LEARNINGS doc describes the rule's purpose as keeping "the test aimed at internal reasoning rather than at the model's mouth."
  - **Nanda** raises the opposite worry for the eval-awareness ablation: "if we simply removed all those tokens from the output vocabulary, it seems plausible to me that there would also be a substantial drop in verbalized developments."
  - Neither invited commentary mentions "top-10" or the exclusion rule. I searched the commentary PDF text.
- This looks like an open point you could raise yourself. The protection rule directly shields the clean model's likely next tokens, and "next-token prediction survives" is one of the selectivity measurements. So part of the automatic/deliberate split may be partly engineered. The jspace-4b result on Qwen3-4B is the only data I found that bears on it.

---

## 2. Replications and extensions on open models (priority 2)

### 2.1 Neuronpedia lenses. Is gpt2-small included? **Yes.**
- Hugging Face repo: https://huggingface.co/neuronpedia/jacobian-lens (MIT). [READ via the HF API]
- Created 2026-06-17, last modified 2026-09-21. It currently has **39 model directories**: deepseek-v4-flash, gemma-2-2b(-it), gemma-2-9b(-it), gemma-2-27b, gemma-3-270m(-it), gemma-3-1b(-it), gemma-3-4b(-it), gemma-3-12b(-it), gemma-3-27b(-it), gemma-4-31b, gemma-4-e2b, gemma-4-e4b, gpt-oss-20b, **gpt2-small**, llama3.1-8b(-it), llama3.3-70b-it, olmo-3-1025-7b, olmo-3-1125-32b, **pythia-70m-deduped**, qwen2.5-7b-it, qwen3-1.7b/4b/8b/14b/32b, qwen3.5-0.8b/2b-pt/4b/9b-pt/27b, qwen3.6-27b.
- **gpt2-small lens:** `gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt` (about 13 MB), with `config.yaml` and a convergence CSV.
  - Fit on 2026-06-11 on an NVIDIA B200 with `fit_lens.py openai-community/gpt2`.
  - Data: wikitext-103-raw-v1 train, max_seq_len 128, bf16, n_prompts cap 1000, min 100.
  - Early stop at `stop_at_delta 0.002` → **277 prompts fitted**, final identity_distance 1.3056.
  - The config notes the environment was "backfilled 2026-09-20… the fit did not record its environment".
- Interactive site: https://www.neuronpedia.org/jlens redirects to `/qwen3.6-27b/jlens`. Tabs for Qwen 3.6 27B, Llama 3.1 8B, Gemma 3 12B and DeepSeek V4 Flash, plus a dropdown of "inference-enabled" models (source: `hijohnnylin/neuronpedia`, `jlens-model-selector.tsx`). **I could not confirm whether gpt2-small is live in the interactive UI.** `neuronpedia.org/gpt2-small/jlens` returns 200 but renders client-side.
- The site shows this disclaimer: "J-Lens is typically degenerate in the first 1/3 layers, leading to unreliable readouts."
- Neuronpedia blog, "Welcome to the J-Space", Johnny Lin & David Chanin, 2026-07-10: https://www.neuronpedia.org/blog/jacobian-lens [READ]. At launch: "12 Models Supported"; "We provide pre-fitted J-lenses for 36 models"; and "**For smaller models, we noticed that it was easy to oversteer swaps - it required selecting fewer layers to swap to the desired result.**"

### 2.2 The `anthropics/jacobian-lens` GitHub repo
- https://github.com/anthropics/jacobian-lens. Created 2026-07-02, about 1.9k stars, 283 forks. Discussions are disabled. [READ via `gh`]
- The README describes gpt2 as the small, ungated reference model with a published lens [SNIPPET]. You have `ref/jacobian-lens` locally.
- Issues relevant to your replication:
  - **#5 (cayerbe, 2026-07-13), "Two easy-to-hit measurement pitfalls":**
    - "**Input-copying ceiling.** If a probed token appears anywhere in the prompt, the lens reads it at rank ~1 at that position — the readout reflects the input token, not workspace content."
    - "**Unfitted-position readouts.** `fit()` skips the first `SKIP_FIRST_N_POSITIONS` (16) positions, but `apply()` happily returns readouts at positions 0–15, where the lens is out-of-distribution."
    - They link a six-item pitfall list in their fork (GNS-Foundation).
  - **#6:** `save()` silently overflows large Jacobians to inf in fp16. This hit a Qwen3-1.7B LoRA fit whose early-layer entries were around 1e16.
  - **#2 (tao-hpu):** the package ships only the read side, so replicators must write the swap-intervention hooks themselves.
  - **#15:** OLMo-3 lenses fitted with transformers < 5.13 used the wrong RoPE. Neuronpedia re-fitted them.
  - **#13:** a membership-inference application using regression-fitted transport.
  - PR #17 adds gemma-4-12b-it (closed 2026-09-23).

### 2.3 Nelith Bandularatne, "J-Lens: A Failed Replication on GPT-2"
- https://www.lesswrong.com/posts/tgn3pD2gLpZvepkWk/j-lens-a-failed-replication-on-gpt-2 · 2026-09-07 · repo https://github.com/nelithb/mech-interp-lab · [READ]
- **Directly relevant to you.**
  - A clean CPU re-implementation, checked in two ways. Setting J = I reproduces the logit lens byte-for-byte. A float64 finite-difference gradcheck gives about 8.4e-7 relative error.
  - Metric A, recovering the model's own top-1 next token: J-lens loses to the logit lens at **11/11 layers of GPT-2 small** and wins **1/23 layers on medium** (rank 221 vs 220).
  - Metric B, future tokens p+1..p+5 at layer 9: J-lens is worse by about 8–9 nats at every offset on small. On medium the gap widens with distance.
  - Stress tests did not change the result: refit on 150 sequences; frequency confound (shared bias vs log frequency r = 0.656); sparsity thresholding; skip_first 0/16/32; scaling to medium.
  - Sparsity thresholding produced an apparent 7/11 "win". It collapsed to generic filler tokens traced to GPT-2's massive-activation outlier dimensions.
  - Caveats: only 30–150 fitting sequences; a lens-quality metric, not the workspace properties.
- Comments:
  - **gpjt**: GPT-2's **weight tying** "constrains the representation to be similar all the way through". This may explain why the logit lens is so strong on GPT-2. He offers untied GPT-2-style checkpoints.
  - **williawa**: "the interesting thing to do on smaller models is checking, not whether you can train it… but whether these higher level properties are still there."

### 2.4 tao-hpu, `jspace-replication` ("Independent replication… on small open models")
- https://github.com/tao-hpu/jspace-replication. Created 2026-07-07, last update 2026-07-16. It says an arXiv paper is "forthcoming". [READ: README, `docs/claims-inventory.md`, `docs/replication-log.md`]
- **Tiny models (E0, 2026-07-07)**, using the Neuronpedia gpt2-small lens and the pythia-70m-deduped lens:
  - "**gpt2-small: J-lens does NOT beat the logit lens on target rank at any layer for any probe** (e.g. eiffel-paris L8: J-rank 232 vs logit 11)… Early-layer J readouts are dominated by GPT-2's known glitch tokens (`ModLoader`, ` enthusi`, …)."
  - "**pythia-70m: the J-lens advantage is clear.**"
  - A self-fitted lens on OpenAI GPT-2 shows the same pattern.
  - On their **self-trained nanoGPT GPT-2 124M** the J-lens advantage is clear (eiffel L9: rank 1 vs 28). Verdict: "the exception (openai gpt2) is a property of that checkpoint, not of the lens fit, the architecture, or the 124M scale."
  - Hypotheses (untested): OpenAI GPT-2's residual basis is already well aligned with output space, or glitch-token directions dominate the averaged Jacobian.
- **Qwen3-1.7B and Qwen3-4B** (Neuronpedia lenses):
  - C1 (France→China multi-fact edit) replicated and got stronger with scale.
  - C2 (two-hop intermediate swap) **not replicated**: swapping the answer token beats swapping the intermediate (4B 85.4% vs 50.0%), and flips track direction cosine rather than source identity.
  - C3 (rhyme planning) and C4 (mental arithmetic) not replicated.
  - C5: "the J-lens shows **no consistent advantage over the vanilla logit lens** on its own eval sets (typo at 4B: 26.0% vs 69.8%)."
  - C6 (ignition, capacity, selectivity) was **not tested** by design.
- **Transport-cone geometry.** J-transported token directions collapse to an effective dimension of 2–12 in every model (11–51×). The self-trained GPT-2 124M is the only counterexample (23.4 → 31.2).

### 2.5 Peter Flo (Harvard), `jspace-4b`: "causal but not selective"
- https://github.com/pgrindehollevik-harvard/jspace-4b (blog: https://pflo.org/jspace-4b.html) · 2026-08-07 · abstract accepted at NEMI 2026 · [READ: README, PREREGISTRATION, LEARNINGS, code]
- A pre-registered replication of the §3.5.2 top-k ablation on Qwen3-4B, using the Neuronpedia lens plus a self-fitted penultimate-target lens.
- "**Causally real.** In the lightest layer band, ablating the top-10 J-lens directions per position costs 16 points on a two-step reasoning control while exactly norm-matched random ablation costs zero."
- "**Not selective.** The same ablation changes 34 to 42 percent of ordinary next-token predictions on held-out text, roughly double the matched-random arm."
- "The workspace signature may be an emergent property of scale."
- They name two possible explanations: small models route routine prediction through the workspace, or the final-layer-target lens smears in last-block artifacts.
- Their pre-registration cites "the paper's own small-model warning about Haiku 4.5."

### 2.6 Nanda's group: Qwen 3.6 27B replication and follow-ups
- **Review** (1.3 above and the commentary PDF). They replicated verbal report (weak but positive), CKA ("two or three somewhat overlapping bands (four or five bands total), and are notably less clean than the paper's"), directed modulation (moderate), and the multilingual and typo evals. Poetry and arithmetic failed. The multi-hop swap was dominated by swapping the answer token. Lens settings: n = 25 prompts, and "n=10 is almost as good".
- **R-lens** (Camila Blank, Agam Bhatia, Neel Nanda), https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers · 2026-08-05 · [READ]
  - Adds LRP stop-gradients to the backward pass.
  - Relevant to the **layer-band** claim: "One conclusion you might draw… is that J-Lens is working as intended, and early layers do not contain the types of 'verbalizable representations' that define workspace content… However, the fact that R-Lens works shows this is clearly false."
  - R-space CKA shows about 2–3 bands versus J-space's 4–5 on Qwen3.6-27B.
  - **MLP-gain replication (their appendix):** on Qwen3.6-27B, "J-lens and R-lens directions are similarly amplified across layers, and these are significantly more amplified than the MLP neuron in later workspace layers."
  - "There is no R-lens advantage for both the smallest dense and MoE models we tested."
  - Comments:
    - **StellaAthena** (tuned-lens author): "many of the metrics in the J-Lens paper were either layer invariant by construction or would obviously face this pitfall" (the pitfall being that you can improve metrics by reporting layer K+2's answer as layer K's).
    - **Burny**: "J-Space paper from Anthropic broke some of these good practices of good science."
    - **Pranav Viswanath** asks whether the attention-gain (broadcast-head) test was also run.
- **Meta-tokens** (Bhatia, Blank, Nanda), 2026-07-20: https://www.lesswrong.com/posts/6ek6n7yZ5DzfarJHy [title and metadata only]
- **WorkspaceBench** (Blank, Bhatia, Euan Ong, Nanda), 2026-09-23: https://www.lesswrong.com/posts/Zeg2JztbdhguL48uH [READ intro]. A benchmark of 3,356 questions for workspace readers on Qwen3.6-27B. It "may need to be adapted for smaller or weaker models."

### 2.7 Elie Bakouch, "J-lens CKA explorer" (38 open models)
- https://eliebak.com/viz/jspace-open · date not stated (it mentions "Guest fits (July 2026)"; Hoel cited it on 07-13) · [READ, static text only; the heatmaps are JavaScript]
- Computes layer×layer and cross-model CKA of the J-lens **token-direction geometry**: 4,096 shared token strings, V_ℓ = (W_U[ids]·γ)J_ℓ, using the Neuronpedia fits.
- Hoel reads it as showing no sharp three-block structure in open models. I could not view the plots myself.

### 2.8 Other small-model engineering and diagnostic posts
- **willkn, "Anthropic's J-Lens: A Research Engineer's Analysis"** (GPT-2-medium), https://www.lesswrong.com/posts/vHxGD5HKsFuBStirq · 2026-07-24 · [READ]
  - "On GPT-2-medium, the raw fitted Jacobian underperforms logit lens on next-token faithfulness." Cause: "the Jacobian's dominant channels carry ~10× the gain of the residual pathway, misweighting structural tokens."
  - "A single-parameter shrinkage regularizer J + λI monotonically recovers faithfulness", beating the logit lens at layer 12 (0.294 vs 0.275).
  - Convergence follows a 1/√n law, saturating around 100 prompts. Layer 4 has about 4× the estimator error of layer 20. The Jacobian is "essentially full-rank".
- **Ameya Panchal, "The J-lens offset is the model's token frequency: z-scoring helps"**, https://www.lesswrong.com/posts/Qe3jdphWCg9kny3e5 · 2026-09-17 · [READ]
  - The J-lens per-token mean offset correlates with log frequency (Spearman 0.48 at mid-depth on Qwen3.5-4B, but −0.05 for the logit lens).
  - "On GPT-2 the same bias correlates with log frequency at r = 0.70."
  - Subtracting the offset hurts readouts. Z-scoring helps.
- **Caleb Briggs, "An Exploration of the J-lens"** (Gemma-4-31B), https://www.lesswrong.com/posts/iQDXuY5Mb5JzjA8xT · 2026-09-07 · [READ]
  - Gets the sensory/workspace/motor CKA pattern, with the seam at L25.
  - The CKA picture depends on how the final-norm diagonal gain is handled. Approximating through the pre-norm final residual gives a plot that "is clearly wrong". Some directions get squashed about 30× more than others.
  - This is a caution that the structural "bands" depend on methodological choices.
- **Pranav Viswanath, "Models are blind outside the J-space. NLAs aren't."** (Llama-3.3-70B), https://www.lesswrong.com/posts/LhDJdccLszLEAqgZ9 · 2026-07-08 · [READ]
  - Reproduces the 6–7% J-fraction (6.1% at r = 200).
  - The split result is sensitive to rank: at r = 200 the effect **flipped** when he went from 6 to 16 concepts. It only came out clean at r ≈ 2000, where the model names the J-part 80% of the time and the non-J part 0%.
  - "the mental workspace doesn't have a clean wall, it's more of a steady spectrum spreading across thousands of directions."
- **Ratnaditya J** (comment, `GeoGGFYaWoEW4wYNv` on the wesg post). A pre-registered audit of the Neuronpedia Qwen2.5-7B-it lens on a hint-following model organism: AUROC 0.746, versus 0.690 for a TF-IDF baseline. It does not beat the "word counter", but the signal survives removing the chain of thought (0.620). On control inputs the lens stayed at chance (reported in a comment on Nanda's review).
- **Hugging Face blog, David Louapre, "J-Space: Yet Another LLM Mind Reader?"**, https://huggingface.co/blog/dlouapre/j-space · 2026-07-13 · [FETCH-SUMMARY] An explainer with its own Gemma-3-4B demos: J-lens surfaces "spider" and "legs" at later layers during two-hop reasoning.
- **Kartikay Luthra, "J-space auditing might be unreliable"** (Qwen3-8B), https://www.lesswrong.com/posts/69fachkoAs2ZHeSst · 2026-09-16 · [READ intro] J-space readouts were very similar between reward-hacking and honest checkpoints, and added little for an LLM auditor.
- **Melchior de Polignac, "A Topic Detector, Not a Lie Detector"** (DeepSeek-R1-Distill-Qwen-14B), https://www.lesswrong.com/posts/GZCMmCHZiF8vhsczr · 2026-08-11 · [READ intro] J-lens separated guideline-sensitive topics from controls (AUC 0.97 on proper nouns) but tracked "a territory rather than a precise guideline-corrected claim."
- **Small repos:** `awdemos/jspace-toolkit` (auto-discovers the workspace band with CKA, kurtosis and accuracy; defaults to `sshleifer/tiny-gpt2`); `tsepokfun/j-lens` (gpt2-large/xl); `idhantgulati/j-lens` (qwen3.5-4B); `WeZZard/jlens-qwen36` (MLX; there is also a public visualiser at https://jlens.wezzard.com/); HF lenses `solarkyle/jspace-lenses` and `praxagent-org/jacobian-lens-qwen3.5-397b-a17b`. I did not check results in any of these.

### 2.9 arXiv follow-ups [abstracts READ via the arXiv API; full papers not read]
- **Wang & Reid, "Looped Transformers under the Jacobian Lens: Does the Global Workspace Survive Recurrence?"**, arXiv 2609.01924 (2026-09-01). Covers Ouro-2.6B and Huginn-0125 against Qwen3.6-27B: "a workspace forms in the iterated part of each architecture, but… recurrence changes how it can be accessed."
- **Yan et al., "Short Horizons and Sparse Concepts: a Mathematical View of the Readout in the J-lens"**, arXiv 2608.25347 (2026-08-26). Treats the J-lens as a "first-order causal transfer operator" and "an expectation over anticipated future readouts". Its Jacobian energy "concentrates in an extremely small proportion", which they decompose into short-horizon, sparse concept predictions. Relevant to the argument that properties follow from construction.
- **Gong & Wang, "The First Token Is a Clue: Verbalizing Multi-Token Concepts from the J-lens"**, arXiv 2608.31084 (Gemma-3-12B-IT, Llama-3.1-8B, Qwen3-14B).
- **Kowalski et al., "Measuring Activation Control in Large Language Models"**, arXiv 2608.21664. Most LLMs can steer their residual stream on instruction, and in simple tasks can evade monitors including the Jacobian lens. This bears on how specific "directed modulation" is.
- **Prosvirnin et al., "Silent Alarm: A J-Space Protocol…"**, arXiv 2607.12792 (Qwen3-1.7B/4B/8B, Gemma 2 9B).
- **Wu et al., "J-CoT: Chain-of-Thought in J-Space"**, arXiv 2607.21981.
- Also: Kawada & Kellis, "Evidence Integration in LLMs", arXiv 2609.04290 (uses a J-lens decomposition); Davide & Collova, "Endognostics… Minerva-7B", arXiv 2609.22219.
- **Background, predates the paper:** Rahbar, "The Ignition Index", arXiv 2608.05160 (submitted 2026-05-26). Fits sigmoid probe accuracy against signal strength across 11 models, including Pythia-410M. It finds "hypotheses linking ignition to model scale and signal strength were not confirmed." This suggests ignition-like transitions may not need scale.

---

## 3. Philosophers and consciousness scientists, plus background literature (priority 3)

### Direct reactions
- **Dehaene & Naccache** and **Eleos**: see §0 for URLs. Dehaene's public post (X mirror of Bluesky): https://x.com/StanDehaene/status/2074200352555966967. Text [READ via Zvi]: "now Anthropic researchers have discovered a similar workspace inside their large language model !"
- **Eleos on construction**, commentary PDF p. 23 [READ]: "finding that J-lens vectors are unusually influential—broadcast unusually widely—could, for example, be accounted for by the fact that they are all identified via the J-lens, which we should expect to identify vectors that are able to have large internal effects (even if they each do so in different ways). This is not to say that this paper's finding is trivial, far from it."
- **David Chalmers**: §1.1.
- **Anil Seth**: §1.5.
- **Robert Long**: §1.4.
- **Erik Hoel**: §1.2.
- **Bernard Baars, "Global Workspace Theory, AI, and the Lost Baby Test"**, https://bernardbaars.substack.com/p/global-workspace-theory-ai-and-the · 2026-08-13 · [NO ACCESS beyond the intro; paywalled] The visible intro warns against confusing "a useful model of consciousness with consciousness itself" (orrery analogy). The Digital Minds newsletter cites it as one of the "other commentators."
- **Eric Schwitzgebel (The Splintered Mind / eschwitz.substack.com)**: **I found no post about the J-space paper.** I checked his Substack archive from April to September 2026 and grepped the posts listed in the Digital Minds newsletter ("Do Computers Have the Wrong Substrate…", "The Cognitive Advantages of Being of Two Minds", "AI and Consciousness", "Bare Functionalism…"). None mention the workspace paper. Relevant background from him:
  - His Cambridge Element *AI and Consciousness: A Skeptical Overview*, announced at http://schwitzsplinters.blogspot.com/2026/04/ai-and-consciousness-skeptical-overview.html [READ ch. 1]. It notes that Dehaene, Lau & Kouider (2017) argued "with a few straightforward tweaks, self-driving cars could be conscious."
  - Michel cites a forthcoming "problem of minimal instantiation" paper by him.
  - A search snippet attributes to him the line "If the simplest version of Global Workspace Theory is correct, we can easily create a conscious machine." [SNIPPET, not verified]
- **Jonathan Birch, Hakwan Lau, Ned Block**: **nothing J-space-specific found** in several searches.
- **Digital Minds Newsletter #4**, "The J-Space Debate, Agent Swarms, and Pacing Frontier AI" (Viswanathan, Alexander, Saad et al.), https://www.digitalminds.news/p/the-j-space-debate-agent-swarms-and · 2026-09-17 · [READ] A good index of reactions. It summarises Hoel as: "the J-lens uses what a model may later say as evidence for a global workspace, making the argument partly circular and difficult to falsify."
- **Noa Weiss, "The State of AI Consciousness Research"**, https://www.lesswrong.com/posts/pxvWgtSjR4pmFoS7c · 2026-07-15 · [READ] A survey. It notes that the experiential-language flattening also happens for third-person narration, and that GW indicators "now have potential mechanistic support".

### Background: are GWT indicators too cheap?
- **Herzog, Esfeld & Gerstner (2007), "Consciousness & the small network argument"**, *Neural Networks* 20(9):1054–1056. https://pubmed.ncbi.nlm.nih.gov/17900860/ [SNIPPET] Most computational theories, GWT included, imply that networks of under 10 neurons can be conscious.
- **Doerig, Schurger & Herzog (2021), "Hard criteria for empirical theories of consciousness"**, *Cognitive Neuroscience* 12(2). https://www.tandfonline.com/doi/full/10.1080/17588928.2020.1772214 [SNIPPET]. As quoted in Michel [READ]: "a network consisting of two peripheral neurons connected to a small recurrent global workspace fulfills the criteria for consciousness proposed by GWT."
- **Goldstein & Kirk-Giannini, "A Case for AI Consciousness: Language Agents and Global Workspace Theory"**, arXiv 2410.11407 (2024-10-15); now published in *JCS* 33(7–8):61–96 (2026), per Chalmers's bibliography. [READ §9]
  - They argue language agents "might easily be made phenomenally conscious" if GWT is true.
  - §9 treats the **"small model objection"**: a five-neuron system with perceptual "modules", an ignition neuron and a broadcast neuron "might come close to satisfying our functional conditions."
  - Their reply: "The small model challenge affects most functional theories of consciousness, since most of these theories embrace functional roles that are elegant enough for a simple system to satisfy… simplicity is a theoretical virtue." They are "sympathetic" to the objection but treat it as pointing to a missing extra condition X.
- **Matthias Michel (MIT), "On Cheap Artificial Consciousness"**, https://philarchive.org/rec/MICOCA-3 · v1 2026-06-17, v2 2026-08-12 · [READ] Predates the paper and does not mention it.
  - Names the family of objections: "the small network argument…, the problem of minimal instantiation (Schwitzgebel, Forthcoming), the minimal implementation problem (Butlin et al. 2026), and the small model objection (Goldstein & Kirk-Giannini, Forthcoming)."
  - On GWT: "conditions 1 to 3 are easy to implement in simple neural networks." His answer separates **core** from **total** realizers: "the fact that one can build a workspace-like architecture into a simple artificial system reveals a good deal less than advertised, since the core realizer is not sufficient for consciousness."
- **Butlin, Long et al., "Identifying indicators of consciousness in AI systems"**, *Trends in Cognitive Sciences* (2026), https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(25)00286-4 [SNIPPET] Source of the "minimal implementation problem" label, per Michel. Also Butlin, Long et al. 2023, arXiv 2308.08708.
- **Larissa Albantakis, "On the utility of toy models for theories of consciousness"**, arXiv 2508.00190 (2025) [abstract READ]. Uses GWT and IIT toy models to probe what theories commit to. A useful framing for "run the criteria on a tiny model".
- **Dehaene, Lau & Kouider (2017), "What is consciousness, and could machines have it?"**, *Science* 358:486–492. Cited by Schwitzgebel and David Manheim (via Zvi) as the earlier claim that GW-style access could be built.

---

## 4. Media coverage and notable X threads (priority 4)

**Anthropic's own posts**
- Blog, "A global workspace in language models", 2026-07-06: https://www.anthropic.com/research/global-workspace
- X thread: https://x.com/AnthropicAI/status/2074185348142280912 (the follow-up tweet quoted by Hoel is `/status/2074185381637988750`: "we've found Claude has developed a mechanism for conscious access").

**X threads**
- Rob Long, criticising how Anthropic's comms framed the paper: https://x.com/rgblong/status/2074300916455670201
- Eleos TL;DR: https://x.com/rgblong/status/2074211176280690906
- Stanislas Dehaene (enthusiastic): https://x.com/StanDehaene/status/2074200352555966967
- Zvi: https://x.com/TheZvi/status/2074612488956424412
- Brian Roemmele X article, "Anthropic's 'Global Workspace' Paper Is Not Just Research — It's Peak Brand Strategy": https://x.com/BrianRoemmele/article/2074289052996829671 [SNIPPET; title only]
- janus, antra, thebes, Wyatt Walls (Qwen/Gemma J-space screenshots) and Sauers ("damn" when told not to think of the Golden Gate Bridge) are all quoted in Zvi's post. I have no direct URLs for them.

**Press**
- VentureBeat, Michael Nuñez, 2026-07-06: https://venturebeat.com/technology/anthropics-new-j-lens-reveals-a-silent-workspace-inside-claude-that-mirrors-a-leading-theory-of-consciousness. Straight explainer, no outside experts quoted. [FETCH-SUMMARY]
- Tom's Hardware: https://www.tomshardware.com/tech-industry/artificial-intelligence/anthropic-says-it-can-read-claudes-thoughts-as-detailed-in-new-research-paper-models-observed-to-have-a-global-workspace-revealing-more-of-what-makes-llms-tick. The "read Claude's thoughts" framing. [SNIPPET; body not retrieved]
- Forbes, John Werner, 2026-07-12: https://www.forbes.com/sites/johnwerner/2026/07/12/anthropic-illuminates-llm-j-space-with-j-lens/ [NO ACCESS, 403]
- WinBuzzer, 2026-07-07: https://winbuzzer.com/2026/07/07/anthropic-maps-claudes-hidden-workspace-with-j-lens-xcxwbn/ [SNIPPET]
- IBM Think, Dave Bergmann, 2026-08-10: https://www.ibm.com/think/news/what-anthropic-j-space-research-means-future-ai. Industry angle; IBM staff say "It's not really about 'This is AGI'". [FETCH-SUMMARY]
- Guardian, Anil Seth, 2026-07-15 (§1.5).
- The Economist briefing "The search for consciousness inside LLMs" (2026-08-20): https://www.economist.com/interactive/briefing/2026/08/20/the-search-for-consciousness-inside-llms. Also a cover leader, "Could AIs become conscious?" [NO ACCESS; described by the Digital Minds newsletter]
- Nature news, Mariana Lenharo, on AI-sentience debate funding: https://www.nature.com/articles/d41586-026-02300-2 [SNIPPET via newsletter; not confirmed J-space-specific]
- Other explainer or aggregator pages seen only in search: implicator.ai, AI Weekly, MindStudio, explainx.ai, corti.com, developersdigest, alphasignalai.substack.com, stuff.co.nz ("Baffling discovery has scientists asking if AI is already conscious"), and a Japanese DL reading-group deck on Docswell (2026-07-17).

**Blogs**
- Haroun Guessous's review, 2026-07-08: https://aaronqlf.github.io/pflioblog/blog/paper-review-claude-global-workspace/. Summary only, no critique. [FETCH-SUMMARY]
- The Post-Humanist, 2026-07-07: https://theposthumanist.substack.com/p/the-anthropic-global-workspace-paper. Strongly pro; argues demanding more evidence from AI than from humans is a double standard. [FETCH-SUMMARY]
- Chris Riley, "Anthropic's global workvibes theory" (Medium, July 2026): https://mchrisriley.medium.com/anthropics-global-workvibes-theory-f9fbba79bfd1 [NO ACCESS, 403; the title suggests a sceptical take]
- Saanya Ojha, "Mind the J-Space": https://saanyaojha.substack.com/p/mind-the-j-space [not read]

**Discussion forums**
- Hacker News, 467 points and 201 comments: https://news.ycombinator.com/item?id=48808002 (§1.7)

---

## 5. Synthesis against your three questions

**(a) Are the five functional criteria strong evidence of a workspace?**
- **Sceptics converge on one argument: the criteria partly follow from how the J-lens is built.**
  - Leech: 3 of 5 are "near-analytic".
  - Hoel: "strict dependency" on reportability.
  - Chalmers: the criteria are access-like, not workspace-specific. The J-space is a "minimal workspace", defined by being disposed to influence output.
  - Eleos: selecting for large internal effects will surface influential vectors.
  - Jason R Brown: a next-token buffer "from its construction".
  - Raul Cavalcante: "don't think of X" is a plausible-continuation effect.
  - Ethan Garcia: verbal report of injections depends on the prefill and on protocol order.
- **What even the sceptics count as not entailed by the estimator:**
  - task-gated entry;
  - intermediates appearing in the right causal order;
  - the Spanish→French double dissociation;
  - counterfactual reflection training (all Leech);
  - bandit and poetry swaps (Nanda);
  - Chalmers concedes a "minimal workspace".
- **Cheap-implementation background** (Herzog 2007; Doerig 2021; Goldstein & Kirk-Giannini's "small model objection"; Michel's "cheap artificial consciousness") says GWT-style functional criteria can in principle be met by very small systems. The arbiter comment on LessWrong applies this directly to the paper's properties.
- **Top-10 protection rule:** no one has published the argument that selectivity is an artifact of the rule. jspace-4b is the only relevant data (Qwen3-4B still not selective with the rule in place).

**(b) Do the structural properties matter?**
- **Capacity:** "built-in" by the k ≈ 25 sparse decomposition (Chalmers). Rob Long and thebes say capacity limits were under-explored.
- **Ignition:** "remains to be fully demonstrated" (Dehaene & Naccache, quoted by Chalmers). Hoel gives a deflationary reading. Shehu calls it the one decisive test that is missing. The Ignition Index paper suggests ignition-like transitions are generic to transformers and do not scale with size.
- **Broadcast:**
  - Leech and Eleos: MLP gain and broadcast heads follow from selecting by average Jacobian.
  - Chalmers: this is influence, not broadcast; beating 5 populations is not beating all.
  - phoenix: the hub property is read-side.
  - Nanda's group replicated MLP gain on Qwen3.6-27B for both J and R directions.
- **Layer band:**
  - Hoel: the motor boundary is post hoc, and open-model CKA lacks sharp blocks.
  - Nanda: Qwen shows 4–5 bands.
  - R-lens: the early "degenerate" region is partly a J-lens artifact.
  - Caleb Briggs: CKA depends on how the norm gain is handled.
  - Pranav Viswanath: no sharp J/non-J boundary by rank.

**(c) Small models**
- **GPT-2 small is a known weak spot for the J-lens:**
  - Bandularatne: 0/11 layers beat the logit lens.
  - tao-hpu: no advantage, glitch tokens dominate.
  - willkn: raw J worse than the logit lens on GPT-2-medium, fixed by J + λI.
  - Proposed causes: weight tying and a strong logit lens on GPT-2 (gpjt; Belrose et al. 2023), glitch-token or massive-activation dimensions, and token-frequency bias (phoenix, Panchal: r ≈ 0.67–0.70).
- **The GPT-2 architecture itself works:** Pythia-70m and a self-trained GPT-2 124M do show a J-lens advantage (tao-hpu).
- **Other small models:**
  - Qwen3-4B: causal but not selective (jspace-4b).
  - Qwen3-1.7B/4B: only the multi-fact edit replicates (tao-hpu).
  - Neuronpedia: small models are easy to oversteer.
  - R-lens: no gain on the smallest models.
- As williawa put it, the open question for tiny models is whether the high-level properties (mediating verbal report, selectivity) survive, not whether the lens can be fit.
- **Nobody I found has run the full five-criteria plus structural battery on GPT-2 small.** Your replication would appear to be the first.

---

## 6. Could not access or could not find
- Economist briefing and leader (paywall or blocked); Forbes (403); Chris Riley's Medium post (403); Baars's post (paywalled after the intro); Tom's Hardware body (not retrieved).
- X/Twitter pages were not read directly. Tweet texts come from quotations in Zvi's post and search snippets.
- No J-space-specific reactions found from Schwitzgebel, Birch, Lau or Block.
- No source found arguing that the top-10 protection rule manufactures selectivity.
- Could not confirm whether gpt2-small is selectable in Neuronpedia's interactive J-lens UI. The lens file itself is definitely on HF.
