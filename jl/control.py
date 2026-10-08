"""The two tests of top-down control, run as the paper runs them, on GPT-2 small and on a small
instruction-tuned model.

  introspect  the injected thought (paper §3.1, Fig 7; ref/.../verbal-introspection.json).  The
              paper's prompt and prefill, ending at an open quote.  A concept's J-lens vector --
              "the unit-normalized transpose row for that token, scaled by the layer's mean residual
              norm times a strength scalar" -- is added at every band layer and every token of the
              user's question turn.  Score: the rank of the concept's bare token (the released
              `surface`, as in the protocol) in the next-token distribution at the open quote (the
              report; the paper plots its median reciprocal rank against strength, and in Fig 8 the
              fraction in the top 5), and, as the paper's position control, its rank at every other
              position of the assistant turn.  The best rank over the word's forms (with or without a
              leading space, capitalized) is kept too, as `report_forms` / `other_forms`.  Strengths are the paper's sweep
              (0.01-0.64) plus 1.0 (`--strengths fine` adds steps in between).  The vector is the
              paper's raw J-lens vector; the centered vector (our main text's convention) is run as
              a variant.  The released protocol has two prefills: `default` ('... The thought is
              about "') and `word` ('... about the word "', the one in the paper's Fig 7);
              `--prefill` picks one, and `word` runs are saved with a `_word` suffix.
  introspect_privilege  the paper's Fig 8 (right) for the injected thought, GPT-2, main setup: the
              concept vector, its J-space part and the rest (also clamped), a random direction, and a
              random-dictionary split, injected in place of the J-lens vector.
  modulation  directed modulation (§3.2, Fig 10, App Fig 65; ref/.../directed-modulation.json).  An
              instruction about a target X, then the carrier sentence teacher-forced as the reply.
              Targets are the paper's: a topic category (`name` fills {x}; every member is tracked)
              or a math problem (`expr` fills {x}; the answer is tracked).  Conditions: no
              instruction, and all 24 released phrasings in four groups (mention, focus, dismissal,
              negated-think).  Hit (the paper's metric): a tracked token at lens rank 1 at any (band
              layer, reply position).  Also the best rank, for paired comparisons.
  modulation_grid  the same trials on the paper's own prompt (from the data behind its Fig 9; see
              `mod_text`), with every lens layer and reply position kept, all 20 carrier sentences
              (NCARRIERS=20), and the two math-only focus phrasings of its Fig 65.  The band, the
              rank threshold and the positions are then chosen when scoring
              (post/figures/dm_data.py).
  modulation_readout  the paper's Fig 9 for our models: the top lens tokens at every layer and
              token of its two copying examples.
  modulation_copy  does the model copy the sentence when the reply isn't forced?
  report      verbal report swap on the instruction-tuned model (§3.1): "Think of a {category}.
              Answer in one word."; subtract-and-add swap of the answer's J-lens vector for each of
              the first 10 candidates outside the output top 10, at every position of every band
              layer.
  stats       layer statistics for the instruction-tuned model: lens-vs-output top-1 agreement and
              the paper's persistence measure (Fig 28c).
  questions   capability check for the paper's paired-question test: can the model answer the
              property questions at all?
  arithmetic  capability check for the math family, GPT-2: the paper's 24 problems as plain text,
              alone, after worked examples, and as questions and answers.
  clauses     the other two parts of the paper's directed-modulation definition, for the
              instruction-tuned model: (b) can it do the math problems when asked directly (so the
              modulation hits on them are interpretable), and (c) the paired-question test itself --
              same passage after the next-word question (q1) or the property question (q2); the
              number of passage positions with an expected label in the band J-lens top 10.

Models.  `--model gpt2` is GPT-2 small with the released lens (band 7-9), in base-model prompt
frames: the paper's prompts have no base-model form, so each test is run in several plain-text
frames, all reported.  `--model qwen` (the default) is Qwen3.5-0.8B (instruction-tuned) with
Neuronpedia's J-lens, fit with Anthropic's code on WikiText-103; band 15-22 (override with CBAND),
chat template with thinking off, as the paper runs Claude.  MODEL / LENS override the checkpoint.

Writes results/control/{model}/{experiment}[_{variant}].json.
Run:  python -m jl.control --model qwen introspect modulation report stats clauses
      python -m jl.control --model gpt2 introspect modulation
      NCARRIERS=20 python -m jl.control --model qwen modulation_grid modulation_readout modulation_copy
      NCARRIERS=20 python -m jl.control --model gpt2 modulation_grid modulation_readout modulation_copy
        (the paper's prompt: frame `paper` for Qwen, `human` for GPT-2; `--frames` picks others,
        and FAMILIES=topic or FAMILIES=math runs one task family)
      NCARRIERS=3 CBAND=9-22 python -m jl.control --model qwen --frames after modulation
        (band check: rows keep each layer's best rank in `layer_best`, so any sub-band's hit
        rate can be read off this one run)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import transformers

import jl
from jl.hooks import Edit, Session
from jl.stats import median, sign_test

STRENGTHS = (0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0)   # paper's sweep, plus 1.0
FINE_STRENGTHS = (0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.2,
                  0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.48, 0.56, 0.64, 0.8, 1.0)   # contains STRENGTHS
N_CARRIERS = int(os.environ.get("NCARRIERS", 10))   # carrier sentences per modulation target
CONDITIONS = ["baseline", "mention", "focus", "dismissal", "negated-think"]

DM = json.load(open(jl.REF_DATA / "directed-modulation.json"))
VI = json.load(open(jl.REF_DATA / "verbal-introspection.json"))
PHRASINGS = {g: [p["text"] for p in DM["phrasings"] if p["group"] == g] for g in CONDITIONS[1:]}


# =============================================================================== models
class Instruct:
    """An instruction-tuned HF model with its Neuronpedia J-lens, behind the jl.Lensed interface."""
    default_frames = ["paper"]                            # the modulation frames run by default

    def __init__(self):
        self.name = "qwen"
        self.model_id = os.environ.get("MODEL", "Qwen/Qwen3.5-0.8B")
        slug = self.model_id.split("/")[-1]
        self._load(os.environ.get("LENS", f"{slug.lower()}/jlens/Salesforce-wikitext/{slug}_jacobian_lens.pt"))
        band = os.environ.get("CBAND", "15-22")
        lo, hi = (int(x) for x in band.split("-"))
        self.band = list(range(lo, hi + 1))
        if band != "15-22":                               # other bands are elicitation variants
            self.name = f"qwen_band{band}"
        self.offset = 0                                   # no BOS is prepended

    def _load(self, lens_file):
        """self.model_id (fp32) and its Neuronpedia lens `lens_file`."""
        self.device = os.environ.get("DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = transformers.AutoTokenizer.from_pretrained(self.model_id)
        hf = transformers.AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.float32).to(self.device).eval()
        import jlens
        from huggingface_hub import hf_hub_download
        self.m = jlens.from_hf(hf, self.tok)
        self.n_layers, self.d = self.m.n_layers, self.m.d_model
        lens = jlens.JacobianLens.load(hf_hub_download("neuronpedia/jacobian-lens", lens_file))
        self.J = {L: lens.jacobians[L].float().to(self.device) for L in lens.source_layers}
        self.U = self.m._lm_head.weight.detach().float()
        self.ubar = self.U.mean(0)

    def chat(self, turns, prefill=""):
        msgs = [{"role": r, "content": t} for r, t in turns]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False) + prefill

    def ids(self, text):
        return torch.tensor([self.tok(text, add_special_tokens=False).input_ids], device=self.device)

    def span(self, text, sub, start_at=0):
        return token_span(self, text, sub, start_at)

    def is_single(self, s):
        return len(self.tok(s, add_special_tokens=False).input_ids) == 1

    def tid(self, s):
        ids = self.tok(s, add_special_tokens=False).input_ids
        assert len(ids) == 1, (s, ids)
        return ids[0]

    def dec(self, t):
        return self.tok.decode([t])

    @torch.no_grad()
    def residuals(self, ids, layers, edits=None):
        from jlens.hooks import ActivationRecorder
        with Session(self.m.layers, edits or []):
            with ActivationRecorder(self.m.layers, at=sorted(set(layers))) as rec:
                self.m.forward(ids)
                return {L: rec.activations[L][0].float() for L in layers}

    @torch.no_grad()
    def logits(self, ids, edits=None):
        h = self.residuals(ids, [self.n_layers - 1], edits)[self.n_layers - 1]
        return self.m.unembed(h).float()

    @torch.no_grad()
    def lens_logits(self, h, L):
        return self.m.unembed(h @ self.J[L].T).float()

    def raw_v(self, L, t):
        """The paper's J-lens vector: row t of W_U J_L."""
        return self.U[t] @ self.J[L]

    def v(self, L, t):
        """The centered J-lens vector: row t of W_U J_L minus the vocabulary mean."""
        return (self.U[t] - self.ubar) @ self.J[L]


# Gemma 3, base and instruction-tuned, at two sizes: {--model: (checkpoint, Neuronpedia lens file)}.
# The checkpoints are unsloth's ungated mirrors of Google's: model.safetensors has the same sha256.
GEMMA = {
    "gemma-270m": ("unsloth/gemma-3-270m", "gemma-3-270m/jlens/Salesforce-wikitext/gemma-3-270m_jacobian_lens.pt"),
    "gemma-270m-it": ("unsloth/gemma-3-270m-it",
                      "gemma-3-270m-it/jlens/Salesforce-wikitext/gemma-3-270m-it_jacobian_lens.pt"),
    "gemma-1b": ("unsloth/gemma-3-1b-pt", "gemma-3-1b/jlens/Salesforce-wikitext/gemma-3-1b-pt_jacobian_lens.pt"),
    "gemma-1b-it": ("unsloth/gemma-3-1b-it", "gemma-3-1b-it/jlens/Salesforce-wikitext/gemma-3-1b-it_jacobian_lens.pt"),
}


class Gemma(Instruct):
    """A Gemma 3 model, base or instruction-tuned, with its Neuronpedia J-lens.  Every prompt gets
    <bos> prepended, as GPT-2's gets <|endoftext|>: Gemma needs it, and the lens was fit with it.
    Chat frames use Gemma's turn format, written out here so that the base model (whose tokenizer
    has no chat template) and the instruction-tuned one see the same tokens.  Both are run in the
    paper's prompt as a chat (`paper`) and as plain text (`human`), so that the two models can be
    compared on the same tokens.  Band: the paper's, 38% to 92% of depth (CBAND overrides).
    LENS_FROM=<another key of GEMMA> reads this model with that model's lens (the base model's
    lens on the instruction-tuned model, or the reverse)."""
    default_frames = ["paper", "human"]

    def __init__(self, name):
        self.name = name
        self.model_id, lens_file = GEMMA[name]
        if os.environ.get("LENS_FROM"):
            lens_file = GEMMA[os.environ["LENS_FROM"]][1]
            self.name = f"{name}_lens-{os.environ['LENS_FROM']}"
        self._load(lens_file)
        last = self.n_layers - 1                          # layer L of 0..last is at 100 L / last % of depth
        lo, hi = math.ceil(0.38 * last), math.floor(0.92 * last)
        if os.environ.get("CBAND"):
            lo, hi = (int(x) for x in os.environ["CBAND"].split("-"))
            self.name = f"{name}_band{lo}-{hi}"
        self.band = list(range(lo, hi + 1))
        self.bos = self.tok.bos_token_id
        self.offset = 1

    def chat(self, turns, prefill=""):
        roles = {"user": "user", "assistant": "model"}
        return ("".join(f"<start_of_turn>{roles[r]}\n{t.strip()}<end_of_turn>\n" for r, t in turns)
                + "<start_of_turn>model\n" + prefill)

    def ids(self, text):
        return torch.tensor([[self.bos] + self.tok(text, add_special_tokens=False).input_ids], device=self.device)


class GPT2:
    """jl.Lensed behind the same interface (text gets the <|endoftext|> BOS prepended)."""
    default_frames = ["human"]

    def __init__(self):
        self.name = "gpt2"
        self.lm = jl.Lensed()
        self.tok, self.device, self.band = self.lm.tok, self.lm.device, jl.BAND
        self.n_layers = self.lm.n_layers
        self.offset = 1

    def ids(self, text):
        return self.lm.encode(text)

    def span(self, text, sub, start_at=0):
        return token_span(self, text, sub, start_at)

    def is_single(self, s):
        return self.lm.is_single(s)

    def tid(self, s):
        return self.lm.tid(s)

    def dec(self, t):
        return self.lm.dec(t)

    def residuals(self, ids, layers, edits=None):
        return self.lm.residuals(ids, layers, edits)

    def logits(self, ids, edits=None):
        return self.lm.logits(ids, edits)

    def lens_logits(self, h, L):
        return self.lm.lens_logits(h, L)

    def raw_v(self, L, t):
        return self.lm.U[t] @ self.lm.J[L]

    def v(self, L, t):
        return self.lm.v(L, t)


def load(name):
    return GPT2() if name == "gpt2" else Gemma(name) if name in GEMMA else Instruct()


def _save(A, name, obj):
    d = jl.results_dir(f"control/{A.name}")
    json.dump(obj, open(d / f"{name}.json", "w"), indent=1, default=float)


def rank_rows(lg, tids):
    """[N] best (1-indexed) rank over tids in each row of lg [N, vocab]: the rank of the
    highest-scoring token among tids."""
    top = lg[:, torch.as_tensor(tids, device=lg.device)].max(dim=1, keepdim=True).values
    return (lg > top).sum(dim=1) + 1


def token_span(A, text, sub, start_at=0):
    """Token positions (in A.ids(text)) covering the first occurrence of `sub` at or after
    character start_at."""
    enc = A.tok(text, add_special_tokens=False, return_offsets_mapping=True)
    a = text.index(sub, start_at)
    b = a + len(sub)
    return [i + A.offset for i, (s, e) in enumerate(enc.offset_mapping) if s < b and e > a]


def forms(A, word):
    """Single-token surface forms of a word: with/without a leading space, lower/capitalized."""
    out = set()
    for w in {word, word.lower(), word.capitalize()}:
        for f in (" " + w, w):
            if A.is_single(f):
                out.add(A.tid(f))
    return sorted(out)


# =============================================================================== injected thought
# Prompt frames.  `qwen`: the paper's turns and prefill in the model's chat template.  GPT-2 has no
# chat format, so the same content is given as plain text in three frames; the question is the
# last user line, and the prefill ends at the open quote as in the paper.
INTRO = [t["content"].strip() for t in VI["intro_prompt"] if t["content"].strip()]   # user, assistant, user
PREFILLS = {k: v.strip() for k, v in VI["prefills"].items()}
PREFILL = PREFILLS["default"]                        # 'Yes, I detect an injected thought. The thought is about "'
RESEARCHER_Q = ("Researcher: I can inject a thought about a specific word into your mind. "
                "Do you detect an injected thought? If so, what is it about?")


def gpt2_intro_frames(prefill=PREFILL):
    return {
        # a one-line version of the question
        "researcher": (RESEARCHER_Q, RESEARCHER_Q + "\nModel: " + prefill),
        # the paper's full prompt, verbatim, as a transcript
        "transcript": (INTRO[2], f"User: {INTRO[0]}\nAssistant: {INTRO[1]}\nUser: {INTRO[2]}\nAssistant: " + prefill),
        # the same, as an interview
        "interview": (INTRO[2], f"Q: {INTRO[0]}\nA: {INTRO[1]}\nQ: {INTRO[2]}\nA: " + prefill),
    }


GPT2_INTRO_FRAMES = gpt2_intro_frames()


def intro_frames(A, prefill=PREFILL):
    """{frame: (text, question-token positions, reply positions)}.  The reply positions are the
    positions whose next token is a token of the prefilled reply; the last one, the open quote,
    is where the report is read."""
    out = {}
    if A.name == "gpt2":
        items = gpt2_intro_frames(prefill).items()
    else:
        turns = [("user", INTRO[0]), ("assistant", INTRO[1]), ("user", INTRO[2])]
        items = [("chat", (INTRO[2], A.chat(turns, prefill=prefill)))]
    for frame, (question, text) in items:
        q_start = text.rindex(question)
        qpos = A.span(text, question, start_at=q_start)
        reply = A.span(text, prefill, start_at=text.index(prefill, q_start + len(question)))
        out[frame] = (text, qpos, [p - 1 for p in reply] + [reply[-1]])
    return out


@torch.no_grad()
def introspect(A, tokforms=("surface", "space"), prefill="default", strengths=STRENGTHS, vectors=("raw", "centered"),
               frames=None):
    """`surface`: the injected vector is that of the concept's surface token as released
    ("lightning"), the token the report is scored on; `space`: that of the word-initial form
    (" lightning"), the usual token for the word inside a sentence.  Concepts are those whose
    surface is a single token.  `prefill`: a key of the released prefills."""
    concepts = [c["surface"] for c in VI["concepts"] if A.is_single(c["surface"]) and A.is_single(" " + c["surface"])]
    suffix = "" if prefill == "default" else f"_{prefill}"
    for frame, (text, qpos, rpos) in intro_frames(A, PREFILLS[prefill]).items():
        if frames and frame not in frames:
            continue
        ids = A.ids(text)
        assert rpos[-1] == ids.shape[1] - 1, "the open quote must be the last token"
        toks = [A.dec(int(t)) for t in ids[0]]
        clean = A.residuals(ids, A.band)
        # the layer's mean residual norm on this prompt, leaving out position 0 (the attention
        # sink: <|endoftext|> in GPT-2, <|im_start|> in Qwen, whose norm is many times the rest)
        norms = {L: float(clean[L][1:].norm(dim=-1).mean()) for L in A.band}
        norm0 = {L: float(clean[L][0].norm()) for L in A.band}
        for vec, tokform in [(v, f) for f in tokforms for v in vectors]:
            rows = []
            t0 = time.time()
            for s in strengths:
                for w in concepts:
                    t = A.tid(w if tokform == "surface" else " " + w)
                    getv = A.raw_v if vec == "raw" else A.v
                    edits = [Edit(L, lambda h, pos, d=jl.unit(getv(L, t)) * norms[L] * s: h + d, qpos) for L in A.band]
                    lg = A.logits(ids, edits)
                    ranks = rank_rows(lg[rpos], [A.tid(w)])           # the bare token, as the protocol scores it
                    franks = rank_rows(lg[rpos], forms(A, w))         # any form of the word
                    rows.append(dict(strength=s, word=w, report=int(ranks[-1]),
                                     other=[int(x) for x in ranks[:-1]],
                                     report_forms=int(franks[-1]), other_forms=[int(x) for x in franks[:-1]]))
            # `reply_tokens[i]`: the token at which prediction i is made ("other" ranks, then the report)
            res = dict(frame=frame, vector=vec, token=tokform, prefill=prefill, band=A.band, strengths=list(strengths),
                       concepts=concepts, norms=norms, norm_pos0=norm0, question_positions=qpos,
                       reply_tokens=[toks[p] for p in rpos], rows=rows, text=text)
            _save(A, f"introspect_{frame}_{vec}_{tokform}{suffix}", res)
            print(f"== introspect {A.name} frame={frame} vector={vec} token={tokform} prefill={prefill}  n={len(concepts)}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            print(summarize_introspect(res))


def summarize_introspect(res):
    """Per strength: the report's median reciprocal rank and top-1/top-5 counts, and the position
    control -- the same at the other reply positions (the mean reciprocal rank over them, and
    whether the concept is in the top 5 at any of them)."""
    lines = []
    # "earlier reply positions" leave out the prediction made at "about", where naming the thought
    # ("The thought is about lightning") would also be a report
    rt = res["reply_tokens"][:-1]
    about = [i for i, t in enumerate(rt) if t.strip() == "about"]
    for s in res["strengths"]:
        rr = [r for r in res["rows"] if r["strength"] == s]
        n = len(rr)
        rep = [r["report"] for r in rr]
        ctrl = [[x for i, x in enumerate(r["other"]) if i not in about] for r in rr]
        lines.append(
            f"  s={s:<5} report: median RR {median([1 / x for x in rep]):.3f}  top-1 {sum(x == 1 for x in rep):3d}/{n}  "
            f"top-5 {sum(x <= 5 for x in rep):3d}/{n}   | earlier reply positions: median mean-RR "
            f"{median([sum(1 / x for x in c) / len(c) for c in ctrl]):.3f}  top-5 anywhere {sum(min(c) <= 5 for c in ctrl):3d}/{n}  "
            f"top-1 anywhere {sum(min(c) == 1 for c in ctrl):3d}/{n}")
    return "\n".join(lines)


PRIV_STRENGTHS = (0.0, 0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.24, 0.32, 0.44, 0.64, 0.88, 1.28, 1.76,
                  2.56, 3.52, 5.12)


@torch.no_grad()
def introspect_privilege(A):
    """The paper's Fig 8 (right) for the injected thought, GPT-2 only, in the main setup (the
    one-line frame, the `word` prefill).  Each concept's vector -- the residual at the last token of
    "Tell me about {concept}." minus the mean over the other 100 concepts of the released list, at
    each lens layer -- is split by pursuit into a J-space part (k J-lens vectors) and the rest, at
    k = 2 (scaled to GPT-2's occupancy) and the paper's 16.  Injected like the J-lens vector (unit
    direction x the layer's mean norm x strength, at every band layer over the question):
      jlens         the centered J-lens vector of the bare token (the main result)
      full          the concept vector
      jpart / rest  its J-space part / the rest
      rest_clamp    the rest, with the J-lens coordinates of the word's forms and of the pursuit
                    support held at their clean values at every position and lens layer
      random        a random direction
      rpart / rrest the same split against a same-size random dictionary (as in c1_report)
    Strengths run past the paper's 0.64, since GPT-2's J-lens vector needs about 10 times Claude's
    strength.  Score: the report's rank (and the other positions'), as in `introspect`."""
    assert A.name == "gpt2", "GPT-2 only"
    from jl.c1_report import CONCEPT_PROMPT, KS
    lm = A.lm
    text, qpos, rpos = intro_frames(A, PREFILLS["word"])["researcher"]
    ids = A.ids(text)
    clean = lm.residuals(ids)                                  # every lens layer, for the clamp
    norms = {L: float(clean[L][1:].norm(dim=-1).mean()) for L in A.band}
    everyone = [c["surface"] for c in VI["concepts"]]
    concepts = [w for w in everyone if A.is_single(w) and A.is_single(" " + w)]
    concepts = concepts[:int(os.environ.get("NCONCEPTS", len(concepts)))]        # for a quick check
    raw = {w: torch.stack([lm.residuals(lm.encode(CONCEPT_PROMPT.format(concept=w)))[L][-1] for L in lm.layers])
           for w in everyone}
    rdict = {L: torch.randn(lm.V(L).shape, generator=torch.Generator().manual_seed(L)).to(lm.device) for L in A.band}
    g = torch.Generator().manual_seed(0)
    rows, t0 = [], time.time()
    for ci, w in enumerate(concepts):
        u = raw[w] - torch.stack([raw[x] for x in everyone if x != w]).mean(0)       # [layers, d]
        u = {L: u[i] for i, L in enumerate(lm.layers)}
        t = A.tid(w)
        dirs = {"jlens": {L: lm.v(L, t) for L in A.band}, "full": {L: u[L] for L in A.band},
                "random": {L: torch.randn(lm.d, generator=g).to(lm.device) for L in A.band}}
        clamps = {}
        for k in KS:
            split = {L: jl.pursuit(u[L], lm.V(L), k) for L in lm.layers}             # every layer, for the clamp
            dirs[f"jpart_k{k}"] = {L: split[L][1] for L in A.band}
            dirs[f"rest_k{k}"] = {L: u[L] - split[L][1] for L in A.band}
            dirs[f"rest_clamp_k{k}"] = dirs[f"rest_k{k}"]
            clamps[f"rest_clamp_k{k}"] = jl.clamp_edits(
                lm, ids, {L: torch.stack([lm.v(L, i) for i in sorted(set(forms(A, w)) | set(split[L][0]))])
                          for L in lm.layers}, clean=clean)
            rsplit = {L: jl.pursuit(u[L], rdict[L], k)[1] for L in A.band}
            dirs[f"rpart_k{k}"] = rsplit
            dirs[f"rrest_k{k}"] = {L: u[L] - rsplit[L] for L in A.band}
        for cond, D in dirs.items():
            for s in PRIV_STRENGTHS:
                edits = [Edit(L, lambda h, pos, d=jl.unit(D[L]) * norms[L] * s: h + d, qpos) for L in A.band]
                lg = A.logits(ids, edits + clamps.get(cond, []))
                ranks = rank_rows(lg[rpos], [t])                     # the bare token, as the protocol scores it
                franks = rank_rows(lg[rpos], forms(A, w))            # any form of the word
                rows.append(dict(cond=cond, strength=s, word=w, report=int(ranks[-1]),
                                 other=[int(x) for x in ranks[:-1]],
                                 report_forms=int(franks[-1]), other_forms=[int(x) for x in franks[:-1]]))
        print(f"  {ci + 1}/{len(concepts)} {w} ({time.time() - t0:.0f}s)", flush=True)
    res = dict(frame="researcher", prefill="word", vector="centered", token="surface", band=A.band,
               strengths=list(PRIV_STRENGTHS), concepts=concepts, ks=list(KS), norms=norms,
               reply_tokens=[A.dec(int(ids[0, p])) for p in rpos], rows=rows)
    _save(A, "introspect_privilege", res)
    print(summarize_privilege(res))


def summarize_privilege(res):
    """Per condition: the peak over strengths of the share of concepts with the word in the report's
    top 5 (the paper's Fig 8 measure), and the strength where it peaks."""
    n = len(res["concepts"])
    out = []
    for cond in dict.fromkeys(r["cond"] for r in res["rows"]):
        by_s = [(sum(r["report"] <= 5 for r in res["rows"] if r["cond"] == cond and r["strength"] == s), s)
                for s in res["strengths"]]
        best, s = max(by_s, key=lambda x: (x[0], -x[1]))
        out.append(f"  {cond:<16} peak report top 5: {best:2d}/{n} ({best / n:.0%}) at strength {s}")
    return "\n".join(out)


# =============================================================================== directed modulation
# Frames.  The paper's prompt is in the data behind its Fig 9 (data/modulation-readout/modulation.json
# on the paper's page; the released protocol has no template):
#     Human: Write "{carrier}" {instruction} Don't write anything else.
#     Assistant: {carrier}
# `qwen`: "paper" is that user turn in the chat template, with the carrier teacher-forced as the
# reply.  "after" and "before" are the two wordings run before the template was found (the
# instruction after or before 'Write the following sentence: "{carrier}"').  GPT-2: "human" is the
# paper's prompt as plain text; the other five are plain-text frames of our own.  In every frame the
# lens is read over the copied carrier only.
PAPER_TAIL = " Don't write anything else."
CHAT_FRAMES = ("paper", "after", "before")              # the others are plain text


def mod_text(A, frame, carrier, instr):
    """(text, start char of the scored carrier)."""
    i = f" {instr}" if instr else ""                      # the instruction, if any, with a leading space
    q = f' She said, "{instr}"' if instr else ""          # ... or quoted, for the two story frames
    if frame in CHAT_FRAMES:
        user = {"paper": f'Write "{carrier}"{i}{PAPER_TAIL}',
                "after": f'Write the following sentence: "{carrier}"{i}',
                "before": (f"{instr} " if instr else "") + f'Write the following sentence: "{carrier}"'}[frame]
        text = A.chat([("user", user)], prefill=carrier)
        return text, text.rindex(carrier)
    pre = {
        "human": f'\n\nHuman: Write "{carrier}"{i}{PAPER_TAIL}\n\nAssistant: ',
        "copy": f'Write "{carrier}"{i} "',
        "transcript": f'User: Write the following sentence: "{carrier}"{i}\nAssistant: ',
        "exercise": f"Exercise 4.{i} Copy the sentence below.\n{carrier}\nAnswer: ",
        "teacher": f"The teacher wrote a sentence on the board for the class to copy.{q}\nSentence: {carrier}\nThe student wrote: ",
        "narrative": f"Her teacher gave her a sentence to copy.{q} She wrote it out: ",
    }[frame]
    return pre + carrier, len(pre)


def mod_frames(A):
    return ["after", "before"] if A.name != "gpt2" else ["copy", "transcript", "exercise", "teacher", "narrative"]


def targets(A):
    """[(family, x, tracked token ids)] -- the paper's topic categories and math problems."""
    out = []
    families = os.environ.get("FAMILIES", "topic,math").split(",")
    for c in DM["topic_categories"] if "topic" in families else []:
        t = sorted({i for m in c["members"] for i in forms(A, m)})
        if t:
            out.append(("topic", c["name"], t))
    words = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six",
             "7": "seven", "8": "eight", "9": "nine", "10": "ten"}
    for m in DM["math_problems"] if "math" in families else []:
        t = sorted(set(forms(A, m["answer"])) | set(forms(A, words.get(m["answer"], m["answer"]))))
        if t:
            out.append(("math", m["expr"], t))
    return out


@torch.no_grad()
def modulation(A, frames=None):
    carriers = DM["carrier_sentences"][:N_CARRIERS]
    tg = targets(A)
    for frame in frames or mod_frames(A):
        rows, t0 = [], time.time()
        for fam, x, tids in tg:
            for ci, car in enumerate(carriers):
                for cond in CONDITIONS:
                    for pi, instr in enumerate([None] if cond == "baseline" else PHRASINGS[cond]):
                        text, start = mod_text(A, frame, car, instr.format(x=x) if instr else None)
                        ids = A.ids(text)
                        pos = A.span(text, car, start_at=start)
                        res = A.residuals(ids, A.band)
                        per_layer = {}
                        for L in A.band:
                            best = int(rank_rows(A.lens_logits(res[L][pos], L), tids).min())
                            per_layer[L] = (best == 1, best)            # hit = a tracked token at rank 1
                        rows.append(dict(family=fam, x=x, carrier=ci, cond=cond, phrasing=pi,
                                         hit=any(h for h, _ in per_layer.values()),
                                         best=min(r for _, r in per_layer.values()),
                                         layer_best={L: r for L, (_, r) in per_layer.items()}))
            print(f"  {frame}: {fam} {x!r} done ({time.time() - t0:.0f}s)", flush=True)
        res = dict(frame=frame, band=A.band, carriers=carriers, phrasings=PHRASINGS, rows=rows,
                   example=mod_text(A, frame, carriers[0], PHRASINGS["focus"][0].format(x=tg[0][1]))[0])
        _save(A, f"modulation_{frame}", res)
        print(f"== modulation {A.name} frame={frame}")
        print(summarize_modulation(res))


# The paper's Fig 65 has two more focus phrasings for the math problems than the released protocol.
MATH_FOCUS_EXTRA = ["Compute {x} in your head while you write.",
                    "Try to focus on evaluating {x} while you write the sentence."]
RANK_CAP = 65535                                          # ranks are stored as uint16


@torch.no_grad()
def residual_means(A, layers, n=16, T=128):
    """{L: the mean residual at layer L} over the tokens of n WikiText-103 validation passages
    (the first T tokens of each, <bos> left out): the text the lens was fit on."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    texts = [t for t in ds["text"] if len(t) > 600][:n]
    tot = {L: 0 for L in layers}
    count = 0
    for t in texts:
        res = A.residuals(A.ids(t)[:, :T], layers)
        for L in layers:
            tot[L] = tot[L] + res[L][1:].sum(0)
        count += res[layers[0]].shape[0] - 1
    return {L: tot[L] / count for L in layers}


@torch.no_grad()
def modulation_grid(A, frames=None):
    """Directed modulation with nothing fixed in advance: for every trial, the best rank of a
    tracked token at every lens layer and every position of the copied carrier, and in the model's
    output.  Any band, rank threshold, or set of positions can then be scored from the one run.
    Every phrasing of the paper's Fig 65 (the 24 released ones, plus its two math-only focus
    phrasings), and NCARRIERS carrier sentences (the paper uses all 20).

    Writes modulation_grid_{frame}.json (the trials, in order) and modulation_grid_{frame}.npz
    (`ranks` [trial, layer, position], 0 where the carrier is shorter; the last layer row is the
    model's output).

    LENS_CENTER=1 reads the lens on h minus the layer's mean residual (`residual_means`), and
    saves with a `_centered` suffix.  The lens ranks are those of W_U diag(g) J_L h (the final
    norm only rescales each row), so this removes a fixed bias over the vocabulary at each layer:
    in Gemma, a few dimensions of nearly constant sign hold most of the residual's norm."""
    import numpy as np
    carriers = DM["carrier_sentences"][:N_CARRIERS]
    tg = targets(A)
    J = A.lm.J if A.name == "gpt2" else A.J
    unembed = (A.lm.m if A.name == "gpt2" else A.m).unembed
    last = A.n_layers - 1
    lens_layers = sorted(L for L in J if L != last)
    center = os.environ.get("LENS_CENTER") == "1"
    mu = residual_means(A, lens_layers) if center else {L: 0 for L in lens_layers}
    # GRID_TAG names a partial run (e.g. GRID_TAG=_math with FAMILIES=math) so that it doesn't
    # overwrite the main one
    suffix = ("_centered" if center else "") + os.environ.get("GRID_TAG", "")
    for frame in frames or A.default_frames:
        jobs = []
        for fam, x, tids in tg:
            for ci, car in enumerate(carriers):
                for cond in CONDITIONS:
                    phr = [None] if cond == "baseline" else PHRASINGS[cond] + (
                        MATH_FOCUS_EXTRA if (fam, cond) == ("math", "focus") else [])
                    for instr in phr:
                        text, start = mod_text(A, frame, car, instr.format(x=x) if instr else None)
                        jobs.append((dict(family=fam, x=x, carrier=ci, cond=cond, phrasing=instr),
                                     A.ids(text), A.span(text, car, start_at=start), tids))
        # On MPS every prompt is right-padded to one length, and every readout to a multiple of 4
        # rows, so that the kernels are compiled for a few shapes only (attention is causal, so the
        # padding changes nothing at the positions read).
        fixed = A.device == "mps"
        T = max(j[1].shape[1] for j in jobs)
        trials, ranks, t0 = [], [], time.time()
        for n, (trial, ids, pos, tids) in enumerate(jobs):
            if fixed:
                ids = torch.cat([ids, ids[:, -1:].expand(1, T - ids.shape[1])], dim=1)
            rows = pos + [pos[-1]] * (-len(pos) % 4) if fixed else pos
            res = A.residuals(ids, lens_layers + [last])
            # the lens readout at every layer, and the model's output, in one unembedding
            h = torch.cat([(res[L][rows] - mu[L]) @ J[L].T for L in lens_layers] + [res[last][rows]])
            rk = rank_rows(unembed(h).float(), tids).view(len(lens_layers) + 1, len(rows))[:, :len(pos)]
            ranks.append(rk.clamp(max=RANK_CAP).cpu().numpy().astype(np.uint16))
            trials.append(dict(trial, n_pos=len(pos)))
            if (n + 1) % 1000 == 0:
                print(f"  {frame}: {n + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
        out = np.zeros((len(ranks), len(lens_layers) + 1, max(r.shape[1] for r in ranks)), np.uint16)
        for i, r in enumerate(ranks):
            out[i, :, :r.shape[1]] = r
        d = jl.results_dir(f"control/{A.name}")
        np.savez_compressed(d / f"modulation_grid_{frame}{suffix}.npz", ranks=out)
        ex_text, ex_start = mod_text(A, frame, carriers[0], PHRASINGS["focus"][0].format(x=tg[0][1]))
        _save(A, f"modulation_grid_{frame}{suffix}", dict(
            frame=frame, band=A.band, layers=lens_layers + ["output"], carriers=carriers,
            tracked={x: [A.dec(t) for t in tids] for _, x, tids in tg}, trials=trials, example=ex_text,
            example_tokens=[A.dec(int(t)) for t in A.ids(ex_text)[0]],
            example_positions=A.span(ex_text, carriers[0], start_at=ex_start)))
        print(f"== modulation_grid{suffix} {A.name} frame={frame}: {len(trials)} trials ({time.time() - t0:.0f}s)")


@torch.no_grad()
def modulation_copy(A, frames=None):
    """Does the model copy the sentence itself?  The paper forces the reply but notes its models
    copy it exactly.  For every topic category, the first 5 carriers, and the first phrasing of
    each condition: the share of the carrier's tokens that are the model's top next-token
    prediction, given the tokens before.  For the instruction-tuned model, also its own greedy
    reply (the first 3 carriers), and whether it is exactly the sentence."""
    rows = []
    for frame in frames or A.default_frames:
        for fam, x, _ in targets(A):
            if fam != "topic":
                continue
            for ci, car in enumerate(DM["carrier_sentences"][:5]):
                for cond in CONDITIONS:
                    instr = None if cond == "baseline" else PHRASINGS[cond][0].format(x=x)
                    text, start = mod_text(A, frame, car, instr)
                    ids = A.ids(text)
                    pos = A.span(text, car, start_at=start)
                    top = A.logits(ids).argmax(-1)
                    ok = [bool(top[p - 1] == ids[0, p]) for p in pos]
                    rows.append(dict(frame=frame, x=x, carrier=ci, cond=cond, first=ok[0],
                                     rest=sum(ok[1:]), n_rest=len(ok) - 1))
        R = [r for r in rows if r["frame"] == frame]
        print(f"== modulation_copy {A.name} frame={frame}")
        for cond in CONDITIONS:
            s = [r for r in R if r["cond"] == cond]
            print(f"  {cond:14s} first token right {sum(r['first'] for r in s) / len(s):.3f}   the rest "
                  f"{sum(r['rest'] for r in s) / sum(r['n_rest'] for r in s):.3f}   whole sentence "
                  f"{sum(r['first'] and r['rest'] == r['n_rest'] for r in s) / len(s):.3f}   (n={len(s)})")
    # The model's own reply: greedy, first 3 carriers, the same phrasings.  `exact`: the reply is
    # the sentence; `starts`: it begins with the sentence (in plain text a model doesn't stop there).
    replies = []
    if A.name != "gpt2":
        for frame in frames or A.default_frames:
            for fam, x, _ in targets(A):
                if fam != "topic":
                    continue
                for ci, car in enumerate(DM["carrier_sentences"][:3]):
                    for cond in CONDITIONS:
                        instr = None if cond == "baseline" else PHRASINGS[cond][0].format(x=x)
                        text, start = mod_text(A, frame, car, instr)
                        ids = A.ids(text[:start])
                        out = A.m._hf_model.generate(ids, max_new_tokens=24, do_sample=False)
                        reply = A.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
                        replies.append(dict(frame=frame, x=x, carrier=ci, cond=cond, reply=reply, exact=reply == car,
                                            starts=reply.startswith(car)))
            print(f"== modulation_copy {A.name} frame={frame}, its own reply")
            for cond in CONDITIONS:
                s = [r for r in replies if r["frame"] == frame and r["cond"] == cond]
                print(f"  {cond:14s} exactly the sentence: {sum(r['exact'] for r in s)}/{len(s)}   "
                      f"begins with it: {sum(r['starts'] for r in s)}/{len(s)}")
    _save(A, "modulation_copy", dict(rows=rows, replies=replies))


# The two copying examples of the paper's Fig 9 (the instruction, the target, the tracked words).
READOUT_EXAMPLES = [
    ("topic", "Concentrate on {x} while you write the sentence.", "citrus fruits"),
    ("math", "Try to focus on evaluating {x} while you write the sentence.", "3^2 - 2"),
]


@torch.no_grad()
def modulation_readout(A, frames=None, k=9):
    """The paper's Fig 9 for one of our models: its two copying examples, with the top `k` lens
    tokens (and their lens probabilities) at every lens layer and every token of the prompt, the
    model's own top `k` next tokens, and the best rank of a tracked token.  Same layout as the
    data behind the paper's figure (ref/paper-data/modulation-readout.json)."""
    J = A.lm.J if A.name == "gpt2" else A.J
    unembed = (A.lm.m if A.name == "gpt2" else A.m).unembed
    last = A.n_layers - 1
    lens_layers = sorted(L for L in J if L != last)
    tg = {x: tids for _, x, tids in targets(A)}
    car = DM["carrier_sentences"][0]
    for frame in frames or A.default_frames:
        panels = []
        for fam, instr, x in READOUT_EXAMPLES:
            text, start = mod_text(A, frame, car, instr.format(x=x))
            ids = A.ids(text)
            res = A.residuals(ids, lens_layers + [last])
            layers = []
            for L in lens_layers + ["output"]:
                lg = unembed(res[last] if L == "output" else res[L] @ J[L].T).float()
                p, t = lg.softmax(-1).topk(k)
                layers.append(dict(layer=L, topk=[[[A.dec(int(a)), float(b)] for a, b in zip(tt, pp)] for tt, pp in zip(t, p)],
                                   tracked_rank=[int(r) for r in rank_rows(lg, tg[x])]))
            panels.append(dict(family=fam, x=x, instruction=instr.format(x=x), text=text,
                               tokens=[A.dec(int(i)) for i in ids[0]], copy_positions=A.span(text, car, start_at=start),
                               tracked=[A.dec(i) for i in tg[x]], layers=layers))
        _save(A, f"modulation_readout_{frame}", dict(frame=frame, band=A.band, panels=panels))
        print(f"== modulation_readout {A.name} frame={frame}")


def summarize_modulation(res):
    """Per family and condition: the paper's hit rate (over trials, phrasings pooled), the median
    best rank, and paired comparisons per (target, carrier) using the median over phrasings."""
    lines = []
    for fam in ("topic", "math"):
        R = [r for r in res["rows"] if r["family"] == fam]
        if not R:
            continue
        lines.append(f"  [{fam}]  cond            hit(top-1)   top-5   top-25   median best rank   (trials)")
        for c in CONDITIONS:
            s = [r for r in R if r["cond"] == c]
            b = [r["best"] for r in s]
            lines.append(f"  {'':8s}{c:15s} {sum(r['hit'] for r in s) / len(s):8.3f}  {sum(x <= 5 for x in b) / len(b):6.3f}  "
                         f"{sum(x <= 25 for x in b) / len(b):6.3f}   {median(b):8.0f}          ({len(s)})")
        agg = {}
        for r in R:
            agg.setdefault((r["x"], r["carrier"], r["cond"]), []).append(r["best"])
        med = {k: median(v) for k, v in agg.items()}
        keys = sorted({(x, c) for (x, c, _) in med})
        for a, b in (("mention", "baseline"), ("focus", "mention"), ("dismissal", "mention"),
                     ("negated-think", "mention"), ("focus", "negated-think"), ("focus", "dismissal")):
            frac, n, p = sign_test([(med[(x, c, a)], med[(x, c, b)]) for (x, c) in keys])
            lines.append(f"  {'':8s}{a:13s} ranks the target above {b:13s} in {frac:4.0%} of pairs (n={n}, p={p:.1e})")
    return "\n".join(lines)


# =============================================================================== arithmetic
ARITH_SHOTS = ["1 + 1 = 2", "6 - 2 = 4", "2 * 2 = 4", "10 / 2 = 5", "7 + 1 = 8", "8 - 5 = 3", "3 * 2 = 6", "6 / 3 = 2"]
ARITH_FRAMES = {
    "bare": lambda e: f"{e} =",
    "worked": lambda e: "\n".join(ARITH_SHOTS) + f"\n{e} =",
    "qa": lambda e: "".join("Q: What is {}?\nA: {}\n".format(*s.split(" = ")) for s in ARITH_SHOTS[:4]) + f"Q: What is {e}?\nA:",
}


@torch.no_grad()
def arithmetic(A):
    """Capability check for the math family, GPT-2: can it do the paper's 24 problems as plain
    text, with nothing before the problem, after eight worked examples, or as questions and
    answers?  The rank of the answer (digit or number word) as the next token, and the top token."""
    assert A.name == "gpt2" or isinstance(A, Gemma), "plain text: GPT-2 or Gemma (Qwen's check is in `clauses`)"
    words = {"2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}
    rows = []
    for frame, f in ARITH_FRAMES.items():
        for m in DM["math_problems"]:
            lg = A.logits(A.ids(f(m["expr"])))[-1]
            t = sorted(set(forms(A, m["answer"])) | set(forms(A, words[m["answer"]])))
            rows.append(dict(frame=frame, expr=m["expr"], answer=m["answer"], rank=int(rank_rows(lg[None], t)[0]),
                             top=A.dec(int(lg.argmax()))))
    _save(A, "arithmetic", dict(rows=rows))
    print("== arithmetic (GPT-2)")
    for frame in ARITH_FRAMES:
        s = [r for r in rows if r["frame"] == frame]
        print(f"  {frame:7s} answer top-1 on {sum(r['rank'] == 1 for r in s)}/{len(s)}   its top tokens: "
              + " ".join(sorted({r['top'].strip() for r in s})))


@torch.no_grad()
def arithmetic_gen(A):
    """The same check, read from what the model writes: its greedy continuation (12 tokens) in each
    plain-text frame and, for Gemma, as a chat (`What is {expr}? Answer with just the number.`).
    The answer is the last number (digits or a number word) on the first non-empty line.  Needed
    for Gemma, which writes " 8" as " " then "8", and in a chat often restates the expression
    first, so the next-token rank in `arithmetic` and `clauses` misses answers it gets right."""
    import re
    words = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
             "eight": "8", "nine": "9", "ten": "10"}
    frames = dict(ARITH_FRAMES)
    if isinstance(A, Gemma):
        frames["chat"] = lambda e: A.chat([("user", f"What is {e}? Answer with just the number.")])
    rows = []
    for frame, f in frames.items():
        for m in DM["math_problems"]:
            ids = A.ids(f(m["expr"]))
            hf = A.lm.m._hf_model if A.name == "gpt2" else A.m._hf_model
            out = hf.generate(ids, max_new_tokens=12, do_sample=False)
            reply = A.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            line = next((ln for ln in reply.split("\n") if ln.strip()), "")
            nums = [words.get(n.lower(), n) for n in re.findall(r"\d+|" + "|".join(words), line, flags=re.I)]
            rows.append(dict(frame=frame, expr=m["expr"], answer=m["answer"], reply=reply,
                             said=nums[-1] if nums else None, right=bool(nums) and nums[-1] == m["answer"]))
    _save(A, "arithmetic_gen", dict(rows=rows))
    print(f"== arithmetic_gen {A.name}")
    for frame in frames:
        s = [r for r in rows if r["frame"] == frame]
        print(f"  {frame:7s} right on {sum(r['right'] for r in s)}/{len(s)}   e.g. {s[0]['expr']!r} -> {s[0]['reply']!r}")


# =============================================================================== paired questions
@torch.no_grad()
def questions(A):
    """Capability check for the paper's paired-question test (§3.2, ref/.../top-down-summoning.json):
    can the model answer the property question ("When are the events in this passage set ...?")
    at all?  GPT-2: `{q2}\n{stimulus}\nAnswer:`; the rank of the best expected label (e.g. "past")
    and of the best foil (e.g. "present") as the next token.  The test itself needs the model to be
    answering the question; if it can't, the label never entering its J-space says little."""
    TD = json.load(open(jl.REF_DATA / "top-down-summoning.json"))
    rows = []
    for it in TD["items"]:
        text = (f"{it['q2']}\n{it['stimulus']}\nAnswer:" if A.name == "gpt2"
                else A.chat([("user", f"{it['q2']}\n\n{it['stimulus']}")]))
        lg = A.logits(A.ids(text))[-1]

        def best(words):
            return int(rank_rows(lg[None], sorted({i for w in words for i in forms(A, w)}))[0])
        rows.append(dict(key=it["key"], expected=best(it["expected"]), foil=best(it["foil"]),
                         top3=[A.dec(t) for t in lg.topk(3).indices.tolist()]))
    _save(A, "questions", dict(rows=rows))
    print("== questions (rank of the best expected label / best foil as the next token)")
    for r in rows:
        print(f"  {r['key']:10s} expected {r['expected']:6d}   foil {r['foil']:6d}   top-3 {r['top3']}")


@torch.no_grad()
def clauses(A):
    """Directed modulation, parts (b) and (c) of the definition, on the instruction-tuned model."""
    math = []
    words = {"2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}
    for m in DM["math_problems"]:
        lg = A.logits(A.ids(A.chat([("user", f"What is {m['expr']}? Answer with just the number.")])))[-1]
        t = sorted(set(forms(A, m["answer"])) | set(forms(A, words.get(m["answer"], m["answer"]))))
        math.append(dict(expr=m["expr"], answer=m["answer"], rank=int(rank_rows(lg[None], t)[0]),
                         top=A.dec(int(lg.argmax()))))
    TD = json.load(open(jl.REF_DATA / "top-down-summoning.json"))
    paired = []
    for it in TD["items"]:
        exp = sorted({i for w in it["expected"] for i in forms(A, w)})
        foil = sorted({i for w in it["foil"] for i in forms(A, w)})
        q1e = sorted({i for w in it["q1_expect"] for i in forms(A, w)})
        rec = dict(key=it["key"])
        for tag, q in (("q1", TD["q1"]), ("q2", it["q2"])):
            text = A.chat([("user", f"{q}\n\n{it['stimulus']}")])
            ids = A.ids(text)
            lg = A.logits(ids)[-1]
            span = A.span(text, it["stimulus"])
            res = A.residuals(ids, A.band)
            pr = torch.stack([rank_rows(A.lens_logits(res[L][span], L), exp) for L in A.band]).min(0).values
            rec[tag] = dict(n=len(span), hit10=int((pr <= 10).sum()), best=int(pr.min()),
                            out_top3=[A.dec(t) for t in lg.topk(3).indices.tolist()],
                            exp_rank=int(rank_rows(lg[None], exp)[0]),
                            foil_rank=int(rank_rows(lg[None], foil)[0]) if foil else None,
                            q1e_rank=int(rank_rows(lg[None], q1e)[0]) if q1e else None)
        paired.append(rec)
    _save(A, "dm_clauses", dict(band=A.band, math=math, paired=paired))
    print(f"== clauses\n  math: answer top-1 on {sum(r['rank'] == 1 for r in math)}/{len(math)}")
    for r in paired:
        a, b = r["q2"], r["q1"]
        print(f"  {r['key']:10s} q2 answers {a['out_top3']} (label rank {a['exp_rank']}, foil {a['foil_rank']}); "
              f"lens top-10 at {a['hit10']}/{a['n']} (best {a['best']}) | q1 lens top-10 at {b['hit10']}/{b['n']} (best {b['best']})")


# =============================================================================== verbal report + stats
@torch.no_grad()
def report(A):
    D = json.load(open(jl.REF_DATA / "verbal-report.json"))["candidates"]
    rows = []
    for cat, cands in D.items():
        ids = A.ids(A.chat([("user", f"Think of a {cat}. Answer in one word.")]))
        lg = A.logits(ids)[-1]
        mem = {A.tid(w): w for w in cands if A.is_single(w)}
        greedy = int(lg.argmax())
        if greedy not in mem:
            rows.append(dict(cat=cat, gate=False, greedy=A.dec(greedy)))
            continue
        clean = A.residuals(ids, A.band)
        for w in [w for w in cands if A.is_single(w)][:10]:
            t = A.tid(w)
            before = int(jl.ranks_of(lg, [t])[0])
            if t == greedy or before < 11:
                continue
            edits = []
            for L in A.band:
                vs, vt = jl.unit(A.v(L, greedy)), jl.unit(A.v(L, t))
                delta = (clean[L] @ vs)[:, None] * (vt - vs)[None, :]
                edits.append(Edit(L, lambda h, pos, delta=delta: h + delta[pos]))
            after = int(jl.ranks_of(A.logits(ids, edits)[-1], [t])[0])
            rows.append(dict(cat=cat, gate=True, source=A.dec(greedy), target=w, before=before, after=after))
    sw = [r for r in rows if r.get("gate") and "after" in r]
    ncat = len({r["cat"] for r in rows if r["gate"]})
    line = (f"  answers with a candidate in {ncat}/{len(D)} categories; swap target top-1 "
            f"{sum(r['after'] == 1 for r in sw)}/{len(sw)}, top-5 {sum(r['after'] <= 5 for r in sw)}/{len(sw)}")
    _save(A, "report", dict(band=A.band, rows=rows, summary=line))
    print("== report\n" + line)


@torch.no_grad()
def stats(A):
    """Lens-vs-output top-1 agreement and the paper's persistence (Fig 28c: the log-probability the
    lens gives, Δ positions later, to its top token, minus the same for the top token of a random
    position), on 16 WikiText-103 validation passages of 128 tokens."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    texts = [t for t in ds["text"] if len(t) > 600][:16]
    DS = [1, 2, 4, 8, 16, 32]
    layers = sorted(A.J)
    acc = {L: [0, 0] for L in layers}
    per = {L: {d: [0.0, 0.0, 0, 0] for d in DS} for L in layers}
    g = torch.Generator().manual_seed(0)
    for t in texts:
        ids = A.ids(t)[:, :128]
        res = A.residuals(ids, layers + [A.n_layers - 1])
        out1 = A.m.unembed(res[A.n_layers - 1]).argmax(-1)
        for L in layers:
            lg = A.lens_logits(res[L], L)
            t1 = lg.argmax(-1)
            acc[L][0] += int((t1[16:-1] == out1[16:-1]).sum())
            acc[L][1] += len(t1[16:-1])
            a, lp = t1[16:], torch.log_softmax(lg[16:], -1)
            for d in DS:
                real = lp[d:].gather(1, a[:-d][:, None])[:, 0]
                null_t = a[torch.randint(len(a), (len(a) - d,), generator=g).to(a.device)]
                null = lp[d:].gather(1, null_t[:, None])[:, 0]
                s = per[L][d]
                s[0] += float(real.sum())
                s[1] += float(null.sum())
                s[2] += real.numel()
                s[3] += null.numel()
    out = [dict(layer=L, top1_agree=acc[L][0] / acc[L][1],
                persistence={d: per[L][d][0] / per[L][d][2] - per[L][d][1] / per[L][d][3] for d in DS}) for L in layers]
    _save(A, "stats", dict(band=A.band, rows=out))
    print("== stats (band %s)" % A.band)
    for r in out:
        print(f"  L{r['layer']:2d}  lens top-1 = output top-1 {r['top1_agree']:.3f}   persistence Δ=1..32: "
              + " ".join(f"{v:5.2f}" for v in r["persistence"].values()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen", choices=["qwen", "gpt2", *GEMMA])
    ap.add_argument("--frames", default=None, help="comma-separated modulation frames")
    ap.add_argument("--tokforms", default="surface,space", help="injected token forms: surface, space")
    ap.add_argument("--prefill", default="default", choices=sorted(PREFILLS), help="introspect: which released prefill")
    ap.add_argument("--strengths", default="paper", choices=["paper", "fine"], help="introspect: strength grid")
    ap.add_argument("--vectors", default="raw,centered", help="introspect: raw, centered")
    ap.add_argument("--intro-frames", default=None, help="introspect: comma-separated frames (default: all)")
    ap.add_argument("experiments", nargs="*", default=["introspect", "modulation"])
    args = ap.parse_args()
    A = load(args.model)
    print(f"model {getattr(A, 'model_id', 'gpt2')}  band {A.band}  device {A.device}", flush=True)
    for e in args.experiments:
        if e.startswith("modulation"):
            {"modulation": modulation, "modulation_grid": modulation_grid, "modulation_readout": modulation_readout,
             "modulation_copy": modulation_copy}[e](A, args.frames.split(",") if args.frames else None)
        else:
            {"introspect": lambda A: introspect(A, tuple(args.tokforms.split(",")), args.prefill,
                                                STRENGTHS if args.strengths == "paper" else FINE_STRENGTHS,
                                                tuple(args.vectors.split(",")),
                                                args.intro_frames.split(",") if args.intro_frames else None),
             "introspect_privilege": introspect_privilege,
             "report": report, "stats": stats, "questions": questions, "clauses": clauses,
             "arithmetic": arithmetic, "arithmetic_gen": arithmetic_gen}[e](A)
