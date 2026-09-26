"""The tests GPT-2 small fails, run on a small instruction-tuned model: Qwen3-1.7B with the
Neuronpedia J-lens (qwen3-1.7b/jlens/Salesforce-wikitext).  Chat format, thinking disabled.

  stats       layer statistics (lens-vs-output top-1 agreement, top-1 persistence), to pick a band
  report      verbal report swap: "Think of a {category}. Answer in one word." (paper §3.1)
  modulation  directed modulation: instruction about X in the user turn, the assistant copies an
              unrelated sentence; X's J-lens rank over the copy, by condition (paper §3.2, App Fig 65)
  introspect  injected thought: the paper's prompt and prefill; a concept's J-lens vector added over
              the user's question; the concept's output rank at the report vs. elsewhere in the
              assistant's line (paper §3.1, Fig 7)

Writes results/qwen_control/{name}.json.  Run:  python -m jl.qwen_control [stats|report|modulation|introspect]
DEVICE defaults to mps when available.
"""
from __future__ import annotations

import json
import os
import random
import sys

import torch
import transformers
from huggingface_hub import hf_hub_download

import jl
import jlens
from jlens.hooks import ActivationRecorder
from jl.hooks import Edit, Session
from jl.stats import median, sign_test, wilson

MODEL_ID = "Qwen/Qwen3-1.7B"
LENS_FILE = "qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt"
BAND = [int(x) for x in os.environ.get("QBAND", "12,14,16,18,20").split(",")]


class QLensed:
    def __init__(self):
        self.device = os.environ.get("DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
        hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(self.device).eval()
        self.m = jlens.from_hf(hf, self.tok)
        self.n_layers, self.d = self.m.n_layers, self.m.d_model
        lens = jlens.JacobianLens.load(hf_hub_download("neuronpedia/jacobian-lens", LENS_FILE))
        self.layers = list(lens.source_layers)
        self.J = {L: lens.jacobians[L].float().to(self.device) for L in self.layers}
        self.U = self.m._lm_head.weight.detach().float()
        self.vocab = self.U.shape[0]
        self.ubar = self.U.mean(0)

    def chat(self, turns, prefill=""):
        """ids [1,T] for a chat (list of (role, text)) ending in an assistant turn with `prefill`."""
        msgs = [{"role": r, "content": t} for r, t in turns]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return text + prefill

    def ids(self, text):
        return torch.tensor([self.tok(text, add_special_tokens=False).input_ids], device=self.device)

    def span(self, text, sub, start_at=0):
        """Token positions covering the character span of `sub` (its first occurrence after start_at)."""
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        a = text.index(sub, start_at)
        b = a + len(sub)
        return [i for i, (s, e) in enumerate(enc.offset_mapping) if s < b and e > a]

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

    def v(self, L, t):
        """Centered J-lens vector of token t at layer L, (u_t - mean_u) J_L, without building the
        full [vocab, d] dictionary (151,936 x 2,048 per layer)."""
        return (self.U[t] - self.ubar) @ self.J[L]


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _save(name, obj):
    json.dump(obj, open(jl.results_dir("qwen_control") / f"{name}.json", "w"), indent=1, default=float)


# =============================================================================== stats
@torch.no_grad()
def stats():
    from datasets import load_dataset
    q = QLensed()
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    texts = [t for t in ds["text"] if len(t) > 600][:16]
    layers = q.layers
    st = {L: dict(top1=0, n=0, auto=0, null=0, m=0) for L in layers}
    g = torch.Generator().manual_seed(0)
    for t in texts:
        ids = q.ids(t)[:, :128]
        res = q.residuals(ids, layers + [q.n_layers - 1])
        out1 = q.m.unembed(res[q.n_layers - 1]).argmax(-1)
        for L in layers:
            t1 = q.lens_logits(res[L], L).argmax(-1)
            st[L]["top1"] += int((t1[16:-1] == out1[16:-1]).sum())
            st[L]["n"] += len(t1[16:-1])
            a = t1[16:]
            st[L]["auto"] += int((a[:-1] == a[1:]).sum())
            perm = a[torch.randperm(len(a), generator=g)]
            st[L]["null"] += int((perm[:-1] == perm[1:]).sum())
            st[L]["m"] += len(a) - 1
    out = [dict(layer=L, top1=s["top1"] / s["n"], persistence=(s["auto"] - s["null"]) / s["m"]) for L, s in st.items()]
    for r in out:
        print(f"  L{r['layer']:2d}  lens top-1 = output top-1: {r['top1']:.3f}   persistence above shuffled: {r['persistence']:+.3f}")
    _save("stats", out)


# =============================================================================== report
@torch.no_grad()
def report():
    q = QLensed()
    D = json.load(open(jl.REF_DATA / "verbal-report.json"))["candidates"]
    rows = []
    for cat, cands in D.items():
        text = q.chat([("user", f"Think of a {cat}. Answer in one word.")])
        ids = q.ids(text)
        lg = q.logits(ids)[-1]
        forms = {w: w for w in cands if q.is_single(w)}
        if not forms:
            continue
        mem = {q.tid(w): w for w in forms}
        greedy = int(lg.argmax())
        if greedy not in mem:
            rows.append(dict(cat=cat, gate=False, greedy=q.dec(greedy)))
            continue
        clean = q.residuals(ids, BAND)
        for w in list(forms)[:10]:
            t = q.tid(w)
            before = int(jl.ranks_of(lg, [t])[0])
            if t == greedy or before < 11:
                continue
            edits = []
            for L in BAND:
                vs, vt = unit(q.v(L, greedy)), unit(q.v(L, t))
                mag = clean[L] @ vs
                delta = mag[:, None] * (vt - vs)[None, :]
                edits.append(Edit(L, lambda h, pos, delta=delta: h + delta[pos]))
            after = int(jl.ranks_of(q.logits(ids, edits)[-1], [t])[0])
            rows.append(dict(cat=cat, gate=True, source=q.dec(greedy), target=w, before=before, after=after))
    sw = [r for r in rows if r.get("gate") and "after" in r]
    p, lo, hi = wilson(sum(r["after"] == 1 for r in sw), len(sw))
    print(f"  gate: {len({r['cat'] for r in rows if r['gate']})}/{len(D)} categories; swap top-1 {p:.0%} [{lo:.0%},{hi:.0%}] n={len(sw)}; "
          f"top-5 {sum(r['after'] <= 5 for r in sw)/len(sw):.0%}")
    _save("report", rows)


# =============================================================================== modulation
N_WORDS, N_CARRIERS, N_PHRASINGS = 20, 6, 3


@torch.no_grad()
def modulation():
    q = QLensed()
    DM = json.load(open(jl.REF_DATA / "directed-modulation.json"))
    groups = {"mention": ["mention"], "focus": ["focus"], "dismissal": ["dismissal"], "negated": ["negated-think"]}
    phr = {c: [p["text"] for p in DM["phrasings"] if p["group"] in g][:N_PHRASINGS] for c, g in groups.items()}
    words = []
    for c in DM["topic_categories"]:
        for w in c["members"]:
            if w.isalpha() and w.islower() and q.is_single(" " + w) and w not in words:
                words.append(w)
    rng = random.Random(0)
    words = rng.sample(words, N_WORDS)
    carriers = DM["carrier_sentences"][:N_CARRIERS]
    rows = []
    for w in words:
        tids = [q.tid(" " + w)] + ([q.tid(" " + w.capitalize())] if q.is_single(" " + w.capitalize()) else [])
        for ci, car in enumerate(carriers):
            for cond in ["baseline"] + list(groups):
                for pi, instr in enumerate([None] if cond == "baseline" else phr[cond]):
                    user = f'Write the following sentence: "{car}"' + (f" {instr.format(x=w)}" if instr else "")
                    text = q.chat([("user", user)], prefill=car)
                    ids = q.ids(text)
                    pos = q.span(text, car, start_at=text.index("<|im_start|>assistant"))
                    res = q.residuals(ids, BAND)
                    best = 10 ** 9
                    for L in BAND:
                        lg = q.lens_logits(res[L][pos], L)                 # [P, vocab]
                        for t in tids:
                            best = min(best, int(((lg > lg[:, t:t + 1]).sum(1) + 1).min()))
                    rows.append(dict(word=w, carrier=ci, cond=cond, phrasing=pi, best=best))
        print(".", end="", flush=True)
    print()
    per = {}
    for r in rows:
        per.setdefault((r["word"], r["carrier"], r["cond"]), []).append(r["best"])
    med = {k: median(v) for k, v in per.items()}
    keys = {(w, c) for (w, c, _) in med}
    summary = {}
    for a, b in (("mention", "baseline"), ("focus", "mention"), ("dismissal", "mention"), ("negated", "mention"), ("dismissal", "focus")):
        frac, n, pval = sign_test([(med[(w, c, a)], med[(w, c, b)]) for (w, c) in keys])
        summary[f"{a}<{b}"] = dict(frac=frac, n=n, p=pval)
        print(f"  {a:9s} higher in the lens than {b:9s}: {frac:.0%} of pairs (n={n}, p={pval:.1e})")
    for cond in ["baseline"] + list(groups):
        v = [med[(w, c, cond)] for (w, c) in keys]
        print(f"  {cond:9s} median best band rank {median(v):6d}   top-25: {sum(x <= 25 for x in v)/len(v):.0%}")
    _save("modulation", dict(rows=rows, summary=summary, band=BAND))


# =============================================================================== introspect
STRENGTHS = (0.0, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0)


@torch.no_grad()
def introspect():
    q = QLensed()
    D = json.load(open(jl.REF_DATA / "verbal-introspection.json"))
    turns = [(t["role"], t["content"].strip()) for t in D["intro_prompt"] if t["content"].strip()]
    prefill = D["prefills"]["default"].lstrip()
    text = q.chat(turns, prefill=prefill)
    ids = q.ids(text)
    last_user = turns[-1][1]
    q_pos = q.span(text, last_user)
    a_start = text.rindex("<|im_start|>assistant")
    a_pos = [p for p in q.span(text, prefill, start_at=a_start)][:-2]      # before 'about "'
    concepts = [c["surface"] for c in D["concepts"] if q.is_single(" " + c["surface"]) and q.is_single(c["surface"])]
    clean = q.residuals(ids, BAND)
    norms = {L: float(clean[L][1:].norm(dim=-1).mean()) for L in BAND}
    rows = []
    for s in STRENGTHS:
        for w in concepts:
            targets = [q.tid(w), q.tid(" " + w)]
            edits = [Edit(L, lambda h, pos, v=unit(q.v(L, q.tid(" " + w))) * norms[L] * s: h + v, q_pos) for L in BAND]
            lg = q.logits(ids, edits)
            rep = int(min(jl.ranks_of(lg[-1], targets).min(), jl.ranks_of(lg[-2], targets).min()))
            other = min(int(jl.ranks_of(lg[p], targets).min()) for p in a_pos)
            rows.append(dict(strength=s, word=w, report=rep, other=other))
        rr = [r for r in rows if r["strength"] == s]
        print(f"  strength {s:4.2f}: report median rank {median([r['report'] for r in rr]):6d}  top-1 {sum(r['report'] == 1 for r in rr)}/{len(rr)}  "
              f"top-10 {sum(r['report'] <= 10 for r in rr)}/{len(rr)}  | elsewhere in the assistant line: top-10 {sum(r['other'] <= 10 for r in rr)}/{len(rr)}", flush=True)
    _save("introspect", dict(rows=rows, band=BAND, n_concepts=len(concepts), a_tokens=[q.dec(int(ids[0, p])) for p in a_pos],
                             q_tokens=len(q_pos)))


if __name__ == "__main__":
    for name in sys.argv[1:] or ["stats", "report", "modulation", "introspect"]:
        print(f"===== {name}  band {BAND}", flush=True)
        {"stats": stats, "report": report, "modulation": modulation, "introspect": introspect}[name]()
