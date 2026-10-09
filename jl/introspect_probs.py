"""The injected-thought test (paper §3.1, Fig 7) in probabilities rather than top-10 ranks.

On the prefilled transcript (as in `followups introspect`), with a
concept injected over the question, record the concept's probability as the next token (both
surface forms summed), whether it is the top-1 prediction, and the entropy, at every position of
the model's line.  This checks that "in the top 10 earlier in the line" means the model would
actually say the word there, not that it sits in a low-entropy tail.

Run:  python -m jl.introspect_probs
Writes results/introspect_probs/gpt2.json.
"""
from __future__ import annotations

import json

import torch

import jl
from jl.hooks import Edit

def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class Adapter:
    """GPT-2 (jl.Lensed) with the interface the test uses."""

    def __init__(self, which="gpt2"):
        self.which = which
        self.m = jl.Lensed()
        self.band = jl.BAND
        self.strengths = (0.0, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0)
        self.tok = self.m.tok

    def ids(self, text):
        return torch.tensor([[self.m.bos] + self.tok(text, add_special_tokens=False).input_ids], device=self.m.device)

    def is_single(self, s):
        return self.m.is_single(s)

    def tid(self, s):
        return self.m.tid(s)

    def v(self, L, t):
        return self.m.v(L, t)

    def residuals(self, ids, layers):
        return self.m.residuals(ids, layers)

    def logits(self, ids, edits=None):
        return self.m.logits(ids, edits)

    # ---------------------------------------------------------------- prompts
    def report_prompt(self, D):
        """(text, question): the two-speaker transcript."""
        from jl.followups import INTRO_A, INTRO_Q
        return INTRO_Q + INTRO_A, INTRO_Q

    def span(self, text, sub):
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        a = text.index(sub)
        b = a + len(sub)
        off = 1                                          # the prepended <|endoftext|>
        return [i + off for i, (s, e) in enumerate(enc.offset_mapping) if s < b and e > a]


def edits_for(A, w, pos, s, norms):
    t = A.tid(" " + w)
    return [Edit(L, lambda h, p, v=unit(A.v(L, t)) * norms[L] * s: h + v, pos) for L in A.band]


@torch.no_grad()
def forced(A, D, concepts):
    text, q = A.report_prompt(D)
    ids = A.ids(text)
    qpos = A.span(text, q)
    clean = A.residuals(ids, A.band)
    norms = {L: float(clean[L][1:].norm(dim=-1).mean()) for L in A.band}
    first_a = qpos[-1] + 1
    toks = [A.tok.decode([int(x)]) for x in ids[0]]
    rows = []
    for s in A.strengths:
        for w in concepts:
            targets = [A.tid(w), A.tid(" " + w)]
            lp = A.logits(ids, edits_for(A, w, qpos, s, norms)).log_softmax(-1)
            p = lp[:, targets].exp().sum(-1)                 # [T] prob of the concept as next token
            ent = -(lp.exp() * lp).sum(-1)
            top1 = lp.argmax(-1)
            rows.append(dict(strength=s, word=w,
                             p=[float(x) for x in p[first_a:]],
                             top1=[int(int(t) in targets) for t in top1[first_a:]],
                             entropy=[float(x) for x in ent[first_a:]]))
        rr = [r for r in rows if r["strength"] == s]
        rep = [max(r["p"][-2:]) for r in rr]
        oth = [max(r["p"][:-2]) for r in rr]
        print(f"  forced  s={s:<4}: report p median {sorted(rep)[len(rep)//2]:.3f}  >0.1: {sum(x > .1 for x in rep):2d}/{len(rr)} | "
              f"earlier max p median {sorted(oth)[len(oth)//2]:.3f}  >0.1: {sum(x > .1 for x in oth):2d}  >0.5: {sum(x > .5 for x in oth):2d}  "
              f"top-1 somewhere earlier: {sum(any(r['top1'][:-2]) for r in rr):2d}", flush=True)
    return dict(tokens=toks[first_a:], rows=rows, norms=norms)


def main(which):
    A = Adapter(which)
    D = json.load(open(jl.REF_DATA / "verbal-introspection.json"))
    concepts = [c["surface"] for c in D["concepts"] if A.is_single(" " + c["surface"]) and A.is_single(c["surface"])]
    print(f"===== {which} (n={len(concepts)})", flush=True)
    res = dict(model=which, band=A.band, concepts=concepts, forced=forced(A, D, concepts))
    json.dump(res, open(jl.results_dir("introspect_probs") / f"{which}.json", "w"))


if __name__ == "__main__":
    main("gpt2")
