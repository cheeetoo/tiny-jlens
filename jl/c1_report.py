"""Criterion 1 (verbal report) in gpt2-small — the paper's §3.1, in one run.  See PROTOCOL.md.

  gate   clean pass per category: does the model answer with a candidate?
  1a     Spearman(lens logits, output logits) over the 10 candidates at the ':', per layer
  1b     swap source -> target at every position over the band; target's output rank before/after
  1d/1e  concept vectors split into J-space part (k=16 pursuit) and remainder; swap along each,
         same magnitude; remainder also with the J coordinates clamped to clean

Writes results/c1_report/{results.json, prompts.json, summary.txt}.
Run:  python -m jl.c1_report
"""
from __future__ import annotations

import json
import random

import torch
from scipy.stats import spearmanr

import jl
from jl.stats import wilson

K_PURSUIT = 16
N_BASELINE = 100

# =============================================================================== prompt material
# From the paper's released data (ref/jacobian-lens/data/experiments/verbal-report.json:
# 14 categories x 14 candidate words).
#
# Base-model frame (deviation; see PROTOCOL.md): the request is the last line of a 9-line list
# whose first 8 lines are the same request for other categories with an answer filled in.  Demo
# categories are the 8 following the queried one in the paper's order; a demo answer is that
# category's first single-token candidate that is not also a candidate of the queried category.
# The lens is read, and the swap graded, at the final ':'.

DATA = jl.REF_DATA / "verbal-report.json"
CANDIDATES: dict[str, list[str]] = json.load(open(DATA))["candidates"]
CATEGORIES: list[str] = list(CANDIDATES)
LINE = "Name a {cat}:"
K_SHOT = 8
CONCEPT_PROMPT = "Tell me about {concept}."  # §3.1 concept vectors


def members(lm, cat: str) -> list[str]:
    """The paper's candidates that are single GPT-2 tokens as ' Word', in the paper's order."""
    return [w for w in CANDIDATES[cat] if lm.is_single(" " + w)]


def ten(lm, cat: str) -> list[str]:
    """The paper's '10 candidate answers'."""
    return members(lm, cat)[:10]


def prompt(lm, cat: str) -> str:
    i = CATEGORIES.index(cat)
    demos = []
    for j in range(1, len(CATEGORIES)):
        c = CATEGORIES[(i + j) % len(CATEGORIES)]
        demos.append((c, next(w for w in members(lm, c) if w not in CANDIDATES[cat])))
        if len(demos) == K_SHOT:
            break
    return "".join(f"{LINE.format(cat=c)} {a}\n" for c, a in demos) + LINE.format(cat=cat)


# =============================================================================== the run
@torch.no_grad()
def main():
    lm = jl.Lensed()
    band = jl.BAND
    out = dict(band=band, gate=[], corr=[], swap=[], privilege=[], variance_fraction=[])

    # ------------------------------------------------------------------ gate + 1a
    cats = []
    for cat in CATEGORIES:
        text = prompt(lm, cat)
        ids = lm.encode(text)
        res = lm.residuals(ids)
        lg = lm.logits(ids)[-1]
        mem_ids = [lm.tid(" " + w) for w in members(lm, cat)]
        cand = ten(lm, cat)
        cand_ids = [lm.tid(" " + w) for w in cand]
        greedy = int(lg.argmax())
        gate = greedy in mem_ids
        source = greedy if gate else mem_ids[int(jl.ranks_of(lg, mem_ids).argmin())]
        cats.append(dict(cat=cat, text=text, ids=ids, clean=res, lg=lg, gate=gate, source=source,
                         targets=[(w, t) for w, t in zip(cand, cand_ids) if t != source]))
        out["gate"].append(dict(cat=cat, gate=gate, greedy=lm.dec(greedy), source=lm.dec(source),
                                top5=[lm.dec(t) for t in lg.topk(5).indices.tolist()],
                                candidates=cand, candidate_ranks=jl.ranks_of(lg, cand_ids).tolist()))
        for L in lm.layers:
            lens = lm.lens_logits(res[L][-1], L)[cand_ids]
            out["corr"].append(dict(cat=cat, gate=gate, layer=L,
                                    rho=float(spearmanr(lens.cpu(), lg[cand_ids].cpu()).statistic)))

    # ------------------------------------------------------------------ 1b swap
    for c in cats:
        for w, t in c["targets"]:
            lg = lm.logits(c["ids"], jl.swap_edits(lm, c["ids"], c["source"], t, band, clean=c["clean"]))[-1]
            out["swap"].append(dict(cat=c["cat"], gate=c["gate"], source=lm.dec(c["source"]), target=w,
                                    before=int(jl.ranks_of(c["lg"], [t])[0]), after=int(jl.ranks_of(lg, [t])[0]),
                                    top1_after=lm.dec(int(lg.argmax()))))

    # ------------------------------------------------------------------ 1d/1e privileging
    words = sorted({w for cat in CATEGORIES for w in members(lm, cat)})
    raw = {w: torch.stack([lm.residuals(lm.encode(CONCEPT_PROMPT.format(concept=w)))[L][-1] for L in lm.layers]) for w in words}
    rng = random.Random(0)
    parts = {}
    for w in words:
        base = rng.sample([x for x in words if x != w], N_BASELINE)
        u = raw[w] - torch.stack([raw[b] for b in base]).mean(0)
        parts[lm.tid(" " + w)] = {}
        for i, L in enumerate(lm.layers):  # every lens layer (the clamp in 1e needs all of them)
            support, recon = jl.pursuit(u[i], lm.V(L), K_PURSUIT)
            parts[lm.tid(" " + w)][L] = dict(j=recon, n=u[i] - recon, support=support)
            out["variance_fraction"].append(dict(concept=w, layer=L, frac=float(recon.norm() ** 2 / u[i].norm() ** 2)))
    for c in cats:
        s = c["source"]
        for w, t in c["targets"]:
            conds = {
                "lens": jl.swap_edits(lm, c["ids"], s, t, band, clean=c["clean"]),
                "jpart": jl.swap_edits(lm, c["ids"], s, t, band, clean=c["clean"], comp={L: (parts[s][L]["j"], parts[t][L]["j"]) for L in band}),
                "nonj": jl.swap_edits(lm, c["ids"], s, t, band, clean=c["clean"], comp={L: (parts[s][L]["n"], parts[t][L]["n"]) for L in band}),
            }
            relevant = {L: torch.stack([lm.v(L, i) for i in sorted({s, t} | set(parts[s][L]["support"]) | set(parts[t][L]["support"]))]) for L in lm.layers}
            conds["nonj_clamp"] = conds["nonj"] + jl.clamp_edits(lm, c["ids"], relevant, clean=c["clean"])
            for name, edits in conds.items():
                lg = lm.logits(c["ids"], edits)[-1]
                out["privilege"].append(dict(cat=c["cat"], gate=c["gate"], target=w, cond=name,
                                             before=int(jl.ranks_of(c["lg"], [t])[0]), after=int(jl.ranks_of(lg, [t])[0])))

    # ------------------------------------------------------------------ write
    R = jl.results_dir("c1_report")
    json.dump(out, open(R / "results.json", "w"), indent=1)
    json.dump([dict(cat=c["cat"], prompt=c["text"]) for c in cats], open(R / "prompts.json", "w"), indent=1)
    summary = summarize(out)
    open(R / "summary.txt", "w").write(summary)
    print(summary)


def summarize(out) -> str:
    lines = ["gate (does the model answer with a candidate?)"]
    for g in out["gate"]:
        lines.append(f"  {g['cat']:11s} {'pass' if g['gate'] else 'FAIL':4s}  greedy {g['greedy']!r:12s} source {g['source']!r:12s} top-5 {g['top5']}")
    lines.append(f"  {sum(g['gate'] for g in out['gate'])}/{len(out['gate'])} categories pass\n")
    lines.append("1a  Spearman(lens, output) over the 10 candidates at ':', mean over categories (all | gate-passed)")
    for L in sorted({x["layer"] for x in out["corr"]}):
        a = [x["rho"] for x in out["corr"] if x["layer"] == L]
        g = [x["rho"] for x in out["corr"] if x["layer"] == L and x["gate"]]
        lines.append(f"  L{L:<2d} {sum(a)/len(a):+.3f} | {sum(g)/len(g):+.3f}" + ("   <- band" if L in out["band"] else ""))
    for title, key, conds in (("1b  swap: target reaches top-5 (top-1); targets starting at output rank >= 11", "swap", [None]),
                              ("1d/1e  same, swapping along each component at the same magnitude", "privilege", ["lens", "jpart", "nonj", "nonj_clamp"])):
        lines.append("\n" + title)
        for label, flt in (("gate-passed", lambda x: x["gate"]), ("all categories", lambda x: True)):
            for cond in conds:
                sub = [x for x in out[key] if flt(x) and x["before"] >= 11 and (cond is None or x["cond"] == cond)]
                n = len(sub)
                p, lo, hi = wilson(sum(x["after"] <= 5 for x in sub), n)
                lines.append(f"  {label:14s} {cond or '':11s} n={n:3d}  {p:5.1%} [{lo:4.0%}, {hi:4.0%}]  ({sum(x['after'] == 1 for x in sub)/n:4.0%})  median rank {sorted(x['before'] for x in sub)[n//2]} -> {sorted(x['after'] for x in sub)[n//2]}")
    vf = out["variance_fraction"]
    lines.append("\nJ-space component's share of concept-vector variance (median): " +
                 ", ".join(f"L{L} {sorted(x['frac'] for x in vf if x['layer'] == L)[len([x for x in vf if x['layer'] == L])//2]:.1%}" for L in out["band"]))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
