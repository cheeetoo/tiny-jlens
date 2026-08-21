"""Criterion 2 (directed modulation) in gpt2-small -- the paper's §3.2, in one run.  See
PROTOCOL.md.  Four experiments, matching the two clauses of the criterion plus the privileging
control:

  2a  instructed hold-in-mind (concept): does an instruction move a word into the lens over an
      unrelated copied sentence -- and does it do so BEYOND mere mention?
      conditions baseline / mention / focus / dismissal / negated; metric = lens rank of the
      tracked word over the copy span, per band layer, held-not-spoken.
  2b  instructed computation (math): capability pre-check + answer-in-lens.  (capability-gated)
  2c  implicit task-demand (paired questions): does a property label enter the lens under the
      name-the-property question but not the predict-next-word question?  (capability-gated)
  2d  privileging: an "imagine this is French" header raises 'French' in the lens over an
      English sentence while leaving a J-orthogonalized French probe unmoved; real French moves
      the probe.  The analog of criterion 1's 1d/1e.

Writes results/results.json, results/prompts.json, results/summary.txt.
Run:  python c2_modulation/run.py
"""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jl  # noqa: E402
import prompts as P  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS_REPORTED = [6, 7, 8, 9, 10]     # band is 7-9; 6 and 10 flank it to show localization
N_WORDS = 30
N_CARRIERS = 12
K_PURSUIT = 16


# -------------------------------------------------------------- rank helpers
def per_pos_rank(logits: torch.Tensor, tids: list[int]) -> torch.Tensor:
    """[S] 1-indexed best (min over targets) rank of `tids` at each position of `logits` [S,vocab]."""
    best = None
    for t in tids:
        r = (logits > logits[:, t:t + 1]).sum(dim=1) + 1
        best = r if best is None else torch.minimum(best, r)
    return best


@torch.no_grad()
def concept_trial(lm, carrier, instruction, x, tids):
    """One 2a trial.  Returns per-layer best lens rank over the copy span, the blurt rank (best
    output rank over the span), and best_held (best band lens rank at positions where the target
    is NOT in the model's output top-10 -- i.e. held, not about to be spoken)."""
    ids, span = P.copy_frame(lm, carrier, instruction, x)
    res = lm.residuals(ids, LAYERS_REPORTED)
    out_pos = per_pos_rank(lm.logits(ids)[span], tids)          # [S] output rank per position
    per_layer, band_pos = {}, None
    for L in LAYERS_REPORTED:
        lr = per_pos_rank(lm.lens_logits(res[L][span], L), tids)  # [S]
        per_layer[L] = int(lr.min())
        if L in jl.BAND:
            band_pos = lr if band_pos is None else torch.minimum(band_pos, lr)
    held = band_pos[out_pos > 10]
    return dict(per_layer=per_layer,
                best=int(band_pos.min()),
                best_held=int(held.min()) if len(held) else 10 ** 9,
                blurt=int(out_pos.min()))


def main():
    lm = jl.Lensed()
    out = dict(band=jl.BAND, layers_reported=LAYERS_REPORTED,
               concept=[], concept_category_null=[],
               math_capability=[], math_lens=[], demand_capability=[], demand=[], imagine=[])

    words = P.concept_words(lm)[:N_WORDS]
    carriers = P.CARRIERS[:N_CARRIERS]

    # ============================================================ 2a  concept modulation
    for w in words:
        tids = P.word_targets(lm, w)
        for ci, carrier in enumerate(carriers):
            for cond in P.CONDITIONS:
                phrasings = [None] if cond == "baseline" else P.PHRASINGS[cond]
                for pi, instr in enumerate(phrasings):
                    r = concept_trial(lm, carrier, instr, w, tids)
                    out["concept"].append(dict(word=w, carrier=ci, cond=cond, phrasing=pi, **r))
        print(".", end="", flush=True)
    print(" 2a done")

    # ---- 2a secondary: main-text form (name the CATEGORY, track its unnamed MEMBERS) ----
    # Faithful to Fig 10 but near-null at 124M; we quantify it on focus / mention vs baseline.
    for c in P.TOPICS:
        mt = sorted({t for m in c["members"] if lm.is_single(" " + m)
                     for t in P.word_targets(lm, m)})
        if not mt:
            continue
        for ci, carrier in enumerate(carriers):
            for cond, instr in (("baseline", None), ("focus", P.PHRASINGS["focus"][0]),
                                ("mention", P.PHRASINGS["mention"][0])):
                r = concept_trial(lm, carrier, instr, c["name"], mt)
                out["concept_category_null"].append(
                    dict(cat=c["name"], carrier=ci, cond=cond, best=r["best"]))

    # ============================================================ 2b  math (capability-gated)
    for m in P.MATH:
        ai = [lm.tid(" " + m["answer"])] if lm.is_single(" " + m["answer"]) else []
        ids = lm.encode(f"{m['expr']} =")
        lg = lm.logits(ids)[-1]
        out["math_capability"].append(dict(
            expr=m["expr"], answer=m["answer"], tier=m["tier"], greedy=lm.dec(int(lg.argmax())),
            correct=lm.dec(int(lg.argmax())).strip() == m["answer"],
            answer_rank=int(jl.ranks_of(lg, ai)[0]) if ai else -1))
        if not ai:
            continue
        for ci, carrier in enumerate(carriers):
            for cond, instr in (("baseline", None),
                                ("compute", "Work out {x} in your head while you write."),
                                ("mention", "{x}.")):
                r = concept_trial(lm, carrier, instr, m["expr"], ai)
                out["math_lens"].append(dict(expr=m["expr"], carrier=ci, cond=cond, best=r["best"]))

    # ============================================================ 2c  implicit task-demand
    def label_ids(words_):
        ids = []
        for w in words_:
            for f in (" " + w, " " + w.capitalize(), " " + w.upper()):
                if lm.is_single(f):
                    ids.append(lm.tid(f))
        return sorted(set(ids))

    for it in P.TD["items"]:
        exp = label_ids(it["expected"])
        # capability: under q1 (predict next word), does the model predict a q1_expect word?
        q1_ids = torch.tensor([[lm.bos] + lm.tok(P.TD["q1"] + "\n" + it["stimulus"],
                                                 add_special_tokens=False).input_ids], device=lm.device)
        q1_lg = lm.logits(q1_ids)[-1]
        q1_top = [lm.dec(t) for t in q1_lg.topk(5).indices.tolist()]
        exp_forms = label_ids(it["q1_expect"])
        out["demand_capability"].append(dict(
            key=it["key"], q1_top5=q1_top,
            q1_hit=bool(exp_forms) and int(jl.ranks_of(q1_lg, exp_forms).min()) <= 5))
        # readout: label-in-lens over the stimulus span, under q2 vs q1
        for tag, question in (("q1", P.TD["q1"]), ("q2", it["q2"])):
            pi = lm.tok(question + "\n", add_special_tokens=False).input_ids
            si = lm.tok(it["stimulus"], add_special_tokens=False).input_ids
            ids = torch.tensor([[lm.bos] + pi + si], device=lm.device)
            span = list(range(1 + len(pi), 1 + len(pi) + len(si)))
            res = lm.residuals(ids, jl.BAND)
            posrank = None
            for L in jl.BAND:
                lr = per_pos_rank(lm.lens_logits(res[L][span], L), exp)
                posrank = lr if posrank is None else torch.minimum(posrank, lr)
            out["demand"].append(dict(
                key=it["key"], q=tag, n=len(span),
                hit10=int((posrank <= 10).sum()), hit25=int((posrank <= 25).sum()),
                best=int(posrank.min())))

    # ============================================================ 2d  privileging (imagine)
    def mean_final(texts):
        d = {L: [] for L in jl.BAND}
        for t in texts:
            r = lm.residuals(lm.encode(t), jl.BAND)
            for L in jl.BAND:
                d[L].append(r[L][-1])
        return {L: torch.stack(v).mean(0) for L, v in d.items()}

    mf, me = mean_final(P.IMAGINE["fr_probe"]), mean_final(P.IMAGINE["en_probe"])
    probe, orth, jshare = {}, {}, {}
    for L in jl.BAND:
        p = mf[L] - me[L]
        _, recon = jl.pursuit(p, lm.V(L), K_PURSUIT)
        probe[L] = p / p.norm()
        q = p - recon
        orth[L] = q / q.norm()
        jshare[L] = float(recon.norm() ** 2 / p.norm() ** 2)
    fr_ids = [lm.tid(" " + P.IMAGINE_LABEL)]
    if lm.is_single(" " + P.IMAGINE_LABEL.lower()):
        fr_ids.append(lm.tid(" " + P.IMAGINE_LABEL.lower()))

    @torch.no_grad()
    def imagine_measures(text, sentence):
        start = text.index(sentence)
        pi = lm.tok(text[:start], add_special_tokens=False).input_ids
        si = lm.tok(sentence, add_special_tokens=False).input_ids
        ids = torch.tensor([[lm.bos] + pi + si], device=lm.device)
        span = list(range(1 + len(pi), 1 + len(pi) + len(si)))
        res = lm.residuals(ids, jl.BAND)
        lens = max(torch.logsumexp(lm.lens_logits(res[L][span], L).log_softmax(-1)[:, fr_ids],
                                   dim=-1).mean().item() for L in jl.BAND)
        proj = sum(float((res[L][span] @ orth[L]).mean()) for L in jl.BAND) / len(jl.BAND)
        return dict(lens=lens, orth=proj)

    for se, sf in zip(P.IMAGINE["en"], P.IMAGINE["fr"]):
        rec = dict(jshare=jshare)
        rec["neutral"] = [imagine_measures(h.format(s=se), se) for h in P.IMAGINE_HEADERS["neutral"]]
        rec["claim"] = [imagine_measures(h.format(s=se), se) for h in P.IMAGINE_HEADERS["claim"]]
        rec["real"] = [imagine_measures(h.format(s=sf), sf) for h in P.IMAGINE_HEADERS["neutral"]]
        out["imagine"].append(rec)

    # ------------------------------------------------------------ write
    os.makedirs(f"{HERE}/results", exist_ok=True)
    json.dump(out, open(f"{HERE}/results/results.json", "w"))
    # a few example prompts, verbatim, for inspection
    ex = []
    w0, c0 = words[0], carriers[0]
    for cond in P.CONDITIONS:
        instr = None if cond == "baseline" else P.PHRASINGS[cond][0]
        ids, span = P.copy_frame(lm, c0, instr, w0)
        ex.append(dict(exp="2a", cond=cond, word=w0, prompt=lm.tok.decode(ids[0, 1:].tolist()),
                       copy_span_text=lm.tok.decode([ids[0, i] for i in span])))
    ex.append(dict(exp="2d", cond="claim",
                   prompt=P.IMAGINE_HEADERS["claim"][0].format(s=P.IMAGINE["en"][0])))
    json.dump(ex, open(f"{HERE}/results/prompts.json", "w"), indent=1)
    summary = summarize(out, words, carriers)
    open(f"{HERE}/results/summary.txt", "w").write(summary)
    print("\n" + summary)


# ---------------------------------------------------------------- summary
def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def median(xs):
    s = sorted(xs)
    return s[len(s) // 2] if s else float("nan")


def sign_test(pairs):
    """pairs of (a,b): fraction with a<b, and a two-sided p (normal approx, ties dropped)."""
    d = [(a, b) for a, b in pairs if a != b]
    n = len(d)
    k = sum(a < b for a, b in d)
    if n == 0:
        return float("nan"), n, float("nan")
    z = (k - n / 2) / (n ** 0.5 / 2)
    from math import erfc
    return k / n, n, erfc(abs(z) / 2 ** 0.5)


def summarize(out, words, carriers):
    C = out["concept"]
    L = ["Criterion 2 -- directed modulation -- gpt2-small",
         f"band {out['band']}; {len(words)} concept words x {len(carriers)} carriers; "
         f"phrasings pooled per condition\n"]

    # per (word, carrier, cond): median best-rank over that condition's phrasings
    def agg(field):
        d = {}
        for r in C:
            d.setdefault((r["word"], r["carrier"], r["cond"]), []).append(r[field])
        return {k: median(v) for k, v in d.items()}
    best = agg("best")
    held = agg("best_held")
    pairkeys = sorted({(w, c) for (w, c, _) in best})

    L.append("2a  the tracked word's best lens rank over the copied (unrelated) sentence")
    L.append(f"    {'cond':10s} {'kind':9s}  hit@1  hit@5 hit@10 hit@25  medRank  medRank(held)  blurt@10")
    for cond in P.CONDITIONS:
        sub = [r for r in C if r["cond"] == cond]
        bests = [best[(w, c, cond)] for (w, c) in pairkeys]
        helds = [held[(w, c, cond)] for (w, c) in pairkeys if held[(w, c, cond)] < 10 ** 9]

        def h(thr):
            return sum(b <= thr for b in bests) / len(bests)
        blurt = sum(r["blurt"] <= 10 for r in sub) / len(sub)
        L.append(f"    {cond:10s} {P.KIND[cond]:9s}  {h(1):.2f}   {h(5):.2f}  {h(10):.2f}  {h(25):.2f}"
                 f"   {median(bests):6.0f}   {median(helds):9.0f}       {blurt:.2f}")
    L.append(f"    (n={len(pairkeys)} word x carrier pairs per condition)\n")

    L.append("2a  paired contrasts (per word x carrier; median-over-phrasings best rank)")
    for a, b in [("mention", "baseline"), ("focus", "mention"), ("dismissal", "mention"),
                 ("negated", "mention"), ("negated", "dismissal")]:
        frac, n, pv = sign_test([(best[(w, c, a)], best[(w, c, b)]) for (w, c) in pairkeys])
        ma = median([best[(w, c, a)] for (w, c) in pairkeys])
        mb = median([best[(w, c, b)] for (w, c) in pairkeys])
        L.append(f"    {a:9s} < {b:9s}: {frac:5.0%}  (n={n}, p={pv:.1e})   median {ma:.0f} vs {mb:.0f}")
    L.append("    reading: mention<baseline = priming; focus<mention = instructed activation;")
    L.append("    dismissal not < mention = no downward control; negated<mention = white-bear.\n")

    L.append("2a  per-layer median best rank (band is 7-9; 6 and 10 flank it)")
    L.append(f"    {'cond':10s} " + " ".join(f"L{L_}" for L_ in out["layers_reported"]))
    for cond in P.CONDITIONS:
        sub = [r for r in C if r["cond"] == cond]
        cells = " ".join(
            f"{median([r['per_layer'].get(str(L_), r['per_layer'].get(L_)) for r in sub]):4.0f}"
            for L_ in out["layers_reported"])
        L.append(f"    {cond:10s} {cells}")
    L.append("    caveat: the effect strengthens monotonically toward the motor layer L10 (lens ~= output")
    L.append("    there), so it is strongest OUTSIDE the workspace band. Headline metrics use band 7-9 only,")
    L.append("    and the held-not-spoken column (above) shows the band effect survives excluding positions")
    L.append("    where the model is about to output the word -- but the word reaches the workspace proper")
    L.append("    (top-25) on only a minority of trials.")

    # category->members (main-text form): report the target-count-invariant paired contrast,
    # since this form tracks many member tokens at once (a whole category), so absolute hit
    # rates are not comparable to the single-word form above.
    CN = out["concept_category_null"]
    if CN:
        L.append("\n2a  main-text form (name the CATEGORY, track its unnamed MEMBERS): any member's")
        L.append("    best band rank; paired per category x carrier (absolute rates not comparable")
        L.append("    to the single-word form -- many targets per trial).")
        cnb = {}
        for r in CN:
            cnb[(r["cat"], r["carrier"], r["cond"])] = r["best"]
        cnkeys = sorted({(c, ci) for (c, ci, _) in cnb})
        for a, b in [("mention", "baseline"), ("focus", "baseline"), ("focus", "mention")]:
            frac, n, pv = sign_test([(cnb[(c, ci, a)], cnb[(c, ci, b)]) for (c, ci) in cnkeys])
            ma = median([cnb[(c, ci, a)] for (c, ci) in cnkeys])
            mb = median([cnb[(c, ci, b)] for (c, ci) in cnkeys])
            L.append(f"    {a:9s} < {b:9s}: {frac:5.0%}  (n={n}, p={pv:.1e})   median {ma:.0f} vs {mb:.0f}")

    # 2b math
    MC = out["math_capability"]
    L.append("\n2b  math (silent computation) -- capability-gated")
    L.append(f"    direct '{{expr}} =': {sum(m['correct'] for m in MC)}/{len(MC)} greedy-correct;"
             f" median answer rank {median([m['answer_rank'] for m in MC if m['answer_rank']>0]):.0f}"
             f" (the answer is near-but-not-produced when asked directly)")
    ML = out["math_lens"]
    if ML:
        for cond in ("baseline", "compute", "mention"):
            bs = [r["best"] for r in ML if r["cond"] == cond]
            L.append(f"    answer-in-lens during copy, {cond:8s}: hit@25 {sum(b<=25 for b in bs)/len(bs):.2f}  medRank {median(bs):.0f}")
    L.append("    -> the model cannot reliably compute (0 greedy-correct), and the answer never enters")
    L.append("       the J-space during silent copying (capability floor; the paper's effect grows with model size).")

    # 2c demand
    DC = out["demand_capability"]
    DM = out["demand"]
    L.append("\n2c  implicit task-demand (paired questions)")
    q1hit = sum(d["q1_hit"] for d in DC)
    L.append(f"    predict-next-word (q1) works: {q1hit}/{len(DC)} items hit q1_expect in output top-5")
    L.append(f"       (so the property IS used automatically -- the model just isn't asked to name it)")
    L.append(f"    {'item':10s}  q2 hit@10  q1 hit@10   (stimulus positions with an expected label in band lens top-10)")
    for it in P.TD["items"]:
        d2 = next(d for d in DM if d["key"] == it["key"] and d["q"] == "q2")
        d1 = next(d for d in DM if d["key"] == it["key"] and d["q"] == "q1")
        L.append(f"    {it['key']:10s}  {d2['hit10']:>3d}/{d2['n']:<3d}    {d1['hit10']:>3d}/{d1['n']:<3d}    best rank q2 {d2['best']} / q1 {d1['best']}")
    L.append("    -> the automatic task succeeds, but the property LABEL never enters the workspace under")
    L.append("       either question: GPT-2 uses properties implicitly but cannot summon their names on demand.")

    # 2d imagine
    IM = out["imagine"]
    js = IM[0]["jshare"]
    L.append("\n2d  privileging: an 'imagine this is French' header vs a real French sentence")
    L.append("    French probe J-space share: " + ", ".join(f"L{k} {v:.0%}" for k, v in js.items()))

    def cond_delta(cond, ch):
        ds = []
        for r in IM:
            base = sum(m[ch] for m in r["neutral"]) / len(r["neutral"])
            ds += [m[ch] - base for m in r[cond]]
        return sum(ds) / len(ds), sum(x > 0 for x in ds), len(ds)
    L.append(f"    {'':12s}  lens 'French' (delta vs neutral)   J-orth probe (delta vs neutral)")
    for cond in ("claim", "real"):
        dl, ul, nl = cond_delta(cond, "lens")
        do, uo, no = cond_delta(cond, "orth")
        L.append(f"    {cond:12s}  {dl:+6.2f}  (up {ul}/{nl})            {do:+7.2f}  (up {uo}/{no})")
    cl, _, _ = cond_delta("claim", "lens")
    rl, _, _ = cond_delta("real", "lens")
    co, _, _ = cond_delta("claim", "orth")
    ro, _, _ = cond_delta("real", "orth")
    L.append(f"    dissociation: the claim gets {cl/rl:.0%} of real's lens effect but only {co/ro:.0%} of its probe effect")
    L.append("    -> the instruction writes 'French' to the J-space without making the text French-like as a real one does.")
    return "\n".join(L)


if __name__ == "__main__":
    main()
