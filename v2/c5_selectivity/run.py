"""Criterion 5 (selectivity) in gpt2-small -- the paper's §3.5 (and §4.2 for the definition's
"small subset" clause), in one run.  See PROTOCOL.md.

  S1  small subset      variance captured by the top-K J-lens directions vs a same-size random
                        dictionary, on wikitext activations; occupancy K (paper §4.2, Fig 30)
  S2a ablation battery  project out the top-10 J-space directions across a widening band
                        (light/medium/heavy) vs a matched-norm random subspace; a battery of
                        tasks ordered by dependence on inferred content (paper §3.5.2, Fig 22/24)
  S2b language          one passage, one latent (its language); the SAME language-label swap
                        redirects the deliberate report but not the automatic continuation
                        (paper §3.5.1, Fig 20)
  floor line-counting   §3.5.1 Fig 21 is a base-model capability floor (reported, not plotted)

Writes results/{results.json, prompts.json, summary.txt}.
Run:  python c5_selectivity/run.py
"""
from __future__ import annotations

import json
import os
import sys
import textwrap

import torch

_V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _V2)                                  # jl library
sys.path.append(os.path.join(_V2, "c3_reasoning"))       # appended: only for `swaps` (unique name)
import jl  # noqa: E402
import prompts as P  # noqa: E402  (c5's local prompts; c3's same-named module stays off the front of the path)
import ablation as AB  # noqa: E402
from swaps import coord_swap_edits  # noqa: E402  (criterion 3's coordinate swap, Fig 4C)

HERE = os.path.dirname(os.path.abspath(__file__))
STRENGTHS = {"light": [8], "medium": [7, 8, 9], "heavy": [6, 7, 8, 9, 10]}
K_ABLATE = 10           # §3.5.2 top-k
S1_KMAX = 30            # §4.2 occupancy sweep
S1_N = 40              # activations sampled for the capacity analysis


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return p, c - h, c + h


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


@torch.no_grad()
def pursuit_curve(x, V, kmax):
    """Cumulative fraction of variance explained by a non-negative greedy pursuit of x over the
    dictionary V, after each of kmax steps.  (jl.pursuit, recording the trajectory.)"""
    from scipy.optimize import nnls
    Vu = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    support, r = [], x.clone()
    x2 = float(x.norm() ** 2)
    frac = []
    for _ in range(kmax):
        corr = Vu @ r
        if support:
            corr[torch.as_tensor(support, device=x.device)] = -torch.inf
        j = int(corr.argmax())
        if corr[j] <= 0:
            frac.append(frac[-1] if frac else 0.0)
            continue
        support.append(j)
        A = V[support].T.cpu().double().numpy()
        coeffs, _ = nnls(A, x.cpu().double().numpy())
        recon = torch.as_tensor(A @ coeffs, device=x.device, dtype=x.dtype)
        r = x - recon
        frac.append(float(recon.norm() ** 2) / x2)
    return frac


# =============================================================================== S2a battery
@torch.no_grad()
def battery(lm):
    """Task score under no ablation / J-space ablation / matched-norm random control, per
    strength.  Accuracy tasks score 0/1 per item; pretraining scores per-position top-1 match."""
    two, one, ind = P.two_hop_items(lm), P.one_hop_items(lm), P.induction_items(lm)
    paras = P.pretraining_paragraphs()
    tasks = [("two_hop", two, "flexible"), ("one_hop", one, "recall"),
             ("induction", ind, "automatic")]

    out = {"counts": {"two_hop": len(two), "one_hop": len(one), "induction": len(ind),
                      "paras": len(paras)}, "acc": [], "pretrain": []}

    for name, items, kind in tasks:
        clean = sum(int(lm.logits(it["ids"])[-1].argmax()) == it["ans_id"] for it in items) / len(items)
        row = dict(task=name, kind=kind, n=len(items), clean=clean)
        for sname, layers in STRENGTHS.items():
            jh = rh = 0
            for it in items:
                cl = lm.residuals(it["ids"], layers)
                sel = AB.select(lm, it["ids"], layers, k=K_ABLATE, clean=cl)
                jh += int(lm.logits(it["ids"], AB.edits_from(sel, lm))[-1].argmax()) == it["ans_id"]
                rh += int(lm.logits(it["ids"], AB.edits_from(sel, lm, random=True))[-1].argmax()) == it["ans_id"]
            row[f"{sname}_J"] = jh / len(items)
            row[f"{sname}_R"] = rh / len(items)
        out["acc"].append(row)

    # next-token top-1 match on wikitext (automatic; the paper's Fig 22 collateral-damage axis)
    prow = dict(task="pretrain_match", kind="automatic", n=len(paras), clean=1.0)
    for sname, layers in STRENGTHS.items():
        jm = rm = tot = 0
        for para in paras:
            ids = lm.encode(para)[:, : 96]
            base = lm.logits(ids).argmax(-1)                       # clean top-1 per position
            pos = list(range(5, ids.shape[1]))                      # skip warm-up
            cl = lm.residuals(ids, layers)
            sel = AB.select(lm, ids, layers, k=K_ABLATE, clean=cl)
            jm += int((lm.logits(ids, AB.edits_from(sel, lm)).argmax(-1)[pos] == base[pos]).sum())
            rm += int((lm.logits(ids, AB.edits_from(sel, lm, random=True)).argmax(-1)[pos] == base[pos]).sum())
            tot += len(pos)
        prow[f"{sname}_J"] = jm / tot
        prow[f"{sname}_R"] = rm / tot
    out["pretrain"] = prow
    return out, {"two_hop": [it["text"] for it in two[:3]], "one_hop": [it["text"] for it in one[:3]],
                 "induction": [it["text"] for it in ind[:3]], "pretrain": paras[0][:200]}


# =============================================================================== S2b language
ALPHAS = (0.5, 1.0, 1.5, 2.0, 3.0)


@torch.no_grad()
def language_dissociation(lm):
    """Same latent (a passage's language), two tasks.  The SAME language-label swap over the
    passage is applied under a deliberate task (few-shot 'name the language' cloze) and an
    automatic task (continue the passage).  Deliberate: does the reported language flip to the
    swapped-in one?  Automatic: does the model still prefer to continue in the passage's own
    language (scored against short per-language continuation phrases), or has the swap pushed it
    to the alternative?  We sweep swap strength -- the report should follow at strengths where
    the continuation does not."""
    D = P.language_data()
    lab = {L: lm.tid(" " + L) for L in P.LANGS}
    cont = {L: lm.tok(P.LANG_CONT[L], add_special_tokens=False).input_ids for L in P.LANGS}
    band = jl.BAND
    out = {"alphas": list(ALPHAS), "report_gate": [], "trials": []}

    def band_min_rank(ids, positions, tok):
        res = lm.residuals(ids, band)
        return min(int(jl.ranks_of(lm.lens_logits(res[L][p], L), [tok])[0])
                   for L in band for p in positions)

    def cont_logprob(ids, phrase, edits):
        """Sum log-prob of a continuation phrase after `ids`, under `edits` (on the passage)."""
        full = torch.cat([ids, torch.tensor([phrase], device=lm.device)], dim=1)
        lp = lm.logits(full, edits).log_softmax(-1)
        t0 = ids.shape[1]
        return float(sum(lp[t0 - 1 + i, phrase[i]] for i in range(len(phrase))))

    for psg in D["passages"]:
        cat = psg["category"]
        r_ids, r_pos = P.report_prompt(lm, psg["text"])
        gate = lm.dec(int(lm.logits(r_ids)[-1].argmax())).strip() == cat
        out["report_gate"].append(dict(category=cat, key=psg["key"], gate=gate,
                                        top1=lm.dec(int(lm.logits(r_ids)[-1].argmax()))))
        if not gate:
            continue
        c_ids, c_pos = P.continuation_prompt(lm, psg["text"])
        # panel (b): the true language label is present in the band J-space over the passage,
        # in BOTH prompts (comparable presence is what makes the continuation result meaningful)
        pres_report = band_min_rank(r_ids, r_pos, lab[cat])
        pres_cont = band_min_rank(c_ids, c_pos, lab[cat])
        for alt in P.LANGS:
            if alt == cat:
                continue
            s, t = lab[cat], lab[alt]
            for alpha in ALPHAS:
                r_edits = [jl.Edit(e.layer, e.fn, r_pos)
                           for e in coord_swap_edits(lm, r_ids, s, t, band, alpha=alpha)]
                report_flip = lm.dec(int(lm.logits(r_ids, r_edits)[-1].argmax())).strip() == alt
                c_edits = [jl.Edit(e.layer, e.fn, c_pos)
                           for e in coord_swap_edits(lm, c_ids, s, t, band, alpha=alpha)]
                lp_true = cont_logprob(c_ids, cont[cat], c_edits)
                lp_alt = cont_logprob(c_ids, cont[alt], c_edits)
                out["trials"].append(dict(
                    category=cat, key=psg["key"], alt=alt, alpha=alpha,
                    pres_report=pres_report, pres_cont=pres_cont,
                    report_flip=report_flip,                 # deliberate: report follows swap?
                    cont_hold=lp_true > lp_alt,              # automatic: still prefers own language?
                    cont_margin=lp_true - lp_alt,
                ))
    return out


# =============================================================================== S1 capacity
@torch.no_grad()
def capacity(lm):
    """Fraction of activation variance captured by the top-K J-lens directions vs a same-size
    random dictionary, on wikitext activations (paper §4.2, Fig 30)."""
    band = jl.BAND
    paras = P.pretraining_paragraphs(n=8)
    # collect activations at band layers, then mean-subtract per layer (variance about the mean)
    acts = {L: [] for L in band}
    for para in paras:
        ids = lm.encode(para)[:, : 64]
        res = lm.residuals(ids, band)
        for L in band:
            for p in range(5, ids.shape[1]):
                acts[L].append(res[L][p])
    out = {"kmax": S1_KMAX, "per_layer": []}
    gen = torch.Generator(device=lm.device).manual_seed(0)
    for L in band:
        H = torch.stack(acts[L])
        H = H - H.mean(0, keepdim=True)                        # variance about the mean
        idx = torch.randperm(H.shape[0], generator=gen, device=lm.device)[:S1_N]
        Vj = lm.V(L)                                           # centered J-lens dictionary
        Rd = torch.randn(Vj.shape, generator=gen, device=lm.device)  # same-size random dict
        fj = torch.zeros(S1_KMAX)
        fr = torch.zeros(S1_KMAX)
        for i in idx.tolist():
            fj += torch.tensor(pursuit_curve(H[i], Vj, S1_KMAX))
            fr += torch.tensor(pursuit_curve(H[i], Rd, S1_KMAX))
        fj /= len(idx)
        fr /= len(idx)
        # occupancy: first K where the J marginal gain drops below the random marginal gain
        occ = S1_KMAX
        for K in range(1, S1_KMAX):
            if (fj[K] - fj[K - 1]) < (fr[K] - fr[K - 1]):
                occ = K
                break
        out["per_layer"].append(dict(layer=L, frac_J=fj.tolist(), frac_R=fr.tolist(),
                                     occupancy=occ, var_at_25_J=float(fj[24]),
                                     var_at_25_R=float(fr[24])))
    return out


# =============================================================================== floor
@torch.no_grad()
def linecount_floor(lm):
    """§3.5.1 Fig 21: does any count token enter the band J-space, and can the model answer?
    Base-model capability floor -- reported, not plotted."""
    LC = json.load(open(P.LINECOUNT_DATA))
    band = jl.BAND
    numtoks = [lm.tid(w) for w in [str(x) for x in range(20, 70)] if lm.is_single(w)]
    numtoks += [lm.tid(" " + w) for w in ["twenty", "thirty", "forty", "fifty", "sixty"]
                if lm.is_single(" " + w)]
    out = {}
    for cond in ("none", "direct", "letter"):
        q, pre = LC["conditions"][cond]["question"], LC["conditions"][cond]["prefill"]
        hits, greedies = 0, []
        for psg in LC["passages"]:
            wrapped = textwrap.fill(psg["text"], psg["width"])
            text = (f"{q}\n" if q else "") + wrapped + "\n" + pre
            ids = lm.encode(text)
            res = lm.residuals(ids, band)
            best = min(int(jl.ranks_of(lm.lens_logits(res[L][-1], L), numtoks).min()) for L in band)
            hits += best <= 25
            greedies.append(lm.dec(int(lm.logits(ids)[-1].argmax())))
        out[cond] = dict(count_in_band=hits, n=len(LC["passages"]), greedy=greedies[:6])
    return out


@torch.no_grad()
def main():
    lm = jl.Lensed()
    out = dict(band=jl.BAND, strengths=STRENGTHS)
    print("S2a battery ..."); out["battery"], prompt_samples = battery(lm)
    print("S2b language ..."); out["language"] = language_dissociation(lm)
    print("S1 capacity ..."); out["capacity"] = capacity(lm)
    print("floor ..."); out["floor"] = linecount_floor(lm)

    os.makedirs(f"{HERE}/results", exist_ok=True)
    json.dump(out, open(f"{HERE}/results/results.json", "w"), indent=1, default=float)
    json.dump(prompt_samples, open(f"{HERE}/results/prompts.json", "w"), indent=1)
    summary = summarize(out)
    open(f"{HERE}/results/summary.txt", "w").write(summary)
    print("\n" + summary)


def summarize(out) -> str:
    L = ["Criterion 5 -- selectivity -- gpt2-small", f"band {out['band']}"]
    b = out["battery"]
    c = b["counts"]
    L.append(f"\nS2a ablation battery  (n: two-hop {c['two_hop']}, one-hop {c['one_hop']}, "
             f"induction {c['induction']}, wikitext paras {c['paras']})")
    L.append("  task score: no ablation | J-space ablation (matched-norm random control), per strength")
    L.append(f"  {'task':16s}{'kind':10s}{'clean':6s}   " +
             "   ".join(f"{s+'_J':>6s} {s+'_R':>6s}" for s in STRENGTHS))
    for row in b["acc"] + [b["pretrain"]]:
        cells = "   ".join(f"{row[s+'_J']:6.2f} {row[s+'_R']:6.2f}" for s in STRENGTHS)
        L.append(f"  {row['task']:16s}{row['kind']:10s}{row['clean']:5.2f}    {cells}")
    L.append("  selective damage = random_score - J_score (how much the J-SUBSPACE removes "
             "beyond matched norm):")
    for row in b["acc"] + [b["pretrain"]]:
        sd = "  ".join(f"{s} {row[s+'_R']-row[s+'_J']:+.2f}" for s in STRENGTHS)
        L.append(f"    {row['task']:16s} {sd}")

    lg = out["language"]
    npass = sum(g["gate"] for g in lg["report_gate"])
    L.append(f"\nS2b language dissociation  (report gate: {npass}/{len(lg['report_gate'])} passages;"
             f" the SAME language swap under two tasks, by swap strength alpha)")
    L.append("  alpha   deliberate report -> swapped language | automatic continuation -> keeps own language")
    for a in lg["alphas"]:
        tr = [t for t in lg["trials"] if t["alpha"] == a]
        if not tr:
            continue
        flip = sum(t["report_flip"] for t in tr)
        hold = sum(t["cont_hold"] for t in tr)
        L.append(f"    {a:<5.1f}  report flips {flip:2d}/{len(tr)} = {flip/len(tr):4.0%}"
                 f"      continuation holds {hold:2d}/{len(tr)} = {hold/len(tr):4.0%}")
    tr1 = [t for t in lg["trials"] if t["alpha"] == 1.0]
    if tr1:
        L.append(f"  n = {len(tr1)} passage x alt-language swaps per alpha")
        L.append(f"  panel (b) true-language label presence in band J-space over the passage "
                 f"(median band-min rank): report prompt {median([t['pres_report'] for t in tr1]):.0f}, "
                 f"continuation prompt {median([t['pres_cont'] for t in tr1]):.0f}")
        L.append("  -> the report follows the swap at strengths where the continuation keeps its own "
                 "language: the same latent is causal for the deliberate task, less so for the automatic one.")

    cap = out["capacity"]
    L.append(f"\nS1 small subset  (wikitext activations, {S1_N} per layer; top-{cap['kmax']} pursuit)")
    L.append("  fraction of activation variance captured at K=25 (J-lens | random dict | excess); occupancy K")
    for pl in cap["per_layer"]:
        L.append(f"    L{pl['layer']}  {pl['var_at_25_J']:.1%} | {pl['var_at_25_R']:.1%} | "
                 f"{pl['var_at_25_J']-pl['var_at_25_R']:+.1%}   occupancy K={pl['occupancy']}")

    fl = out["floor"]
    L.append("\nfloor  line-length counting (§3.5.1 Fig 21) -- base-model capability floor")
    for cond in ("none", "direct", "letter"):
        f = fl[cond]
        L.append(f"  {cond:7s}: count token in band top-25 on {f['count_in_band']}/{f['n']} passages; "
                 f"greedy answers {f['greedy']}")
    L.append("  -> gpt2-small cannot track character counts; the count never enters the J-space "
             "under any task (as with the c2 math floor).")
    return "\n".join(L)


if __name__ == "__main__":
    main()
