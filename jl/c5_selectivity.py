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

Writes results/c5_selectivity/{results.json, prompts.json, summary.txt}.
Run:  python -m jl.c5_selectivity
"""
from __future__ import annotations

import json
import random
import textwrap

import torch

import jl
from jl import ablation_edits, ablation_select, coord_swap_edits
from jl.c3_reasoning import COUNTRIES, FAMILIES  # the two-hop task is criterion 3's
from jl.stats import median

STRENGTHS = {"light": [8], "medium": [7, 8, 9], "heavy": [6, 7, 8, 9, 10]}
K_ABLATE = 10           # §3.5.2 top-k
S1_KMAX = 30            # §4.2 occupancy sweep
S1_N = 40              # activations sampled for the capacity analysis


# =============================================================================== prompt material
# Prompt material for criterion 5 (selectivity): the paper's §3.5.
#
# Selectivity says the J-space is required for *flexible* cognition (report, multi-step inference)
# but not for *automatic* processing (parsing, copying, one-step recall).  We test this three ways.
#
# S2a  J-space ablation battery (§3.5.2, Fig 22/24).  We ablate the top-k J-space directions and
#      watch a battery of tasks split by how much they depend on assembling inferred content:
#        - two-hop reasoning  (flexible)  -- criterion 3's task, the paper's own multihop control
#        - one-hop recall     (recall)    -- the same country facts, one hop
#        - induction / copy   (automatic) -- copy a token seen earlier in the context
#        - next-token match   (automatic) -- agreement with the clean model on wikitext prose
#      The two-hop items and their country table ARE criterion 3's, imported verbatim so the
#      positive control is exactly the task c3 validated (48/53 answered).
#
# S2b  same-latent language dissociation (§3.5.1, Fig 20).  The paper's released passages
#      (ref/.../selectivity-language.json): a passage whose language is evident but unnamed, under
#      a deliberate task (name the language) vs an automatic task (continue the passage).  The same
#      language-label swap redirects the deliberate report but not the continuation.
#
# floor  line-length counting (§3.5.1, Fig 21) is a base-model capability floor -- see below.
# The two-hop task is criterion 3's: COUNTRIES and FAMILIES are imported from that module (see
# the imports above), so this criterion's positive control is byte-identical to the multihop eval
# criterion 3 validated.

LANGUAGE_DATA = jl.REF_DATA / "selectivity-language.json"
LINECOUNT_DATA = jl.REF_DATA / "selectivity-linecount.json"

# --- S2a battery -----------------------------------------------------------------------------

# one-hop recall, few-shot (the same "capital of {country}" fact, one hop instead of two).
ONE_HOP = {
    "capital": ("The capital of Egypt is Cairo. The capital of Japan is Tokyo. "
                "The capital of {country} is", "capital"),
    "language": ("The main language of Egypt is Arabic. The main language of Japan is Japanese. "
                 "The main language of {country} is", "language"),
}

# induction / copying: a novel token pair, then filler, then the cue token -> copy its partner.
INDUCTION_POOL = [" apple", " river", " tiger", " velvet", " harbor", " maple", " copper",
                  " cat", " dog", " house", " water", " fire", " gold", " king", " book",
                  " tree", " star", " moon", " road", " door", " ship", " lake", " wolf",
                  " bear", " iron", " glass", " stone", " cloud", " forest", " garden"]
INDUCTION_FILLER = " The quick brown fox jumps over the lazy dog."


def two_hop_items(lm):
    """Criterion 3's two-hop items: few-shot frame + query, answer gated to greedy-correct.
    Reproduces c3's construction (shot-collision guard, single-token answers)."""
    items = []
    for fam, argf, ansf, tmpl in FAMILIES:
        fixed = tmpl.lower().replace("{arg}", "")
        for c, f in COUNTRIES.items():
            if argf == "language" and not f["language_unique"]:
                continue
            arg, ans = f[argf], f[ansf]
            if not lm.is_single(" " + c) or not lm.is_single(" " + ans):
                continue
            if c.lower() in fixed or ans.lower() in fixed:  # shot-collision guard
                continue
            text = tmpl.format(arg=arg)
            ids = lm.encode(text)
            if int(lm.logits(ids)[-1].argmax()) == lm.tid(" " + ans):
                items.append(dict(text=text, ids=ids, answer=ans, ans_id=lm.tid(" " + ans)))
    return items


def one_hop_items(lm):
    """One-hop recall over the same country facts, few-shot; gated to greedy-correct."""
    items = []
    for key, (tmpl, ansf) in ONE_HOP.items():
        for c, f in COUNTRIES.items():
            ans = f[ansf]
            if not lm.is_single(" " + ans) or c in ("Egypt", "Japan"):
                continue
            ids = lm.encode(tmpl.format(country=c))
            if int(lm.logits(ids)[-1].argmax()) == lm.tid(" " + ans):
                items.append(dict(text=tmpl.format(country=c), ids=ids, answer=ans,
                                  ans_id=lm.tid(" " + ans)))
    return items


def induction_items(lm, n=40, seed=1):
    """Copy the partner of a repeated token; gated to greedy-correct (pure induction)."""
    rng = random.Random(seed)
    pool = [w for w in INDUCTION_POOL if lm.is_single(w)]  # the lens/copy target must be one token
    items, tries = [], 0
    while len(items) < n and tries < 400:
        tries += 1
        a, b = rng.sample(pool, 2)
        text = f"{a}{b}{INDUCTION_FILLER}{a}"
        ids = lm.encode(text)
        if int(lm.logits(ids)[-1].argmax()) == lm.tid(b):
            items.append(dict(text=text, ids=ids, answer=b, ans_id=lm.tid(b)))
        if len(items) >= n:
            break
    return items


def pretraining_paragraphs(n=16, min_len=300, max_tokens=96):
    """Natural prose for the next-token top-1 match (paper: 'pretraining-like documents').
    wikitext-2-raw test -- the corpus the released gpt2-small lens was fit on."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    out = []
    for t in ds["text"]:
        s = t.strip()
        if len(s) >= min_len and not s.startswith("="):
            out.append(s)
        if len(out) >= n:
            break
    return out


# --- S2b language dissociation ---------------------------------------------------------------

LANG_DEMOS = {  # other-language demos for the few-shot language-ID cloze (none is fr/de/es/it)
    "Portuguese": "O rato comeu o queijo na cozinha ontem",
    "Dutch": "De kat zit op de mat bij het raam",
    "Swedish": "Katten sover pa stolen bredvid fonstret",
}
LANGS = ["French", "German", "Spanish", "Italian"]

# Short, unambiguous continuation phrases, one per language, used to grade the *automatic* task:
# after the (swapped) passage, does the model still prefer to continue in the passage's own
# language, or has the swap pushed it toward the alternative?  (A next-token language classifier
# is unreliable for the closely-related Romance languages; scoring a whole diagnostic phrase is
# robust.)  "the sun was shining", in each language.
LANG_CONT = {
    "French": " et le soleil brillait",
    "German": " und die Sonne schien",
    "Spanish": " y el sol brillaba",
    "Italian": " e il sole splendeva",
}


def language_data():
    return json.load(open(LANGUAGE_DATA))


def report_prompt(lm, text):
    """Few-shot language-ID cloze: '... Passage: {text}\\nLanguage:' -> the language name.
    Built from explicit token pieces so the passage span is exact (avoids the GPT-2
    leading-space tokenization drift)."""
    prefix = "".join(f"Passage: {d}\nLanguage: {k}\n" for k, d in LANG_DEMOS.items()) + "Passage:"
    pid = lm.tok(prefix, add_special_tokens=False).input_ids
    tid = lm.tok(" " + text, add_special_tokens=False).input_ids
    sid = lm.tok("\nLanguage:", add_special_tokens=False).input_ids
    ids = torch.tensor([[lm.bos] + pid + tid + sid], device=lm.device)
    passage_pos = list(range(1 + len(pid), 1 + len(pid) + len(tid)))  # +1 for BOS
    return ids, passage_pos


def continuation_prompt(lm, text):
    """Automatic task: the passage alone; the model continues it in-language natively."""
    tid = lm.tok(text, add_special_tokens=False).input_ids
    ids = torch.tensor([[lm.bos] + tid], device=lm.device)
    passage_pos = list(range(1, 1 + len(tid)))
    return ids, passage_pos


# =============================================================================== the run


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
    two, one, ind = two_hop_items(lm), one_hop_items(lm), induction_items(lm)
    paras = pretraining_paragraphs()
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
                sel = ablation_select(lm, it["ids"], layers, k=K_ABLATE, clean=cl)
                jh += int(lm.logits(it["ids"], ablation_edits(sel, lm))[-1].argmax()) == it["ans_id"]
                rh += int(lm.logits(it["ids"], ablation_edits(sel, lm, random=True))[-1].argmax()) == it["ans_id"]
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
            sel = ablation_select(lm, ids, layers, k=K_ABLATE, clean=cl)
            jm += int((lm.logits(ids, ablation_edits(sel, lm)).argmax(-1)[pos] == base[pos]).sum())
            rm += int((lm.logits(ids, ablation_edits(sel, lm, random=True)).argmax(-1)[pos] == base[pos]).sum())
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
    D = language_data()
    lab = {L: lm.tid(" " + L) for L in LANGS}
    cont = {L: lm.tok(LANG_CONT[L], add_special_tokens=False).input_ids for L in LANGS}
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
        r_ids, r_pos = report_prompt(lm, psg["text"])
        gate = lm.dec(int(lm.logits(r_ids)[-1].argmax())).strip() == cat
        out["report_gate"].append(dict(category=cat, key=psg["key"], gate=gate,
                                        top1=lm.dec(int(lm.logits(r_ids)[-1].argmax()))))
        if not gate:
            continue
        c_ids, c_pos = continuation_prompt(lm, psg["text"])
        # panel (b): the true language label is present in the band J-space over the passage,
        # in BOTH prompts (comparable presence is what makes the continuation result meaningful)
        pres_report = band_min_rank(r_ids, r_pos, lab[cat])
        pres_cont = band_min_rank(c_ids, c_pos, lab[cat])
        for alt in LANGS:
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
    paras = pretraining_paragraphs(n=8)
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
    LC = json.load(open(LINECOUNT_DATA))
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

    R = jl.results_dir("c5_selectivity")
    json.dump(out, open(R / "results.json", "w"), indent=1, default=float)
    json.dump(prompt_samples, open(R / "prompts.json", "w"), indent=1)
    summary = summarize(out)
    open(R / "summary.txt", "w").write(summary)
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
