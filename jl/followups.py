"""Follow-up experiments that the five criterion modules do not cover.

  protect   §3.5.2 ablation battery with and without the paper's rule "we do not ablate any
            tokens that appear in the top-10 tokens of a clean forward pass", plus how often that
            rule fires on each task.  Asks how much of criterion 5's selectivity comes from the rule.
  ignition  §4.1.1 (Fig 29, App Figs 70-71): a country token's input embedding is replaced by a
            mixture of two countries' embeddings; per layer, how sharply does the residual at that
            position switch from one country to the other as the mixture weight sweeps 0 -> 1, and
            is the switch bimodal across carrier sentences at the most ambiguous mixture?
  lists     §4.2 (Fig 31): comma-separated word lists, single-category vs unrelated words; at each
            comma, how many list words already read sit in the band J-lens top-25?
  neurons   App Fig 77: for each MLP neuron, is its input-weight direction's best match a J-lens
            vector or an adjacent-layer neuron direction?  (Paper: ~15% before the workspace,
            60-65% within it.)

Writes results/followups/{name}.json and prints a summary.
Run:  python -m jl.followups [protect|ignition|lists|neurons]   (default: all)
"""
from __future__ import annotations

import itertools
import json
import random
import sys
import time

import torch
from jlens.hooks import ActivationRecorder

import jl
from jl import ablation_edits, ablation_select
from jl.stats import median

OUT = "followups"


def _save(name, obj):
    json.dump(obj, open(jl.results_dir(OUT) / f"{name}.json", "w"), indent=1, default=float)


# =============================================================================== protect
@torch.no_grad()
def protect():
    """Criterion 5's S2a battery, J-space ablation with the output-top-10 protection (the paper's
    rule, and what c5 runs) and without it, each against its own matched-norm random control.

    Also records, per task and layer, how often the protection rule changes which directions are
    ablated: the fraction of items whose lens top-10 (at the scored position) contains a token from
    the clean output top-10, and the lens rank of the answer token itself."""
    from jl.c5_selectivity import (KS_ABLATE, induction_items, one_hop_items, pretraining_paragraphs,
                                   two_hop_items)
    lm = jl.Lensed()
    strengths = {"light": [8], "medium": [7, 8, 9]}
    tasks = [("two_hop", two_hop_items(lm)), ("one_hop", one_hop_items(lm)),
             ("induction", induction_items(lm))]
    paras = pretraining_paragraphs()
    out = dict(strengths=strengths, ks=list(KS_ABLATE), rows=[], overlap=[])

    for name, items in tasks:
        # how often the rule fires at the scored (final) position, and where the answer sits
        for L in range(12):
            fires = ans_in_lens = 0
            ranks = []
            for it in items:
                h = lm.residuals(it["ids"], [L])[L][-1]
                lens_top = set(lm.lens_logits(h, L).topk(10).indices.tolist())
                out_top = set(lm.logits(it["ids"])[-1].topk(10).indices.tolist())
                fires += bool(lens_top & out_top)
                ans_in_lens += it["ans_id"] in lens_top
                ranks.append(int(jl.ranks_of(lm.lens_logits(h, L), [it["ans_id"]])[0]))
            out["overlap"].append(dict(task=name, layer=L, n=len(items), fires=fires / len(items),
                                       answer_in_lens_top10=ans_in_lens / len(items),
                                       answer_lens_rank_median=median(ranks)))
        for sname, layers in strengths.items():
            for k in KS_ABLATE:
                row = dict(task=name, strength=sname, k=k, n=len(items))
                for prot in (10, 0):
                    jh = rh = 0
                    for it in items:
                        cl = lm.residuals(it["ids"], layers)
                        sel = ablation_select(lm, it["ids"], layers, k=k, exclude_output_top=prot, clean=cl)
                        jh += int(lm.logits(it["ids"], ablation_edits(sel, lm))[-1].argmax()) == it["ans_id"]
                        rh += int(lm.logits(it["ids"], ablation_edits(sel, lm, random=True))[-1].argmax()) == it["ans_id"]
                    row[f"J_protect{prot}"] = jh / len(items)
                    row[f"R_protect{prot}"] = rh / len(items)
                out["rows"].append(row)
                print(row, flush=True)

    # wikitext next-token top-1 match
    for L in range(12):
        fires = n = 0
        for para in paras:
            ids = lm.encode(para)[:, :96]
            h = lm.residuals(ids, [L])[L]
            lens_top = lm.lens_logits(h, L).topk(10).indices
            out_top = lm.logits(ids).topk(10).indices
            for p in range(5, ids.shape[1]):
                fires += bool(set(lens_top[p].tolist()) & set(out_top[p].tolist()))
                n += 1
        out["overlap"].append(dict(task="pretrain_match", layer=L, n=n, fires=fires / n))
    for sname, layers in strengths.items():
        for k in KS_ABLATE:
            row = dict(task="pretrain_match", strength=sname, k=k, n=len(paras))
            for prot in (10, 0):
                jm = rm = tot = 0
                for para in paras:
                    ids = lm.encode(para)[:, :96]
                    base = lm.logits(ids).argmax(-1)
                    pos = list(range(5, ids.shape[1]))
                    cl = lm.residuals(ids, layers)
                    sel = ablation_select(lm, ids, layers, k=k, exclude_output_top=prot, clean=cl)
                    jm += int((lm.logits(ids, ablation_edits(sel, lm)).argmax(-1)[pos] == base[pos]).sum())
                    rm += int((lm.logits(ids, ablation_edits(sel, lm, random=True)).argmax(-1)[pos] == base[pos]).sum())
                    tot += len(pos)
                row[f"J_protect{prot}"] = jm / tot
                row[f"R_protect{prot}"] = rm / tot
            out["rows"].append(row)
            print(row, flush=True)

    _save("protect", out)
    lines = ["protect: task score under J-space ablation, with / without the output-top-10 rule",
             f"  {'task':15s}{'strength':8s}{'k':>3s}{'n':>4s}   J(rule) R(rule)   J(none) R(none)"]
    for r in out["rows"]:
        lines.append(f"  {r['task']:15s}{r['strength']:8s}{r['k']:3d}{r['n']:4d}   {r['J_protect10']:.2f}    {r['R_protect10']:.2f}      "
                     f"{r['J_protect0']:.2f}    {r['R_protect0']:.2f}")
    lines.append("  how often the rule fires (lens top-10 meets output top-10) at the scored position:")
    for task in ("two_hop", "one_hop", "induction", "pretrain_match"):
        cells = [o for o in out["overlap"] if o["task"] == task]
        lines.append(f"    {task:15s} " + " ".join(f"L{o['layer']}:{o['fires']:.2f}" for o in cells))
    lines.append("  answer token's median lens rank at the scored position:")
    for task in ("two_hop", "one_hop", "induction"):
        cells = [o for o in out["overlap"] if o["task"] == task]
        lines.append(f"    {task:15s} " + " ".join(f"L{o['layer']}:{o['answer_lens_rank_median']}" for o in cells))
    print("\n".join(lines))


# =============================================================================== ignition
IGN_ALPHAS = [i / 20 for i in range(21)]
IGN_PAIRS = 16
IGN_CARRIERS = 40


@torch.no_grad()
def ignition():
    """Embedding-mixture sweep (paper §4.1.1).  For each (country pair A/B, carrier sentence), the
    {W} token's input embedding is set to (1-a) e_B + a e_A for a on a 21-point grid, and the
    residual at that position is recorded at every layer.  Readouts per layer:

      share_full   where the residual sits on the line from the a=0 residual to the a=1 residual
                   (0 at a=0, 1 at a=1 by construction)
      share_J      the same, restricted to the span of the two countries' centered J-lens vectors
      width        a(share=0.9) - a(share=0.1), per trial (the paper's transition width)
      bimodality   at each pair's most ambiguous a (mean share closest to 0.5), the fraction of
                   carriers whose share is within 0.25 of 0.5 (low = bimodal / committed)
      best rank    J-lens rank of whichever country ranks higher, at the most ambiguous a
    """
    lm = jl.Lensed()
    D = json.load(open(jl.REF_DATA / "ignition.json"))
    countries = [c for c in D["countries_12"] if lm.is_single(" " + c)]
    rng = random.Random(0)
    pairs = rng.sample(list(itertools.combinations(countries, 2)), IGN_PAIRS)
    carriers = D["ctx_templates"][:IGN_CARRIERS]
    wte = lm.m._embed_tokens.weight
    layers = list(range(12))
    res = {}  # (pair, carrier) -> [n_alpha, n_layer+1, d]  (index 0 = embedding output)
    t0 = time.time()
    for A, B in pairs:
        ea, eb = wte[lm.tid(" " + A)], wte[lm.tid(" " + B)]
        for ci, tmpl in enumerate(carriers):
            pre, post = tmpl.split("{W}")
            pre_ids = lm.tok(pre.rstrip(), add_special_tokens=False).input_ids
            post_ids = lm.tok(post, add_special_tokens=False).input_ids if post else []
            ids = [lm.bos] + pre_ids + [lm.tid(" " + A)] + post_ids
            w = 1 + len(pre_ids)
            emb = wte[torch.tensor(ids, device=lm.device)][None].repeat(len(IGN_ALPHAS), 1, 1)
            for i, a in enumerate(IGN_ALPHAS):
                emb[i, w] = (1 - a) * eb + a * ea
            with ActivationRecorder(lm.m.layers, at=layers) as rec:
                lm.m._text_module(inputs_embeds=emb, use_cache=False)
                acts = torch.stack([rec.activations[L][:, w].float() for L in layers], 1)
            res[(A, B, ci)] = acts                                   # [n_alpha, 12, d]
    print("ignition forward passes", round(time.time() - t0), "s", flush=True)

    def crossing(xs, level):
        for i in range(1, len(xs)):
            if (xs[i - 1] - level) * (xs[i] - level) <= 0 and xs[i] != xs[i - 1]:
                f = (level - xs[i - 1]) / (xs[i] - xs[i - 1])
                return IGN_ALPHAS[i - 1] + f * (IGN_ALPHAS[i] - IGN_ALPHAS[i - 1])
        return None

    out = dict(pairs=pairs, n_carriers=len(carriers), per_layer=[])
    for L in layers:
        widths = {"full": [], "J": []}
        shares = {"full": {}, "J": {}}
        best_rank = []
        for (A, B, ci), acts in res.items():
            h = acts[:, L]                                           # [n_alpha, d]
            d_full = h[-1] - h[0]
            s_full = ((h - h[0]) @ d_full) / d_full.norm() ** 2
            V = torch.stack([lm.v(L, lm.tid(" " + A)), lm.v(L, lm.tid(" " + B))])
            Q, _ = torch.linalg.qr(V.T)                              # [d, 2] orthonormal span
            hp = (h - h[0]) @ Q
            d_J = hp[-1]
            s_J = (hp @ d_J) / d_J.norm().clamp_min(1e-9) ** 2
            for key, s in (("full", s_full), ("J", s_J)):
                s = s.tolist()
                shares[key][(A, B, ci)] = s
                a10, a90 = crossing(s, 0.1), crossing(s, 0.9)
                if a10 is not None and a90 is not None:
                    widths[key].append(abs(a90 - a10))
        row = dict(layer=L, width_full=median(widths["full"]), width_J=median(widths["J"]))
        for key in ("full", "J"):
            mid_frac, dist = [], []
            for A, B in pairs:
                trials = [shares[key][(A, B, ci)] for ci in range(len(carriers))]
                mean = [sum(t[i] for t in trials) / len(trials) for i in range(len(IGN_ALPHAS))]
                i_star = min(range(len(IGN_ALPHAS)), key=lambda i: abs(mean[i] - 0.5))
                vals = [t[i_star] for t in trials]
                mid_frac.append(sum(abs(v - 0.5) < 0.25 for v in vals) / len(vals))
                dist += vals
                if key == "full":
                    for ci in range(len(carriers)):
                        h = res[(A, B, ci)][i_star, L]
                        lg = lm.lens_logits(h, L)
                        best_rank.append(int(jl.ranks_of(lg, [lm.tid(" " + A), lm.tid(" " + B)]).min()))
            row[f"mid_frac_{key}"] = sum(mid_frac) / len(mid_frac)
            row[f"hist_{key}"] = torch.histc(torch.tensor(dist).clamp(-0.25, 1.25), bins=12, min=-0.25, max=1.25).tolist()
        row["best_rank_at_ambiguous"] = median(best_rank) if best_rank else None
        out["per_layer"].append(row)
    _save("ignition", out)
    print("ignition: per layer, median 10->90% transition width in alpha (full residual | J-span), "
          "fraction of carriers with share in [0.25,0.75] at the most ambiguous alpha, and the better "
          "country's median J-lens rank there")
    for r in out["per_layer"]:
        print(f"  L{r['layer']:<2d} width {r['width_full']:.3f} | {r['width_J']:.3f}   "
              f"mid-fraction {r['mid_frac_full']:.2f} | {r['mid_frac_J']:.2f}   best rank {r['best_rank_at_ambiguous']}")


# =============================================================================== lists
LIST_LEN = 80
N_LISTS = 8
LIST_K = 25


@torch.no_grad()
def lists():
    """Paper §4.2 Fig 31 B: at each comma of an 80-word list, the number of list words read so far
    whose band-min J-lens rank is <= 25 (solid line), and the number of all 80 list words (dashed).
    Related lists = one category (names, surnames, countries, cities); unrelated lists = random
    lowercase single-token words from the GPT-2 vocabulary."""
    lm = jl.Lensed()
    C = json.load(open(jl.REF_DATA / "capacity.json"))
    pools = {p["name"]: [w for w in p["pool"] if lm.is_single(" " + w)] for p in C["candidate_pools"]}
    rng = random.Random(0)
    fams = [f for f in ("names", "surnames", "countries", "cities") if len(pools.get(f, [])) >= LIST_LEN]
    vocab_words = [lm.dec(t) for t in range(lm.vocab)]
    generic = sorted({w.strip() for w in vocab_words
                      if w.startswith(" ") and w[1:].isalpha() and w[1:].islower() and len(w) >= 6})
    band = jl.BAND
    out = dict(families=fams, pool_sizes={f: len(pools[f]) for f in fams}, related=[], unrelated=[],
               band=band, single_layer=[], k=LIST_K)

    def run(words):
        text = "List:" + ",".join(" " + w for w in words)
        ids = lm.encode(text)
        toks = ids[0].tolist()
        comma = lm.tid(",")
        commas = [i for i, t in enumerate(toks) if t == comma]
        wid = [lm.tid(" " + w) for w in words]
        res = lm.residuals(ids, band)
        read, allw, single = [], [], []
        for j, p in enumerate(commas):          # comma after word j+1
            ranks = torch.stack([jl.ranks_of(lm.lens_logits(res[L][p], L), wid) for L in band])  # [3, 80]
            best = ranks.min(0).values
            read.append(int((best[: j + 1] <= LIST_K).sum()))
            allw.append(int((best <= LIST_K).sum()))
            single.append(int((ranks[band.index(8)][: j + 1] <= LIST_K).sum()))
        return read, allw, single

    for i in range(N_LISTS):
        fam = fams[i % len(fams)]
        words = rng.sample(pools[fam], LIST_LEN)
        r, a, s = run(words)
        out["related"].append(dict(family=fam, read=r, all=a, single_L8=s))
        words = rng.sample(generic, LIST_LEN)
        r, a, s = run(words)
        out["unrelated"].append(dict(read=r, all=a, single_L8=s))
        print(f"list {i} ({fam}) done", flush=True)
    _save("lists", out)

    def avg(key, cond, idx):
        return sum(x[key][idx] for x in out[cond]) / len(out[cond])
    print(f"lists: words present in band J-lens top-{LIST_K} at comma i (mean over {N_LISTS} lists)")
    print("   i   related: read-so-far / all-80 / read@L8     unrelated: read-so-far / all-80 / read@L8")
    for idx in (0, 1, 3, 7, 15, 31, 47, 63, 78):
        print(f"  {idx+1:3d}   {avg('read','related',idx):5.1f} / {avg('all','related',idx):5.1f} / {avg('single_L8','related',idx):4.1f}"
              f"            {avg('read','unrelated',idx):5.1f} / {avg('all','unrelated',idx):5.1f} / {avg('single_L8','unrelated',idx):4.1f}")


@torch.no_grad()
def blocks():
    """Paper Fig 31 E/F: 80-word lists made of four 20-word blocks from different categories
    (names, surnames, countries, cities, in shuffled order).  At each comma, the fraction of each
    block's words that are in the band J-lens top 25; summarized as the presence of a block's words
    during that block (after its first 5 words) and during the next block (after its first 5 words)."""
    lm = jl.Lensed()
    C = json.load(open(jl.REF_DATA / "capacity.json"))
    pools = {p["name"]: [w for w in p["pool"] if lm.is_single(" " + w)] for p in C["candidate_pools"]}
    fams = [f for f in C["block_families"] if len(pools.get(f, [])) >= 20]
    rng = random.Random(0)
    band = jl.BAND
    during, after, curves = [], [], []
    for trial in range(8):
        order = rng.sample(fams, len(fams))
        blocks_ = [rng.sample(pools[f], 20) for f in order]
        words = [w for b in blocks_ for w in b]
        if len(set(words)) < len(words):
            continue
        ids = lm.encode("List:" + ",".join(" " + w for w in words))
        comma = lm.tid(",")
        commas = [i for i, t in enumerate(ids[0].tolist()) if t == comma]
        res = lm.residuals(ids, band)
        wid = [lm.tid(" " + w) for w in words]
        pres = torch.zeros(len(commas), len(words))
        for j, p in enumerate(commas):
            r = torch.stack([jl.ranks_of(lm.lens_logits(res[L][p], L), wid) for L in band]).min(0).values
            pres[j] = (r <= LIST_K).float()
        curves.append(pres.tolist())
        for b in range(len(blocks_)):
            cols = slice(20 * b, 20 * b + 20)
            d = pres[20 * b + 5:20 * b + 19, cols].mean().item()      # commas inside block b
            during.append(d)
            if b + 1 < len(blocks_):
                a = pres[20 * (b + 1) + 5:20 * (b + 1) + 19, cols].mean().item()   # inside block b+1
                after.append(a)
        print(f"trial {trial} {order} done", flush=True)
    out = dict(during=during, after=after, curves=curves, k=LIST_K)
    _save("blocks", out)
    print(f"blocks: fraction of a block's 20 words in the band J-lens top-{LIST_K}: during the block "
          f"{sum(during)/len(during):.3f}, during the next block {sum(after)/len(after):.3f}")


# =============================================================================== neurons
@torch.no_grad()
def neurons():
    """App Fig 77: each MLP neuron of block L+1 (input weights, ln_2 gain folded in, mean-centered)
    is matched by |cos| against a pool of N J-lens vectors at layer L plus N neuron output
    directions of block L; fraction of neurons whose best match is a J-lens vector (50% = chance)."""
    lm = jl.Lensed()
    gen = torch.Generator().manual_seed(0)
    out = []
    for L in range(11):
        blk = lm.m.layers[L + 1]
        g = blk.ln_2.weight.float()
        Win = blk.mlp.c_fc.weight.float()                      # [d, d_ff]; neuron i reads Win[:, i]
        read = (Win * g[:, None]).T                            # [d_ff, d]
        read = read - read.mean(-1, keepdim=True)
        read = jl.unit(read)
        N = 3072
        J = lm.V(L)[torch.randperm(lm.vocab, generator=gen)[:N].to(lm.device)]
        J = jl.unit(J - J.mean(-1, keepdim=True))              # LayerNorm only sees the centered part
        W = lm.m.layers[L].mlp.c_proj.weight.float()           # [d_ff, d] block L neuron outputs
        Wn = jl.unit(W[torch.randperm(W.shape[0], generator=gen)[:N].to(lm.device)])
        Wn = Wn - Wn.mean(-1, keepdim=True)
        Wn = jl.unit(Wn)
        R = torch.randn(N, lm.d, generator=gen).to(lm.device)
        R = jl.unit(R - R.mean(-1, keepdim=True))
        cj = (read @ J.T).abs().max(1).values
        cn = (read @ Wn.T).abs().max(1).values
        cr = (read @ R.T).abs().max(1).values
        out.append(dict(layer=L, frac_J=float((cj > cn).float().mean()),
                        frac_J_vs_random=float((cj > cr).float().mean()),
                        median_best_cos_J=float(cj.median()), median_best_cos_neuron=float(cn.median()),
                        median_best_cos_random=float(cr.median())))
        print(out[-1], flush=True)
    _save("neurons", out)


# =============================================================================== variants
def _set_variant(lm, variant):
    """centered J-lens (the repo's default), raw J-lens (rows of W_U J_L, no centering: the paper's
    literal definition), or the logit lens (J_L = I, centered), for readout and interventions alike."""
    if variant == "logit":
        for L in lm.layers:
            lm.J[L] = torch.eye(lm.d, device=lm.device)
    lm._V = {}
    if variant == "raw":
        for L in list(lm.layers) + [lm.final]:
            lm._V[L] = lm.U @ lm.J[L]


@torch.no_grad()
def variants(configs=None, name="variants"):
    """Three headline interventions under three choices of lens vectors: the centered J-lens (what
    every criterion module uses), the raw J-lens (no vocabulary-mean subtraction), and the centered
    logit lens (J = I).  C1b subtract-and-add swap (gate-passed categories, targets starting at
    rank >= 11; top-5 and top-1), C3 E3 coordinate swap (top-1), and C5 S2a ablation at the band's
    middle layer on two-hop and wikitext next-token match, J vs matched-norm random, at the scaled
    k (c5's K_ABLATE) and the paper's k = 10.
    `configs` = [(label, variant, band)]; the default compares lens variants on band 7-9."""
    from jl import c1_report as c1
    from jl.c3_reasoning import COUNTRIES, FAMILIES
    from jl.c5_selectivity import KS_ABLATE, pretraining_paragraphs, two_hop_items
    configs = configs or [(v, v, jl.BAND) for v in ("centered", "raw", "logit")]
    out = {}
    for label, variant, band in configs:
        lm = jl.Lensed()
        _set_variant(lm, variant)
        row = dict(variant=variant, band=band)
        # C1b
        hits5 = hits1 = n = 0
        for cat in c1.CATEGORIES:
            ids = lm.encode(c1.prompt(lm, cat))
            lg = lm.logits(ids)[-1]
            mem = [lm.tid(" " + w) for w in c1.members(lm, cat)]
            greedy = int(lg.argmax())
            if greedy not in mem:
                continue
            clean = lm.residuals(ids, band)
            for w in c1.ten(lm, cat):
                t = lm.tid(" " + w)
                if t == greedy or int(jl.ranks_of(lg, [t])[0]) < 11:
                    continue
                after = int(jl.ranks_of(lm.logits(ids, jl.swap_edits(lm, ids, greedy, t, band, clean=clean))[-1], [t])[0])
                hits5 += after <= 5
                hits1 += after == 1
                n += 1
        row["c1b_top5"], row["c1b_top1"], row["c1b_n"] = hits5 / n, hits1 / n, n
        # C3 E3
        items = []
        for fam, argf, ansf, tmpl in FAMILIES:
            fixed = tmpl.lower().replace("{arg}", "")
            for c, f in COUNTRIES.items():
                if argf == "language" and not f["language_unique"]:
                    continue
                arg, ans = f[argf], f[ansf]
                if not lm.is_single(" " + c) or not lm.is_single(" " + ans) or c.lower() in fixed or ans.lower() in fixed:
                    continue
                ids = lm.encode(tmpl.format(arg=arg))
                lg = lm.logits(ids)[-1]
                if int(lg.argmax()) == lm.tid(" " + ans):
                    items.append(dict(fam=fam, c=c, ans=ans, ids=ids, lg=lg, clean=lm.residuals(ids, band)))
        hits = n = 0
        for it in items:
            for p in items:
                if p["fam"] != it["fam"] or p["c"] == it["c"] or p["ans"] == it["ans"]:
                    continue
                t_ans = lm.tid(" " + p["ans"])
                if int(jl.ranks_of(it["lg"], [t_ans])[0]) <= 10:     # target answer outside the top 10
                    continue
                edits = jl.coord_swap_edits(lm, it["ids"], lm.tid(" " + it["c"]), lm.tid(" " + p["c"]), band, clean=it["clean"])
                hits += int(lm.logits(it["ids"], edits)[-1].argmax()) == t_ans
                n += 1
        row["c3_top1"], row["c3_n"] = hits / n, n
        # C5 S2a light: the band's middle layer
        L8 = [band[len(band) // 2]]
        two = two_hop_items(lm)
        paras = pretraining_paragraphs()
        for k in KS_ABLATE:
            jh = rh = 0
            for it in two:
                sel = ablation_select(lm, it["ids"], L8, k=k)
                jh += int(lm.logits(it["ids"], ablation_edits(sel, lm))[-1].argmax()) == it["ans_id"]
                rh += int(lm.logits(it["ids"], ablation_edits(sel, lm, random=True))[-1].argmax()) == it["ans_id"]
            row[f"c5_twohop_J_k{k}"], row[f"c5_twohop_R_k{k}"] = jh / len(two), rh / len(two)
            jm = rm = tot = 0
            for para in paras:
                ids = lm.encode(para)[:, :96]
                base = lm.logits(ids).argmax(-1)
                pos = list(range(5, ids.shape[1]))
                sel = ablation_select(lm, ids, L8, k=k)
                jm += int((lm.logits(ids, ablation_edits(sel, lm)).argmax(-1)[pos] == base[pos]).sum())
                rm += int((lm.logits(ids, ablation_edits(sel, lm, random=True)).argmax(-1)[pos] == base[pos]).sum())
                tot += len(pos)
            row[f"c5_pretrain_J_k{k}"], row[f"c5_pretrain_R_k{k}"] = jm / tot, rm / tot
        out[label] = row
        print(label, row, flush=True)
    _save(name, out)


@torch.no_grad()
def linear():
    """How much of criterion 1's J-vs-non-J result the lens itself predicts.  To first order, the
    lens scores a perturbation d by how much it raises the target's logit relative to the source's,
    (v_t - v_s) . d.  For each swap trial we compute the cosine between each swap direction (lens,
    J-space part, non-J remainder of the concept vectors) and v_t - v_s, per band layer."""
    from jl import c1_report as c1
    lm = jl.Lensed()
    band = jl.BAND
    words = sorted({w for cat in c1.CATEGORIES for w in c1.members(lm, cat)})
    raw = {w: lm.residuals(lm.encode(c1.CONCEPT_PROMPT.format(concept=w)), band) for w in words}
    rng = random.Random(0)
    parts = {}
    for w in words:
        base = rng.sample([x for x in words if x != w], c1.N_BASELINE)
        parts[w] = {}
        for L in band:
            u = raw[w][L][-1] - torch.stack([raw[b][L][-1] for b in base]).mean(0)
            support, recon = jl.pursuit(u, lm.V(L), c1.K_PURSUIT)
            parts[w][L] = dict(j=recon, n=u - recon, own_in_support=lm.tid(" " + w) in support)
    rows = []
    for cat in c1.CATEGORIES:
        ids = lm.encode(c1.prompt(lm, cat))
        lg = lm.logits(ids)[-1]
        mem = c1.members(lm, cat)
        greedy = int(lg.argmax())
        if greedy not in [lm.tid(" " + w) for w in mem]:
            continue
        sw = lm.dec(greedy).strip()
        for w in c1.ten(lm, cat):
            t = lm.tid(" " + w)
            if t == greedy or int(jl.ranks_of(lg, [t])[0]) < 11:
                continue
            for L in band:
                target_dir = jl.unit(lm.v(L, t) - lm.v(L, greedy))
                lens_dir = jl.unit(jl.unit(lm.v(L, t)) - jl.unit(lm.v(L, greedy)))
                jdir = jl.unit(parts[w][L]["j"] - parts[sw][L]["j"])
                ndir = jl.unit(parts[w][L]["n"] - parts[sw][L]["n"])
                rows.append(dict(cat=cat, target=w, layer=L,
                                 cos_lens=float(lens_dir @ target_dir), cos_j=float(jdir @ target_dir),
                                 cos_nonj=float(ndir @ target_dir),
                                 own_in_support=parts[w][L]["own_in_support"]))
    _save("linear", rows)
    for key in ("cos_lens", "cos_j", "cos_nonj"):
        vals = [r[key] for r in rows]
        print(f"  {key:9s} median {median(vals):+.3f}   (n={len(vals)} trial x layer)")
    print(f"  target token's own J-lens vector in its concept vector's k={c1.K_PURSUIT} support: "
          f"{sum(r['own_in_support'] for r in rows)}/{len(rows)}")


INTRO_Q = ("Researcher: I can inject a thought about a specific word into your mind. "
           "Do you detect an injected thought? If so, what is it about?")
INTRO_A = "\nModel: Yes, I detect an injected thought. The thought is about \""
INTRO_STRENGTHS = (0.0, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 2.0)


@torch.no_grad()
def introspect():
    """A base-model version of the paper's injected-thought test (§3.1, Fig 7).  The paper's user
    turn and prefill, as a two-speaker transcript; a concept's (centered, unit) J-lens vector,
    scaled by the layer's mean residual norm times a strength, is added at every band layer over
    the researcher's question only.  Score: the concept's output rank at the final open quote
    (the report), and, as the paper's position control, its best output rank at the other
    positions of the model's turn (would it blurt the word out before being asked?).  The report
    rank is the better of the ranks at 'about' and at the open quote."""
    lm = jl.Lensed()
    D = json.load(open(jl.REF_DATA / "verbal-introspection.json"))
    concepts = [c["surface"] for c in D["concepts"]
                if lm.is_single(" " + c["surface"]) and lm.is_single(c["surface"])]
    q_ids = lm.tok(INTRO_Q, add_special_tokens=False).input_ids
    a_ids = lm.tok(INTRO_A, add_special_tokens=False).input_ids
    ids = torch.tensor([[lm.bos] + q_ids + a_ids], device=lm.device)
    q_pos = list(range(1, 1 + len(q_ids)))
    # model-turn positions before '... is about "' (the last two positions, 'about' and the open
    # quote, are both natural places to name the thought, so they are not controls)
    a_pos = list(range(1 + len(q_ids), ids.shape[1] - 2))
    clean = lm.residuals(ids, jl.BAND)
    norms = {L: float(clean[L][1:].norm(dim=-1).mean()) for L in jl.BAND}
    out = dict(n_concepts=len(concepts), strengths=list(INTRO_STRENGTHS), rows=[])
    for s in INTRO_STRENGTHS:
        for w in concepts:
            targets = [lm.tid(w), lm.tid(" " + w)]
            edits = [jl.Edit(L, lambda h, pos, v=jl.unit(lm.v(L, lm.tid(" " + w))) * norms[L] * s: h + v, q_pos)
                     for L in jl.BAND]
            lg = lm.logits(ids, edits)
            report = int(min(jl.ranks_of(lg[-1], targets).min(), jl.ranks_of(lg[-2], targets).min()))
            other = min(int(jl.ranks_of(lg[p], targets).min()) for p in a_pos)
            out["rows"].append(dict(strength=s, word=w, report_rank=report, other_best_rank=other))
        rr = [r for r in out["rows"] if r["strength"] == s]
        mrr = sum(1 / r["report_rank"] for r in rr) / len(rr)
        print(f"  strength {s:4.2f}: report median rank {median([r['report_rank'] for r in rr]):6d}  MRR {mrr:.3f}  "
              f"top-1 {sum(r['report_rank'] == 1 for r in rr)}/{len(rr)}  top-10 {sum(r['report_rank'] <= 10 for r in rr)}/{len(rr)}   "
              f"| other model-turn positions: best-rank median {median([r['other_best_rank'] for r in rr])}, "
              f"in top-10 {sum(r['other_best_rank'] <= 10 for r in rr)}/{len(rr)}", flush=True)
    _save("introspect", out)


def bands():
    """The same headline interventions (centered J-lens) under other choices of band."""
    variants([(f"band{b[0]}-{b[-1]}", "centered", b) for b in ([6, 7, 8], [7, 8, 9], [8, 9, 10], [6, 7, 8, 9, 10], [5, 6, 7])],
             name="bands")


if __name__ == "__main__":
    which = sys.argv[1:] or ["protect", "ignition", "lists", "neurons", "variants", "bands"]
    for name in which:
        print(f"===== {name}", flush=True)
        {"protect": protect, "ignition": ignition, "lists": lists, "neurons": neurons,
         "variants": variants, "bands": bands, "linear": linear, "introspect": introspect, "blocks": blocks}[name]()
