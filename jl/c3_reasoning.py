"""Criterion 3 (internal reasoning) in gpt2-small — the paper's §3.3, in one run.
See PROTOCOL.md.

  gate     capability: does the model answer the two-hop query correctly?
  E1 read  per-layer lens rank of the unspoken intermediate at the answer position,
           vs controls: answer (motor), arg (surface echo), null (random same-category)
  E2 case  one clean example: clean vs swapped top-5 next-token log-probs (Fig 13)
  E3 swap  coordinate swap intermediate->alt across the band; swap_answer's output rank
  E4 depth intermediate-swap vs answer-swap at each single layer; onset depth (Fig 15 right)
  E5 priv  probe -> J-space (k=25 pursuit) + remainder; swap along each at matched
           magnitude; remainder also with the J coordinates clamped to clean (Fig 16)
  floor    the paper's own probe-swap.json, verbatim: capability + swap (the deviation cost)

Writes results/c3_reasoning/{results.json, prompts.json, summary.txt}.
Run:  python -m jl.c3_reasoning            (the criterion)
      python -m jl.c3_reasoning swap_ops   (E3 under both swap operations)
"""
from __future__ import annotations

import json
import random
import sys
import time

import torch

import jl
from jl import coord_swap_edits, delta_edits, swap_edits
from jl.stats import median, wilson

REF_PROBE_SWAP = jl.REF_DATA / "probe-swap.json"
K_PURSUIT = 25          # §3.3 privileging sparsity
DEPTH_LAYERS = [4, 5, 6, 7, 8, 9, 10]  # E4 single-layer sweep


# =============================================================================== prompt material
# Prompt material for criterion 3 (internal reasoning): two-hop factual chains
# with a known *unspoken* intermediate.
#
# The paper's §3.3 uses two-hop prompts whose answer depends on first inferring an
# unspoken bridge entity ("the animal that spins webs" -> spider -> eight legs).
# GPT-2-small cannot resolve the paper's riddle phrasings at all (0/6 on the
# spider/Mars riddles; 7/90 on the released probe-swap.json), so — exactly as
# criterion 1 wrapped "Name a {cat}:" in a few-shot list — we present each two-hop
# query in a few-shot frame that teaches the *relation* while leaving the specific
# bridge entity to be computed.
#
# Two relation families, inverses of each other through the country:
#   lang_capital :  language -> (country) -> capital     "...speak {L}, the capital city is called ___"
#   cap_language :  capital  -> (country) -> language     "...governed from {C} ... language, namely ___"
# The intermediate is always the COUNTRY, which never appears in the prompt or the
# answer.  The few-shot shots use a fixed set of demo countries; any test item whose
# country or answer collides with the shot text is dropped (the answer must not be
# echoable from the prompt).
#
# For the privileging experiment (§3.3, Fig 16) we also build a probe for each
# country from several cue prompts that *imply* the country through different
# surface cues and ask about different attributes, with the country name never the
# next token (paper: "prompts that imply the same intermediate through different
# surface cues and ask different questions about it").
#
# The paper's own two-hop set (ref/.../probe-swap.json) is loaded verbatim for the
# capability-floor check below.
# country -> (language, capital, continent, language_unique)
# world knowledge only; single-token capitals/answers are enforced at build time
# below, and capability + the shot-collision guard decide what survives.
COUNTRIES: dict[str, dict] = {
    "France":      dict(language="French",     capital="Paris",     continent="Europe",  language_unique=True),
    "Germany":     dict(language="German",     capital="Berlin",    continent="Europe",  language_unique=True),
    "Spain":       dict(language="Spanish",    capital="Madrid",    continent="Europe",  language_unique=True),
    "Italy":       dict(language="Italian",    capital="Rome",      continent="Europe",  language_unique=True),
    "Poland":      dict(language="Polish",     capital="Warsaw",    continent="Europe",  language_unique=True),
    "Russia":      dict(language="Russian",    capital="Moscow",    continent="Europe",  language_unique=True),
    "China":       dict(language="Chinese",    capital="Beijing",   continent="Asia",    language_unique=True),
    "Portugal":    dict(language="Portuguese", capital="Lisbon",    continent="Europe",  language_unique=True),
    "Greece":      dict(language="Greek",      capital="Athens",    continent="Europe",  language_unique=True),
    "India":       dict(language="Hindi",      capital="Delhi",     continent="Asia",    language_unique=True),
    "Turkey":      dict(language="Turkish",    capital="Ankara",    continent="Asia",    language_unique=True),
    "Sweden":      dict(language="Swedish",    capital="Stockholm", continent="Europe",  language_unique=True),
    "Norway":      dict(language="Norwegian",  capital="Oslo",      continent="Europe",  language_unique=True),
    "Finland":     dict(language="Finnish",    capital="Helsinki",  continent="Europe",  language_unique=True),
    "Hungary":     dict(language="Hungarian",  capital="Budapest",  continent="Europe",  language_unique=True),
    "Iran":        dict(language="Persian",    capital="Tehran",    continent="Asia",    language_unique=True),
    "Korea":       dict(language="Korean",     capital="Seoul",     continent="Asia",    language_unique=True),
    "Vietnam":     dict(language="Vietnamese", capital="Hanoi",     continent="Asia",    language_unique=True),
    "Thailand":    dict(language="Thai",       capital="Bangkok",   continent="Asia",    language_unique=True),
    "Netherlands": dict(language="Dutch",      capital="Amsterdam", continent="Europe",  language_unique=True),
    "Ukraine":     dict(language="Ukrainian",  capital="Kiev",      continent="Europe",  language_unique=True),
    "Denmark":     dict(language="Danish",     capital="Copenhagen", continent="Europe", language_unique=True),
    "Romania":     dict(language="Romanian",   capital="Bucharest", continent="Europe",  language_unique=True),
    "Bulgaria":    dict(language="Bulgarian",  capital="Sofia",     continent="Europe",  language_unique=True),
    "Serbia":      dict(language="Serbian",    capital="Belgrade",  continent="Europe",  language_unique=True),
    "Indonesia":   dict(language="Indonesian", capital="Jakarta",   continent="Asia",    language_unique=True),
    "Pakistan":    dict(language="Urdu",       capital="Islamabad", continent="Asia",    language_unique=True),
    "Iceland":     dict(language="Icelandic",  capital="Reykjavik", continent="Europe",  language_unique=True),
    "Mongolia":    dict(language="Mongolian",  capital="Ulaanbaatar", continent="Asia",  language_unique=True),
    "Kenya":       dict(language="Swahili",    capital="Nairobi",   continent="Africa",  language_unique=True),
    "Nigeria":     dict(language="Yoruba",     capital="Abuja",     continent="Africa",  language_unique=True),
    "Peru":        dict(language="Quechua",    capital="Lima",      continent="America", language_unique=True),
    "Cuba":        dict(language="Spanish",    capital="Havana",    continent="America", language_unique=False),
    "Argentina":   dict(language="Spanish",    capital="Buenos",    continent="America", language_unique=False),
}

# Two-hop families: (name, arg field, answer field, few-shot template).
# Shots use Egypt/Israel (lang_capital) and Egypt/Japan (cap_language); the
# shot-collision guard below drops any item colliding with the shot text.
FAMILIES = [
    ("lang_capital", "language", "capital",
     "In the country where people speak Arabic, the capital city is called Cairo. "
     "In the country where people speak Hebrew, the capital city is called Jerusalem. "
     "In the country where people speak {arg}, the capital city is called"),
    ("cap_language", "capital", "language",
     "The country governed from Cairo has one main language, namely Arabic. "
     "The country governed from Tokyo has one main language, namely Japanese. "
     "The country governed from {arg} has one main language, namely"),
]

# §3.3 privileging: cues that IMPLY the country through varied surface cues and
# ask about a DIFFERENT attribute, so the country name is never the next token.
# Filled per country from its capital / language / continent.
PROBE_CUES = [
    "People in the country whose capital is {capital} mostly speak the language of",
    "A traveler flying into {capital} has landed on the continent of",
    "The homeland of the {language} language lies on the continent of",
    "Newspapers printed in {capital} are usually written in the language of",
    "The country whose largest city is {capital} is located on the continent of",
    "Someone who grew up speaking {language} was most likely born in the city of",
]

CASE_STUDY = dict(family="lang_capital", source="France", target="China")  # Paris -> Beijing


# =============================================================================== the run


class Words:
    """Single-token variant ids of a word ({' Word',' word',...}), cached."""

    def __init__(self, lm):
        self.lm = lm
        self._c: dict[str, list[int]] = {}

    def __call__(self, word: str) -> list[int]:
        if word not in self._c:
            out = set()
            for w in {word, word.capitalize(), word.lower(), word.upper()}:
                for form in (" " + w, w):
                    ids = self.lm.tok(form, add_special_tokens=False).input_ids
                    if len(ids) == 1:
                        out.add(ids[0])
            self._c[word] = sorted(out)
        return self._c[word]


@torch.no_grad()
def main():
    lm = jl.Lensed()
    band = jl.BAND
    W = Words(lm)
    rng = random.Random(0)

    def rank_out(lg, word):
        v = W(word)
        return int(jl.ranks_of(lg, v).min()) if v else 10**9

    def top1(lg):
        return int(lg.argmax())

    # ------------------------------------------------------------------ items + gate
    # A two-hop item: few-shot frame + query; intermediate = country (unspoken).
    items, gate_rows = [], []
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
            lg = lm.logits(ids)[-1]
            greedy = top1(lg)
            passed = greedy in W(ans)
            gate_rows.append(dict(family=fam, country=c, arg=arg, answer=ans,
                                  gate=passed, greedy=lm.dec(greedy)))
            if passed:
                items.append(dict(family=fam, argf=argf, ansf=ansf, text=text, ids=ids,
                                  arg=arg, intermediate=c, answer=ans, lg=lg,
                                  clean=lm.residuals(ids)))
    n_pass = sum(g["gate"] for g in gate_rows)

    # same-family swap partners (different country, different answer, no id overlap)
    countries_pass = {it["intermediate"] for it in items}
    for it in items:
        it["partners"] = [
            p for p in items
            if p["family"] == it["family"] and p["intermediate"] != it["intermediate"]
            and p["answer"] != it["answer"]
            and not (set(W(p["answer"])) & set(W(it["answer"])))
            and not (set(W(p["intermediate"])) & set(W(it["intermediate"])))
        ]

    out = dict(band=band, n_pass=n_pass, n_total=len(gate_rows),
               gate=gate_rows, readout=[], case=None, swap=[], depth=[],
               privilege=[], variance_fraction=[], floor=None)

    # ------------------------------------------------------------------ E1 readout
    # median lens rank per layer of {intermediate, answer, arg, null}; the
    # intermediate should surface in the band while remaining absent from output.
    for it in items:
        res = it["clean"]
        null = rng.choice([c for c in COUNTRIES if c != it["intermediate"] and lm.is_single(" " + c)])
        probes = dict(intermediate=it["intermediate"], answer=it["answer"], arg=it["arg"], null=null)
        per = {name: {} for name in probes}
        for L in lm.layers:
            lens = lm.lens_logits(res[L][-1], L)
            for name, word in probes.items():
                per[name][L] = rank_out(lens, word)
        out["readout"].append(dict(family=it["family"], intermediate=it["intermediate"],
                                   answer=it["answer"], null=null,
                                   int_out_rank=rank_out(it["lg"], it["intermediate"]),
                                   ranks=per))

    # ------------------------------------------------------------------ E2 case study
    cs = next((it for it in items if it["family"] == CASE_STUDY["family"]
               and it["intermediate"] == CASE_STUDY["source"]), None)
    tgt = next((it for it in items if it["family"] == CASE_STUDY["family"]
                and it["intermediate"] == CASE_STUDY["target"]), None)
    if cs and tgt:
        s, t = lm.tid(" " + cs["intermediate"]), lm.tid(" " + tgt["intermediate"])
        sw = lm.logits(cs["ids"], coord_swap_edits(lm, cs["ids"], s, t, band, clean=cs["clean"]))[-1]
        top = lambda lg: [(lm.dec(i), float(lg.log_softmax(-1)[i]))
                          for i in lg.topk(5).indices.tolist()]
        lp = lambda lg, word: float(lg.log_softmax(-1)[W(word)].max())
        out["case"] = dict(text=cs["text"], source=cs["intermediate"], target=tgt["intermediate"],
                           source_answer=cs["answer"], target_answer=tgt["answer"],
                           clean_top5=top(cs["lg"]), swapped_top5=top(sw),
                           # log-prob of the source's answer and target's answer, clean vs swapped
                           answer_lp=dict(source=dict(clean=lp(cs["lg"], cs["answer"]), swapped=lp(sw, cs["answer"])),
                                          target=dict(clean=lp(cs["lg"], tgt["answer"]), swapped=lp(sw, tgt["answer"]))),
                           int_readout={L: [lm.dec(i) for i in lm.lens_logits(cs["clean"][L][-1], L).topk(6).indices.tolist()]
                                        for L in band})

    # ------------------------------------------------------------------ E3 swap
    for it in items:
        s = lm.tid(" " + it["intermediate"])
        for p in it["partners"]:
            before = rank_out(it["lg"], p["answer"])
            if before < 10:            # paper rule: target answer not already in top-10
                continue
            t = lm.tid(" " + p["intermediate"])
            lg = lm.logits(it["ids"], coord_swap_edits(lm, it["ids"], s, t, band, clean=it["clean"]))[-1]
            out["swap"].append(dict(family=it["family"], intermediate=it["intermediate"],
                                    target=p["intermediate"], answer=it["answer"],
                                    swap_answer=p["answer"], before=before,
                                    after=rank_out(lg, p["answer"]),
                                    hit=top1(lg) in W(p["answer"]), got=lm.dec(top1(lg))))

    # ------------------------------------------------------------------ E4 depth control
    # per single layer L: log-prob pushed onto swap_answer by swapping the
    # INTERMEDIATE (A->B) vs the ANSWER (ansA->ansB).  Onset = earliest L reaching
    # half of that trial's max effect.  Intermediate should onset earlier.
    for it in items:
        sI = lm.tid(" " + it["intermediate"])
        ansI = lm.tid(" " + it["answer"]) if lm.is_single(" " + it["answer"]) else None
        for p in it["partners"][:1]:   # one partner/item (paper: random same-category)
            if rank_out(it["lg"], p["answer"]) < 10:
                continue
            tI = lm.tid(" " + p["intermediate"])
            ansT = lm.tid(" " + p["answer"]) if lm.is_single(" " + p["answer"]) else None
            if ansI is None or ansT is None:
                continue
            base = float(it["lg"].log_softmax(-1)[ansT])
            row = dict(intermediate=it["intermediate"], target=p["intermediate"], effect={})
            for kind, (a, b) in dict(interm=(sI, tI), answer=(ansI, ansT)).items():
                eff = {}
                for L in DEPTH_LAYERS:
                    lg = lm.logits(it["ids"], coord_swap_edits(lm, it["ids"], a, b, [L], clean=it["clean"]))[-1]
                    eff[L] = float(lg.log_softmax(-1)[ansT]) - base
                row["effect"][kind] = eff
            out["depth"].append(row)

    # ------------------------------------------------------------------ E5 privileging
    # probe per country from PROBE_CUES (imply country, ask other attribute), minus
    # grand mean; split by pursuit into J-space (k=25) + remainder at every layer.
    probe = {}
    for c in sorted(countries_pass):
        f = COUNTRIES[c]
        stacks = {L: [] for L in lm.layers}
        for cue in PROBE_CUES:
            r = lm.residuals(lm.encode(cue.format(capital=f["capital"], language=f["language"])))
            for L in lm.layers:
                stacks[L].append(r[L][-1])
        probe[c] = {L: torch.stack(v).mean(0) for L, v in stacks.items()}
    grand = {L: torch.stack([probe[c][L] for c in probe]).mean(0) for L in lm.layers}
    part = {}                                   # part[c][L] = dict(j, n, support)
    for c in probe:
        part[c] = {}
        for L in lm.layers:
            u = probe[c][L] - grand[L]
            probe[c][L] = u                     # store centered probe
            support, recon = jl.pursuit(u, lm.V(L), K_PURSUIT)
            part[c][L] = dict(j=recon, n=u - recon, support=support)
            if L in band:
                out["variance_fraction"].append(dict(country=c, layer=L,
                                                     frac=float(recon.norm() ** 2 / u.norm() ** 2)))

    def matched(d, ref):                         # rescale d to ||ref||
        return d * (ref.norm() / d.norm().clamp_min(1e-8))

    for it in items:
        A = it["intermediate"]
        if A not in probe:
            continue
        s = lm.tid(" " + A)
        for p in it["partners"]:
            B = p["intermediate"]
            if B not in probe or rank_out(it["lg"], p["answer"]) < 10:
                continue
            t = lm.tid(" " + B)
            full = {L: probe[B][L] - probe[A][L] for L in band}
            jdel = {L: matched(part[B][L]["j"] - part[A][L]["j"], full[L]) for L in band}
            ndel = {L: matched(part[B][L]["n"] - part[A][L]["n"], full[L]) for L in band}
            clamp_dirs = {L: torch.stack([lm.v(L, i) for i in sorted(
                {s, t} | set(part[A][L]["support"]) | set(part[B][L]["support"]))]) for L in lm.layers}
            conds = {
                "raw_lens": coord_swap_edits(lm, it["ids"], s, t, band, clean=it["clean"]),
                "full": delta_edits(full),
                "jpart": delta_edits(jdel),
                "nonj": delta_edits(ndel),
                "nonj_clamp": delta_edits(ndel) + jl.clamp_edits(lm, it["ids"], clamp_dirs, clean=it["clean"]),
            }
            rec = dict(family=it["family"], intermediate=A, target=B, swap_answer=p["answer"])
            for name, edits in conds.items():
                lg = lm.logits(it["ids"], edits)[-1]
                rec[name] = top1(lg) in W(p["answer"])
            out["privilege"].append(rec)

    # ------------------------------------------------------------------ floor check
    # the paper's own two-hop prompts, verbatim, no frame.
    paper = json.load(open(REF_PROBE_SWAP))["items"]
    fl = dict(n=len(paper), cap=0, single=0, swap_n=0, swap_hit=0, cap_items=[])
    for it in paper:
        ids = lm.encode(it["prompt"].rstrip())
        lg = lm.logits(ids)[-1]
        cap = top1(lg) in W(it["answer"])
        allsingle = all(lm.is_single(" " + it[k]) for k in ("intermediate", "answer", "swap_to", "swap_answer"))
        fl["cap"] += cap
        fl["single"] += allsingle
        if cap and allsingle:
            fl["cap_items"].append(it["name"])
            if rank_out(lg, it["swap_answer"]) >= 10:
                s, t = lm.tid(" " + it["intermediate"]), lm.tid(" " + it["swap_to"])
                sw = lm.logits(ids, coord_swap_edits(lm, ids, s, t, band))[-1]
                fl["swap_n"] += 1
                fl["swap_hit"] += top1(sw) in W(it["swap_answer"])
    out["floor"] = fl

    # ------------------------------------------------------------------ write
    R = jl.results_dir("c3_reasoning")
    json.dump({k: v for k, v in out.items() if k != "gate"} | dict(gate=out["gate"]),
              open(R / "results.json", "w"), indent=1, default=float)
    json.dump([dict(family=it["family"], intermediate=it["intermediate"],
                    answer=it["answer"], prompt=it["text"]) for it in items],
              open(R / "prompts.json", "w"), indent=1)
    summary = summarize(out, lm)
    open(R / "summary.txt", "w").write(summary)
    print(summary)


def summarize(out, lm) -> str:
    band = out["band"]
    L = ["Criterion 3 — internal reasoning — gpt2-small",
         f"\ncapability gate: {out['n_pass']}/{out['n_total']} two-hop items answered correctly"]

    # E1
    L.append("\nE1 readout — median lens rank at the answer position, per layer (n=%d)" % len(out["readout"]))
    L.append("     intermediate  answer     arg    null   | %% intermediate in top-10")
    for l in lm.layers:
        med = lambda k: median([r["ranks"][k][str(l)] if str(l) in r["ranks"][k] else r["ranks"][k][l]
                                for r in out["readout"]])
        pct = 100 * sum((r["ranks"]["intermediate"].get(str(l), r["ranks"]["intermediate"].get(l))) < 10
                        for r in out["readout"]) / len(out["readout"])
        tag = "  <- band" if l in band else ""
        L.append(f"  L{l:<2d} {med('intermediate'):11.0f} {med('answer'):7.0f} {med('arg'):7.0f} {med('null'):7.0f}   {pct:5.0f}%{tag}")
    uns = [r for r in out["readout"] if r["int_out_rank"] >= 10]
    def bandmin(r):
        return min(r["ranks"]["intermediate"].get(str(l), r["ranks"]["intermediate"].get(l)) for l in band)
    L.append(f"  unspoken items (intermediate output-rank>=10): {len(uns)}/{len(out['readout'])}; "
             f"of these, in lens top-10 at some band layer: {sum(bandmin(r) < 10 for r in uns)}/{len(uns)}")

    # E2
    if out["case"]:
        c = out["case"]
        L.append(f"\nE2 case study — {c['source']}->{c['target']} ({c['source_answer']}->{c['target_answer']})")
        L.append(f"  clean   top-5: {[w for w, _ in c['clean_top5']]}")
        L.append(f"  swapped top-5: {[w for w, _ in c['swapped_top5']]}")

    # E3
    def rate(rows, flt=lambda r: True):
        s = [r for r in rows if flt(r)]
        k, n = sum(r["hit"] for r in s), len(s)
        p, lo, hi = wilson(k, n)
        return f"{p:5.1%} [{lo:3.0%},{hi:3.0%}] (n={n}), median rank {median([r['before'] for r in s]):.0f}->{median([r['after'] for r in s]):.0f}"
    L.append("\nE3 swap — coordinate swap; swap_answer reaches output top-1 (targets starting rank>=10)")
    L.append(f"  all families    {rate(out['swap'])}")
    for fam in sorted({r["family"] for r in out["swap"]}):
        L.append(f"  {fam:15s} {rate(out['swap'], lambda r, f=fam: r['family'] == f)}")

    # E4
    def onset(eff):
        eff = {int(k): v for k, v in eff.items()}
        mx = max(eff.values())
        if mx <= 0:
            return None
        for l in sorted(eff):
            if eff[l] >= 0.5 * mx:
                return l
        return None
    oi = [onset(r["effect"]["interm"]) for r in out["depth"]]
    oa = [onset(r["effect"]["answer"]) for r in out["depth"]]
    pair = [(a, b) for a, b in zip(oi, oa) if a is not None and b is not None]
    L.append(f"\nE4 depth — onset layer (half-max) of the swap effect, intermediate vs answer (n={len(pair)})")
    if pair:
        di = median([a for a, _ in pair]); da = median([b for _, b in pair])
        L.append(f"  median onset: intermediate L{di:.0f}   answer L{da:.0f}   "
                 f"(intermediate earlier by {da - di:.1f} layers = {100 * (da - di) / lm.n_layers:.0f}% of depth)")
        L.append("  mean effect (log-prob pushed onto swap_answer) by layer:")
        for l in DEPTH_LAYERS:
            mi = sum(r["effect"]["interm"][str(l)] if str(l) in r["effect"]["interm"] else r["effect"]["interm"][l]
                     for r in out["depth"]) / len(out["depth"])
            ma = sum(r["effect"]["answer"][str(l)] if str(l) in r["effect"]["answer"] else r["effect"]["answer"][l]
                     for r in out["depth"]) / len(out["depth"])
            L.append(f"    L{l:<2d} intermediate {mi:+6.2f}   answer {ma:+6.2f}")

    # E5
    L.append("\nE5 privileging — swap_answer top-1, per condition (matched magnitude; n=%d)" % len(out["privilege"]))
    for cond in ("raw_lens", "full", "jpart", "nonj", "nonj_clamp"):
        k = sum(r[cond] for r in out["privilege"]); n = len(out["privilege"])
        p, lo, hi = wilson(k, n)
        L.append(f"  {cond:11s} {p:5.1%} [{lo:3.0%},{hi:3.0%}]  ({k}/{n})")
    vf = out["variance_fraction"]
    L.append("  J-space share of probe variance (median): " +
             ", ".join(f"L{l} {median([x['frac'] for x in vf if x['layer'] == l]):.0%}" for l in band))

    # floor
    fl = out["floor"]
    L.append(f"\nfloor — paper's probe-swap.json verbatim (no frame): capability {fl['cap']}/{fl['n']}; "
             f"coordinate swap on usable items {fl['swap_hit']}/{fl['swap_n']}")
    return "\n".join(L)


# =============================================================================== swap operations
@torch.no_grad()
def swap_ops():
    """E3 under BOTH swap operations, on the items and partner rule of `main` (target answer
    starting at output rank >= 10).  The criterion reports the coordinate swap because §3.3 names
    it and it is the more conservative of the two; this is where that comparison comes from.
    Writes results/c3_reasoning/swap_ops.json.  On gpt2-small: coordinate swap 705/999 = 70.6%,
    subtract-and-add 854/999 = 85.5%."""
    lm = jl.Lensed()
    band = jl.BAND
    R = jl.results_dir("c3_reasoning")
    items = json.load(open(R / "prompts.json"))
    for it in items:
        it["ids"] = lm.encode(it["prompt"].rstrip())
        it["clean"] = lm.residuals(it["ids"], band)
        it["lg"] = lm.logits(it["ids"])[-1]
        it["s"] = lm.tid(" " + it["intermediate"])
        it["a"] = lm.tid(" " + it["answer"])
    res = {"subadd": [], "coord": []}
    t0, n = time.time(), 0
    for it in items:
        for p in items:
            if (p["family"] != it["family"] or p["intermediate"] == it["intermediate"]
                    or p["answer"] == it["answer"]):
                continue
            if int(jl.ranks_of(it["lg"], [p["a"]])[0]) < 10:
                continue
            for name, mk in (("subadd", swap_edits), ("coord", coord_swap_edits)):
                lg = lm.logits(it["ids"], mk(lm, it["ids"], it["s"], p["s"], band, clean=it["clean"]))[-1]
                res[name].append((it["family"], int(lg.argmax()) == p["a"]))
            n += 1
    print("trials", n, "time", round(time.time() - t0))
    json.dump(res, open(R / "swap_ops.json", "w"))
    for name in res:
        for fam in sorted({f for f, _ in res["subadd"]}) + ["all"]:
            r = [h for f, h in res[name] if fam == "all" or f == fam]
            print(f"{name:7s} {fam:14s} {sum(r)}/{len(r)} = {100 * sum(r) / len(r):.1f}%")


if __name__ == "__main__":
    swap_ops() if "swap_ops" in sys.argv[1:] else main()
