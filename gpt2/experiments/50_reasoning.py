"""C3 — internal reasoning: unspoken intermediates in the lens, and
intermediate swaps that redirect the answer.

Phases:
  readout  per-layer lens rank of the unspoken intermediate at the answer
           position, on capability-passing two-hop items. Controls: a
           matched random country (null), the surface arg (echo), and the
           answer itself (motor emergence).
  swap     clamped swaps of the intermediate's lens coordinates across layer
           windows x {coordinate, projection} x {raw, centered} x alpha.
           Success = the counterfactual answer becomes the graded top-1.

Run:  python experiments/50_reasoning.py readout|swap [gpt2|gpt2-medium|gpt2-large]
"""

import json
import random
import sys

sys.path.insert(0, "/tiny-jlens/gpt2")

import torch

import core
import pools

PHASE = sys.argv[1] if len(sys.argv) > 1 else "readout"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "gpt2"
kit = core.Kit(MODEL)
tok = kit.tokenizer
rng = random.Random(1234)

ARTICLES = {"a", "an", "the", "called", '"', "'", ""}


def variant_ids(word: str) -> set[int]:
    out = set()
    for w in {word, word.capitalize(), word.lower(), word.upper()}:
        for form in (" " + w, w):
            ids = tok(form, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                out.add(ids[0])
    return out


@torch.no_grad()
def graded_top1(ids: torch.Tensor, edits=()):
    """Greedy top-1 with article extension, under optional edits."""
    edits = list(edits)
    for _ in range(3):
        lg = core.logits_with(kit, ids, edits)[-1]
        t = int(lg.argmax())
        if tok.decode([t]).strip().lower() in ARTICLES:
            ids = torch.cat([ids, torch.tensor([[t]], device=ids.device)], dim=1)
            continue
        return t, lg
    return t, lg


# ---------------- items: capability-passing, with swap partners ----------------

cap = json.load(open(f"/tiny-jlens/gpt2/results/capability_{MODEL}.json"))
passed = {r["prompt"] for r in cap["twohop"] if r["twohop"]}
passed_by_fam: dict[str, set] = {}
for r in cap["twohop"]:
    if r["twohop"]:
        passed_by_fam.setdefault(r["family"], set()).add(r["intermediate"])

items = []
for it in pools.twohop_items(tok):
    if it.prompt not in passed:
        continue
    # swap partner: a capability-passing country in the same family whose
    # answer differs (first token) from this item's; paper items keep theirs
    partners = it.swap_pool
    if it.source == "ours":
        ok_ints = passed_by_fam.get(it.family, set())
        partners = [(c, a) for c, a in partners if c in ok_ints]
    partners = [(c, a) for c, a in partners
                if not (variant_ids(a) & variant_ids(it.answer))
                and not (variant_ids(c) & variant_ids(it.intermediate))]
    if not partners:
        continue
    it.swap_pool = [rng.choice(partners)]
    items.append(it)
print(f"{len(items)} items after capability filter + partner assignment "
      f"({sum(1 for i in items if i.source == 'paper')} from the paper's sets)")


def window(lo_frac: float, hi_frac: float) -> list[int]:
    n = kit.n_layers
    return [l for l in kit.layers if lo_frac * n <= l <= hi_frac * n]


# ---------------- readout ----------------

TABLE = pools.COUNTRIES

if PHASE == "readout":
    all_countries = list(TABLE)
    rows = []
    for it in items:
        ids = kit.encode(it.prompt)
        resid = kit.residuals(ids)
        out_lg = kit.model_logits(ids)[-1]
        out_rank = min(int((out_lg > out_lg[v]).sum())
                       for v in variant_ids(it.intermediate))
        null = rng.choice([c for c in all_countries
                           if c not in (it.intermediate, it.arg) and variant_ids(c)])
        probe_words = dict(intermediate=it.intermediate, answer=it.answer,
                           arg=it.arg, null=null)
        ranks = {}
        for name, word in probe_words.items():
            vids = list(variant_ids(word))
            if not vids:
                ranks[name] = None
                continue
            per_layer = {}
            for l in kit.layers:
                r = kit.ranks(resid[l][-1:], l, vids)[0]
                per_layer[l] = int(r.min())
            ranks[name] = per_layer
        # unspoken = the intermediate is NOT among the model's likely outputs
        # at the readout position (paper: intermediates are neither input nor
        # imminent output). Without this flag a lens hit could be motor preview.
        rows.append(dict(family=it.family, prompt=it.prompt, arg=it.arg,
                         intermediate=it.intermediate, answer=it.answer,
                         source=it.source, ranks=ranks,
                         int_output_rank=out_rank, unspoken=out_rank >= 10))

    print(f"\nper-layer: median rank of each probe word at the answer position "
          f"(n={len(rows)}), and % items with intermediate in top-10")
    print(f"{'L':>3} {'interm':>7} {'top10%':>7} {'answer':>7} {'arg':>7} {'null':>7}")
    for l in kit.layers:
        med = lambda name: int(torch.tensor(
            [r["ranks"][name][l] for r in rows if r["ranks"][name]]).float().median())
        pct = 100 * sum(r["ranks"]["intermediate"][l] < 10 for r in rows) / len(rows)
        print(f"{l:>3} {med('intermediate'):>7} {pct:>6.0f}% {med('answer'):>7} "
              f"{med('arg'):>7} {med('null'):>7}")
    best = [min(r["ranks"]["intermediate"].values()) for r in rows]
    print(f"\nintermediate in top-10 at SOME layer: "
          f"{sum(b < 10 for b in best)}/{len(rows)}")
    uns = [r for r in rows if r["unspoken"]]
    bu = [min(r["ranks"]["intermediate"].values()) for r in uns]
    print(f"UNSPOKEN items (intermediate not in output top-10): {len(uns)}/{len(rows)}; "
          f"of these, in lens top-10 at some layer: {sum(b < 10 for b in bu)}/{len(uns)}")
    json.dump(rows, open(f"/tiny-jlens/gpt2/results/c3_readout_{MODEL}.json", "w"))

# ---------------- probe (privilege) ----------------
# Independently-derived intermediate probes (mean residual over cue prompts
# implying the country, centered over countries), split into a J-space
# component (centered-gauge gradient pursuit, k=16) and the non-J remainder.
# Swap along each part at MATCHED NORM; the J part should carry the effect,
# and the remainder's effect should die when the J coordinates are clamped.

if PHASE == "probe":
    # Two probe constructions (audit 2026-08-19):
    #   naming   — cues end right before the country NAME would be said. This
    #              loads the probe on the country's output direction (high
    #              J-share) and favors J-mediation; kept for continuity with
    #              earlier receipts. GP k=16 (repo historical).
    #   implying — paper-faithful (§3.3): cues imply the country through
    #              varied surface cues and ask about different ATTRIBUTES
    #              (language/capital/continent); the country name is never
    #              the next token. GP k=25 (the paper's sparsity).
    # Each runs at matched norm (perturbation rescaled to the full-probe
    # delta, the paper's §3.1 convention) AND at natural magnitudes
    # (components as-is — matched norm inflates the J component, which
    # naturally carries only ~sqrt(J-share) of the norm).
    CUE_SETS = {
        "naming": (16, [
            "{capital} is the capital of a country called",
            "The {language} language is spoken mainly in the country called",
            "{language} is the official language of",
            "People who live in {capital} are citizens of",
            "The city of {capital} is the seat of government of",
            "A tourist who wants to hear {language} spoken everywhere should visit",
        ]),
        "implying": (25, [
            "People in the country whose capital is {capital} usually greet each other in",
            "A traveler flying into {capital} has landed on the continent of",
            "The homeland of the {language} language lies on the continent of",
            "The national anthem sung in {capital} is written in the language called",
            "Most newspapers printed in {capital} are written in",
            "The {language}-speaking country has its capital at",
        ]),
    }
    lays = window(0.55, 1.0)
    countries = sorted({it.intermediate for it in items if it.source == "ours"})

    probes_by, splits_by = {}, {}
    for cs_name, (K, cues) in CUE_SETS.items():
        resid_by_c = {}
        for c in countries:
            f = TABLE[c]
            hs = {l: [] for l in lays}
            for cue in cues:
                ids = kit.encode(cue.format(capital=f["capital"], language=f["language"]))
                r = kit.residuals(ids, lays)
                for l in lays:
                    hs[l].append(r[l][-1])
            resid_by_c[c] = {l: torch.stack(v).mean(0) for l, v in hs.items()}
        grand = {l: torch.stack([resid_by_c[c][l] for c in countries]).mean(0) for l in lays}
        probes = {c: {l: resid_by_c[c][l] - grand[l] for l in lays} for c in countries}

        def split(c, l, K=K, probes=probes):
            p = probes[c][l]
            ids_sel, _, recon = core.gradient_pursuit(kit, p, l, K, centered=True)
            return recon, p - recon, ids_sel.tolist()

        splits = {c: {l: split(c, l) for l in lays} for c in countries}
        var_share = torch.tensor([
            (splits[c][l][0].norm() ** 2 / probes[c][l].norm() ** 2).item()
            for c in countries for l in lays])
        print(f"[{cs_name}] probe J-component variance share (centered GP k={K}): "
              f"median {var_share.median():.1%}, IQR "
              f"[{var_share.quantile(0.25):.1%}, {var_share.quantile(0.75):.1%}]",
              flush=True)
        probes_by[cs_name], splits_by[cs_name] = probes, splits

    VARIANTS = [("naming", "matched"), ("naming", "natural"),
                ("implying", "matched"), ("implying", "natural")]
    rows = []
    for it in items:
        if it.source != "ours" or it.intermediate not in countries:
            continue
        A = it.intermediate
        B = it.swap_pool[0][0]
        tgt_ans = it.swap_pool[0][1]
        if B not in countries:
            continue
        ids = kit.encode(it.prompt)
        conds_all = {}
        for cs_name, scale in VARIANTS:
            probes, splits = probes_by[cs_name], splits_by[cs_name]
            conds = {}
            for cond in ("full", "J", "nonJ", "nonJ_clamped"):
                edits = []
                for l in lays:
                    full_d = probes[B][l] - probes[A][l]
                    jA, nA, selA = splits[A][l]
                    jB, nB, selB = splits[B][l]
                    d = dict(full=full_d, J=jB - jA, nonJ=nB - nA,
                             nonJ_clamped=nB - nA)[cond]
                    if scale == "matched":
                        d = d * (full_d.norm() / d.norm().clamp_min(1e-8))
                    edits.append(core.add_delta(d, l, None))
                if cond == "nonJ_clamped":
                    clamp_toks = sorted(set(selA) | set(selB)
                                        | {kit.tok_id(" " + A), kit.tok_id(" " + B)})
                    edits += core.clamp_coords(kit, ids, lays, clamp_toks, centered=True)
                t, lg = graded_top1(ids, edits)
                conds[cond] = dict(hit=t in variant_ids(tgt_ans), got=tok.decode([t]))
            conds_all[f"{cs_name}_{scale}"] = conds
        rows.append(dict(family=it.family, A=A, B=B, want=tgt_ans, conds=conds_all))
        print(".", end="", flush=True)
    print(f"\nn={len(rows)} items; swap along probe components "
          f"(rows = cue set x scaling):")
    for cs_name, scale in VARIANTS:
        key = f"{cs_name}_{scale}"
        line = f"  {key:18s}"
        for cond in ("full", "J", "nonJ", "nonJ_clamped"):
            k = sum(r["conds"][key][cond]["hit"] for r in rows)
            line += f"  {cond} {k:3d}/{len(rows)}"
        print(line)
    json.dump(rows, open(f"/tiny-jlens/gpt2/results/c3_probe_{MODEL}.json", "w"))

# ---------------- crossfn (anti-smuggling) ----------------
# One identical intermediate swap (country A -> B) applied under two
# different questions must flip each answer to ITS correct counterfactual.
# A vector that merely smuggles one answer cannot flip both.
#
# Question prompts are two-shot analogy templates with per-item DYNAMIC
# shots: the shot countries are chosen disjoint from the item's countries,
# args, and answers, so no answer can be echoed from the prompt. (Static
# shots would leak: with only 4-6 continents/languages in play, a fixed
# shot's answer collides with some item's.) Continent families need >=4
# continents in the shot pool — two shots plus the item's two continents
# exhaust a 3-continent pool.

if PHASE == "crossfn":
    import itertools

    CROSSFN_TEMPLATES = {
        "lang_capital": (
            "The capital of the country where {shot_arg} is spoken is the city "
            "of {shot_ans}. The capital of the country where {shot2_arg} is "
            "spoken is the city of {shot2_ans}. The capital of the country "
            "where {arg} is spoken is the city of"),
        "lang_continent": (
            "The country where {shot_arg} is spoken is a country on the "
            "continent of {shot_ans}. The country where {shot2_arg} is spoken "
            "is a country on the continent of {shot2_ans}. The country where "
            "{arg} is spoken is a country on the continent of"),
        "city_language": (
            "In the country whose capital city is {shot_arg}, the primary "
            "language is {shot_ans}. In the country whose capital city is "
            "{shot2_arg}, the primary language is {shot2_ans}. In the country "
            "whose capital city is {arg}, the primary language is"),
        "city_continent": (
            "The country whose capital is {shot_arg} is located on the "
            "continent of {shot_ans}. The country whose capital is {shot2_arg} "
            "is located on the continent of {shot2_ans}. The country whose "
            "capital is {arg} is located on the continent of"),
    }
    # preference-ordered shot countries (Nairobi as a capital-answer shot
    # primed ' N' completions, so capital-answer families lead elsewhere)
    SHOT_POOL_BY_FAM = {
        "lang_capital": ["Portugal", "Egypt", "Thailand", "Finland", "Kenya"],
        "city_language": ["Kenya", "Portugal", "Thailand", "Egypt", "Finland"],
        "lang_continent": ["Kenya", "Thailand", "Cuba", "Portugal", "Egypt", "Finland"],
        "city_continent": ["Kenya", "Thailand", "Cuba", "Portugal", "Egypt", "Finland"],
    }
    ARGF = {"lang_capital": "language", "lang_continent": "language",
            "city_language": "capital", "city_continent": "capital"}
    ANSF = {"lang_capital": "capital", "lang_continent": "continent",
            "city_language": "language", "city_continent": "continent"}

    BASE = {}
    for fam in CROSSFN_TEMPLATES:
        BASE[fam] = {}
        for c, f in pools.COUNTRIES.items():
            if ARGF[fam] == "language" and not f["language_unique"]:
                continue
            if not (variant_ids(c) and variant_ids(f[ANSF[fam]])):
                continue
            BASE[fam][c] = dict(arg=f[ARGF[fam]], answer=f[ANSF[fam]])

    def pick_shot(fam, avoid_countries, avoid_answers):
        for c in SHOT_POOL_BY_FAM[fam]:
            row = pools.COUNTRIES[c]
            s_arg, s_ans = row[ARGF[fam]], row[ANSF[fam]]
            if c in avoid_countries:
                continue
            if any(variant_ids(s_ans) & variant_ids(a) for a in avoid_answers):
                continue
            if any(variant_ids(s_arg) & variant_ids(pools.COUNTRIES[c2][ARGF[fam]])
                   for c2 in avoid_countries if c2 in pools.COUNTRIES):
                continue
            return s_arg, s_ans
        return None

    def build_prompt(fam, country, partner=None):
        avoid_c = {country} | ({partner} if partner else set())
        avoid_a = {BASE[fam][c]["answer"] for c in avoid_c if c in BASE[fam]}
        s1 = pick_shot(fam, avoid_c, avoid_a)
        if s1 is None:
            return None
        s2 = pick_shot(fam, avoid_c | {c for c in SHOT_POOL_BY_FAM[fam]
                                       if pools.COUNTRIES[c][ARGF[fam]] == s1[0]},
                       avoid_a | {s1[1]})
        if s2 is None:
            return None
        return CROSSFN_TEMPLATES[fam].format(
            shot_arg=s1[0], shot_ans=s1[1], shot2_arg=s2[0], shot2_ans=s2[1],
            arg=BASE[fam][country]["arg"])

    def capable(fam, country, partner=None):
        p = build_prompt(fam, country, partner)
        if p is None:
            return False
        t, _ = graded_top1(kit.encode(p))
        return t in variant_ids(BASE[fam][country]["answer"])

    lays = window(0.55, 1.0)
    rows = []
    for f1, f2 in itertools.combinations(CROSSFN_TEMPLATES, 2):
        pool1 = {c for c in BASE[f1] if capable(f1, c)}
        pool2 = {c for c in BASE[f2] if capable(f2, c)}
        both = sorted(pool1 & pool2)
        for A in both:
            partners = [B for B in both if B != A
                        and BASE[f1][B]["answer"] != BASE[f1][A]["answer"]
                        and BASE[f2][B]["answer"] != BASE[f2][A]["answer"]]
            if not partners:
                continue
            B = rng.choice(partners)
            src, tgt = [kit.tok_id(" " + A)], [kit.tok_id(" " + B)]
            flips, skip = {}, False
            for fam in (f1, f2):
                prompt = build_prompt(fam, A, B)
                if prompt is None:
                    skip = True
                    break
                ids = kit.encode(prompt)
                edits = core.swap_clamped(kit, ids, lays, src, tgt, alpha=1.0)
                t, lg = graded_top1(ids, edits)
                tgt_ans = BASE[fam][B]["answer"]
                flips[fam] = dict(hit=t in variant_ids(tgt_ans),
                                  got=tok.decode([t]), want=tgt_ans)
            if skip:
                continue
            rows.append(dict(pair=(f1, f2), A=A, B=B, flips=flips,
                             both=all(v["hit"] for v in flips.values()),
                             any=any(v["hit"] for v in flips.values())))
        sel = [r for r in rows if tuple(r["pair"]) == (f1, f2)]
        if sel:
            print(f"{f1} + {f2}: both-flip {sum(r['both'] for r in sel)}/{len(sel)}"
                  f"  any-flip {sum(r['any'] for r in sel)}/{len(sel)}")
    print(f"\nTOTAL both-flip {sum(r['both'] for r in rows)}/{len(rows)}  "
          f"any-flip {sum(r['any'] for r in rows)}/{len(rows)}")
    for r in rows:
        if r["any"] and not r["both"]:
            bad = {f: v for f, v in r["flips"].items() if not v["hit"]}
            print(f"  partial {r['A']}->{r['B']}: " +
                  "; ".join(f"{f} got {v['got']!r} want {v['want']}" for f, v in bad.items()))
    json.dump(rows, open(f"/tiny-jlens/gpt2/results/c3_crossfn_{MODEL}.json", "w"))

# ---------------- swap ----------------

if PHASE == "swap":
    WINDOWS = {
        "all": (0.0, 1.0), "mid+": (0.35, 1.0), "late": (0.55, 1.0),
        "later": (0.65, 1.0), "last3": (0.75, 1.0),
        # nomotor hi must exclude the top fitted layer: 0.87*12 = 10.44 kept
        # L10 and made this window identical to "late" (audit fix; 0.80*12
        # = 9.6 -> L7-9)
        "late-nomotor": (0.55, 0.80), "mid": (0.35, 0.75),
    }
    FLAVORS = [("coord", False), ("coord", True), ("proj", False), ("proj", True)]
    ALPHAS = [1.0, 2.0]

    rows = []
    for it in items:
        ids = kit.encode(it.prompt)
        tgt_int, tgt_ans = it.swap_pool[0]
        clean_lg = kit.model_logits(ids)[-1]
        tgt_rank_pre = min(int((clean_lg > clean_lg[v]).sum()) for v in variant_ids(tgt_ans))
        if tgt_rank_pre < 10:
            continue  # paper rule: target must not already be in the top-10
        src_ids = [kit.tok_id(" " + it.intermediate)]
        tgt_ids = [kit.tok_id(" " + tgt_int)]
        for wname, (lo, hi) in WINDOWS.items():
            lays = window(lo, hi)
            for form, centered in FLAVORS:
                for alpha in ALPHAS:
                    make = core.swap_clamped if form == "coord" else core.swap_projection
                    edits = make(kit, ids, lays, src_ids, tgt_ids,
                                 alpha=alpha, centered=centered)
                    t, lg = graded_top1(ids, edits)
                    hit = t in variant_ids(tgt_ans)
                    tgt_rank = min(int((lg > lg[v]).sum()) for v in variant_ids(tgt_ans))
                    rows.append(dict(
                        family=it.family, prompt=it.prompt, source=it.source,
                        intermediate=it.intermediate, target=tgt_int,
                        answer=it.answer, tgt_answer=tgt_ans, window=wname,
                        form=form, centered=centered, alpha=alpha, hit=hit,
                        tgt_rank=tgt_rank, got=tok.decode([t])))
        print(".", end="", flush=True)

    n_items = len({r["prompt"] for r in rows})
    print(f"\n\nswap success (target answer graded top-1), n={n_items} items")
    hdr = "window       " + "".join(f"{form[:1]}{'c' if c else 'r'}a{int(a)} "
                                    for form, c in FLAVORS for a in ALPHAS)
    print(hdr)
    for wname in WINDOWS:
        cells = []
        for form, c in FLAVORS:
            for a in ALPHAS:
                sel = [r for r in rows if r["window"] == wname and r["form"] == form
                       and r["centered"] == c and r["alpha"] == a]
                cells.append(f"{100 * sum(r['hit'] for r in sel) / max(len(sel), 1):>4.0f}")
        print(f"{wname:12s} " + " ".join(cells))
    json.dump(rows, open(f"/tiny-jlens/gpt2/results/c3_swap_{MODEL}.json", "w"))
