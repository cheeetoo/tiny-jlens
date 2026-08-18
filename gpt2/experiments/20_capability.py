"""Capability survey: which task items can this GPT-2 actually do?

Applied before and independently of any lens measurement (the capability
filter of BRIEF §3). For two-hop items we also grade the two halves
separately (first hop: cue -> intermediate; one hop: intermediate -> answer)
to see where capability walls are.

Run:  python experiments/20_capability.py gpt2 | gpt2-medium | gpt2-large
"""

import json
import sys

sys.path.insert(0, "/tiny-jlens/gpt2")

import torch

import core
import pools

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
kit = core.Kit(MODEL, need_lens=False)
tok = kit.tokenizer

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
def graded_top1(prompt: str, max_ext: int = 3):
    """Greedy top-1 token, extending through article tokens. Returns
    (token_id, logits_at_answer_position)."""
    ids = kit.encode(prompt)
    for _ in range(max_ext):
        lg = kit.model_logits(ids)[-1]
        t = int(lg.argmax())
        if tok.decode([t]).strip().lower() in ARTICLES:
            ids = torch.cat([ids, torch.tensor([[t]], device=ids.device)], dim=1)
            continue
        return t, lg
    return t, lg


def correct(prompt: str, answer: str):
    t, lg = graded_top1(prompt)
    ok = t in variant_ids(answer)
    ans_rank = min((int((lg > lg[v]).sum()) for v in variant_ids(answer)), default=99999)
    return ok, ans_rank, tok.decode([t])


results = {"model": MODEL}

# ---------------- two-hop ----------------
items = pools.twohop_items(tok)
rows = []
for it in items:
    ok2, rank2, got2 = correct(it.prompt, it.answer)
    ok1 = rank1 = okf = None
    if it.onehop:
        ok1, rank1, _ = correct(it.onehop, it.answer)
    if it.firsthop:
        okf, _, _ = correct(it.firsthop, it.intermediate)
    rows.append(dict(family=it.family, arg=it.arg, intermediate=it.intermediate,
                     answer=it.answer, prompt=it.prompt, twohop=ok2,
                     answer_rank=rank2, got=got2, onehop=ok1, firsthop=okf,
                     source=it.source))
results["twohop"] = rows
by_fam: dict[str, list] = {}
for r in rows:
    by_fam.setdefault(r["family"], []).append(r)
print(f"\n== two-hop ({MODEL}) ==")
for fam, rs in sorted(by_fam.items()):
    n2 = sum(r["twohop"] for r in rs)
    n1 = sum(bool(r["onehop"]) for r in rs)
    nf = sum(bool(r["firsthop"]) for r in rs)
    have1 = sum(r["onehop"] is not None for r in rs)
    print(f"  {fam:18s} twohop {n2:2d}/{len(rs):2d}   onehop {n1:2d}/{have1:2d}   firsthop {nf:2d}/{have1 if fam!='riddle' else 0:2d}")
passing = [r for r in rows if r["twohop"]]
print(f"  TOTAL two-hop pass: {len(passing)}/{len(rows)}")

# ---------------- category report ----------------
cats = pools.report_categories(tok, with_additions=True)
rep = {}
for cat, members in cats.items():
    t, lg = graded_top1(pools.REPORT_FEWSHOT.format(cat=cat))
    mem_ids = {v for m in members for v in variant_ids(m)}
    valid = t in mem_ids
    top5 = any(int((lg > lg[v]).sum()) < 5 for v in mem_ids)
    rep[cat] = dict(valid=valid, member_top5=top5, got=tok.decode([t]))
results["report"] = rep
ok_cats = [c for c, r in rep.items() if r["valid"]]
t5_cats = [c for c, r in rep.items() if r["member_top5"]]
print(f"\n== category report ==  top1-valid {len(ok_cats)}/{len(rep)}: {ok_cats}")
print(f"   member-in-top5 {len(t5_cats)}/{len(rep)}: {t5_cats}")
for c, r in rep.items():
    if not r["valid"]:
        print(f"   miss {c:16s} -> {r['got']!r}")

# ---------------- C4 grid ----------------
grid = pools.c4_grid(tok)
c4 = []
print("\n== C4 grid (cells need >=3 correct args) ==")
for catname, fname, tmpl, answers in grid:
    cell = {}
    for arg, ans in answers.items():
        ok, rank, got = correct(tmpl.format(arg=arg), ans)
        cell[arg] = dict(ok=ok, got=got)
    n_ok = sum(v["ok"] for v in cell.values())
    c4.append(dict(category=catname, func=fname, template=tmpl,
                   n_ok=n_ok, n=len(cell), cell=cell))
    print(f"  {catname:12s} {fname:16s} {n_ok}/{len(cell)}")
results["c4"] = c4

# ---------------- C5a language tasks ----------------
classify_language = pools.classify_language
passages = pools.c5_passages() + pools.EXTRA_PASSAGES
c5 = []
print("\n== C5a language tasks ==")
for p in passages:
    ids = kit.encode(p["text"])
    cont = core.generate_with(kit, ids, [], max_new_tokens=16)
    cont_lang = classify_language(cont)
    t, _ = graded_top1(pools.LANG_REPORT_FEWSHOT.format(passage=p["text"]))
    rep_ok = t in variant_ids(p["category"])
    country = {"French": "France", "German": "Germany", "Spanish": "Spain", "Italian": "Italy"}[p["category"]]
    t2, _ = graded_top1(pools.LANG_COUNTRY_FEWSHOT.format(passage=p["text"]))
    ctry_ok = t2 in variant_ids(country)
    c5.append(dict(key=p["key"], category=p["category"], cont_lang=cont_lang,
                   cont_ok=cont_lang == p["category"], report_ok=rep_ok,
                   report_got=tok.decode([t]), country_ok=ctry_ok,
                   country_got=tok.decode([t2]), cont=cont))
    print(f"  {p['key']:4s} {p['category']:8s} cont->{cont_lang:8s} "
          f"report {'Y' if rep_ok else 'n'}({tok.decode([t]).strip()!r:12s}) "
          f"country {'Y' if ctry_ok else 'n'}({tok.decode([t2]).strip()!r})")
results["c5a"] = c5

with open(f"/tiny-jlens/gpt2/results/capability_{MODEL}{pools.SUFFIX}.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nsaved results/capability_{MODEL}.json")
