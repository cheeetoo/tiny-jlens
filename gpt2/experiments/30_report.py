"""C1 — verbal report: the model's category report is predicted by lens
content (corr), caused by it (swap), and steering a concept in makes it the
report (inject).

Phases:
  corr    Spearman correlation, per layer, between lens ranks and output
          ranks of the candidate members at the report anchor (the colon of
          the few-shot list; the position immediately before the report).
  swap    swap the spontaneously-chosen member's lens coordinates for another
          member's; success = target enters the graded top-5 (paper metric).
  inject  steer a concept's (centered) lens vector into an "I am thinking
          about" prompt; report rate vs strength, with a blurt control on an
          unrelated prompt at matched strength.

Run:  python experiments/30_report.py corr|swap|inject [model]
"""

import json
import random
import sys

sys.path.insert(0, "/tiny-jlens/gpt2")

import torch

import core
import pools

PHASE = sys.argv[1] if len(sys.argv) > 1 else "corr"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "gpt2"
kit = core.Kit(MODEL)
tok = kit.tokenizer
rng = random.Random(1234)

ARTICLES = {"a", "an", "the", "called", '"', "'", ""}


def variant_ids(word: str) -> list[int]:
    out = []
    for w in {word, word.capitalize(), word.lower(), word.upper()}:
        for form in (" " + w, w):
            ids = tok(form, add_special_tokens=False)["input_ids"]
            if len(ids) == 1 and ids[0] not in out:
                out.append(ids[0])
    return out


@torch.no_grad()
def anchor_state(prompt: str, edits=()):
    """Extend through articles; return (ids, final logits, final position)."""
    ids = kit.encode(prompt)
    for _ in range(3):
        lg = core.logits_with(kit, ids, list(edits))[-1]
        t = int(lg.argmax())
        if tok.decode([t]).strip().lower() in ARTICLES:
            ids = torch.cat([ids, torch.tensor([[t]], device=ids.device)], dim=1)
            continue
        break
    return ids, lg


cap = json.load(open(f"/tiny-jlens/gpt2/results/capability_{MODEL}.json"))
cats_all = pools.report_categories(tok)          # curated members (swap targets)
usable = {c for c, r in cap["report"].items() if r["member_top5"]}
top1_valid = {c for c, r in cap["report"].items() if r["valid"]}


def spearman(a: list[float], b: list[float]) -> float:
    import scipy.stats

    return float(scipy.stats.spearmanr(a, b).statistic)


# ---------------- corr ----------------

if PHASE == "corr":
    per_layer: dict[int, list[float]] = {l: [] for l in kit.layers}
    for cat in sorted(usable):
        members = cats_all[cat]
        ids, lg = anchor_state(pools.REPORT_FEWSHOT.format(cat=cat))
        resid = kit.residuals(ids)
        mem_ids = [min(variant_ids(m), key=lambda v: int((lg > lg[v]).sum()))
                   for m in members]  # best surface form of each member
        out_ranks = [int((lg > lg[v]).sum()) for v in mem_ids]
        for l in kit.layers:
            lr = kit.ranks(resid[l][-1:], l, mem_ids)[0].tolist()
            per_layer[l].append(spearman([-r for r in out_ranks], [-r for r in lr]))
    print(f"categories: {len(per_layer[kit.layers[0]])} "
          f"(usable = member in output top-5)")
    print("per-layer mean Spearman (lens ranks vs output ranks of members):")
    for l in kit.layers:
        vals = torch.tensor(per_layer[l])
        print(f"  L{l:2d}  {vals.mean():+.3f}  (min {vals.min():+.2f}, max {vals.max():+.2f})")
    json.dump({l: per_layer[l] for l in kit.layers},
              open(f"/tiny-jlens/gpt2/results/c1_corr_{MODEL}.json", "w"))

# ---------------- swap ----------------

if PHASE == "swap":
    WINDOW = sys.argv[3] if len(sys.argv) > 3 else "0.55:1.0"
    lo, hi = map(float, WINDOW.split(":"))
    lays = [l for l in kit.layers if lo * kit.n_layers <= l <= hi * kit.n_layers]
    print(f"swap layers: {lays}")
    rows = []
    for cat in sorted(top1_valid):
        members = cats_all[cat]
        ids, lg = anchor_state(pools.REPORT_FEWSHOT.format(cat=cat))
        src_tok = int(lg.argmax())
        src_name = next((m for m in members if src_tok in variant_ids(m)), None)
        if src_name is None:
            continue
        # eligible targets: members whose every variant is outside output top-10
        eligible = [m for m in members if m != src_name
                    and min(int((lg > lg[v]).sum()) for v in variant_ids(m)) >= 10]
        for tgt_name in rng.sample(eligible, min(5, len(eligible))):
            tgt_tok = min(variant_ids(tgt_name), key=lambda v: int((lg > lg[v]).sum()))
            for centered in (False, True):
                for alpha in (1.0, 2.0):
                    edits = core.swap_clamped(kit, ids, lays, [src_tok], [tgt_tok],
                                              alpha=alpha, centered=centered)
                    lg2 = core.logits_with(kit, ids, edits)[-1]
                    tr = min(int((lg2 > lg2[v]).sum()) for v in variant_ids(tgt_name))
                    rows.append(dict(cat=cat, src=src_name, tgt=tgt_name,
                                     centered=centered, alpha=alpha,
                                     tgt_rank_post=tr,
                                     top1=tr == 0, top5=tr < 5))
        print(".", end="", flush=True)
    print()
    n_pairs = len({(r["cat"], r["tgt"]) for r in rows})
    print(f"{n_pairs} (category, target) swaps")
    for centered in (False, True):
        for alpha in (1.0, 2.0):
            sel = [r for r in rows if r["centered"] == centered and r["alpha"] == alpha]
            if sel:
                print(f"  {'centered' if centered else 'raw     '} a={alpha}: "
                      f"top-1 {100*sum(r['top1'] for r in sel)/len(sel):.0f}%  "
                      f"top-5 {100*sum(r['top5'] for r in sel)/len(sel):.0f}%  "
                      f"median post rank {sorted(r['tgt_rank_post'] for r in sel)[len(sel)//2]}")
    json.dump(rows, open(f"/tiny-jlens/gpt2/results/c1_swap_{MODEL}.json", "w"))

# ---------------- inject ----------------

if PHASE == "inject":
    PROMPT = "The word that best describes what I am thinking about right now is"
    CONTROL = "The weather report for tomorrow morning said it would be"
    CONCEPTS = pools.INJECT_CONCEPTS
    lays = [l for l in kit.layers if 0.55 * kit.n_layers <= l <= 1.0 * kit.n_layers]
    STRENGTHS = (0.0, 0.25, 0.5, 1.0, 2.0)
    # inject at every position EXCEPT the last 3 (the readout anchor is never
    # steered directly — the report must be carried there by the model), as in
    # the paper's inject-on-the-user-turn protocol
    n_prompt = kit.encode(PROMPT).shape[1]
    n_ctrl = kit.encode(CONTROL).shape[1]
    rows = []
    for w in CONCEPTS:
        vids = variant_ids(w)
        if not vids:
            continue
        for strength in STRENGTHS:
            edits = [core.steer(kit, l, vids[0], strength,
                                positions=list(range(n_prompt - 3)),
                                centered=True) for l in lays] if strength else []
            _, lg = anchor_state(PROMPT, edits)
            rrank = min(int((lg > lg[v]).sum()) for v in vids)
            edits_c = [core.steer(kit, l, vids[0], strength,
                                  positions=list(range(n_ctrl - 3)),
                                  centered=True) for l in lays] if strength else []
            _, lgc = anchor_state(CONTROL, edits_c)
            crank = min(int((lgc > lgc[v]).sum()) for v in vids)
            rows.append(dict(word=w, strength=strength, report_rank=rrank,
                             control_rank=crank))
    print(f"{len({r['word'] for r in rows})} concepts; steering centered lens "
          f"vector at layers {lays} (all positions except the last 3)")
    print("strength   report top-1   report top-5   median rank   control top-5 (blurt)")
    for s in STRENGTHS:
        sel = [r for r in rows if r["strength"] == s]
        print(f"  {s:4.0f}    {sum(r['report_rank']==0 for r in sel):>6}/{len(sel)}"
              f"       {sum(r['report_rank']<5 for r in sel):>6}/{len(sel)}"
              f"        {sorted(r['report_rank'] for r in sel)[len(sel)//2]:>6}"
              f"       {sum(r['control_rank']<5 for r in sel):>4}/{len(sel)}")
    json.dump(rows, open(f"/tiny-jlens/gpt2/results/c1_inject_{MODEL}.json", "w"))
