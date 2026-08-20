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
import statistics
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
# capability per few-shot format (format-frame diversity); back-compat with
# capability files that predate report_formats
fmt_cap = cap.get("report_formats", {"0": cap["report"]})
FORMATS = [(int(i), pools.REPORT_FEWSHOTS[int(i)]) for i in sorted(fmt_cap)]
usable = {int(i): {c for c, r in fmt_cap[i].items() if r["member_top5"]} for i in fmt_cap}
top1_valid = {int(i): {c for c, r in fmt_cap[i].items() if r["valid"]} for i in fmt_cap}


def spearman(a: list[float], b: list[float]) -> float:
    import scipy.stats

    return float(scipy.stats.spearmanr(a, b).statistic)


# ---------------- corr ----------------

if PHASE == "corr":
    rows = []
    for fi, fmt in FORMATS:
        for cat in sorted(usable[fi]):
            members = cats_all[cat]
            ids, lg = anchor_state(fmt.format(cat=cat))
            resid = kit.residuals(ids)
            mem_ids = [min(variant_ids(m), key=lambda v: int((lg > lg[v]).sum()))
                       for m in members]  # best surface form of each member
            out_ranks = [int((lg > lg[v]).sum()) for v in mem_ids]
            per_layer = {}
            for l in kit.layers:
                lr = kit.ranks(resid[l][-1:], l, mem_ids)[0].tolist()
                per_layer[l] = spearman([-r for r in out_ranks], [-r for r in lr])
            rows.append(dict(fmt=fi, cat=cat, per_layer=per_layer))
    print(f"{len(rows)} (format, category) cells "
          f"(usable = member in output top-5 under that format)")
    print("per-layer mean Spearman (lens ranks vs output ranks of members):")
    for l in kit.layers:
        vals = torch.tensor([r["per_layer"][l] for r in rows])
        print(f"  L{l:2d}  {vals.mean():+.3f}  (min {vals.min():+.2f}, max {vals.max():+.2f})")
    json.dump(rows, open(f"/tiny-jlens/gpt2/results/c1_corr_{MODEL}.json", "w"))

# ---------------- swap ----------------

if PHASE == "swap":
    WINDOW = sys.argv[3] if len(sys.argv) > 3 else "0.55:1.0"
    lo, hi = map(float, WINDOW.split(":"))
    lays = [l for l in kit.layers if lo * kit.n_layers <= l <= hi * kit.n_layers]
    print(f"swap layers: {lays}")
    rows = []
    for fi, fmt in FORMATS:
        for cat in sorted(top1_valid[fi]):
            members = cats_all[cat]
            ids, lg = anchor_state(fmt.format(cat=cat))
            src_tok = int(lg.argmax())
            src_name = next((m for m in members if src_tok in variant_ids(m)), None)
            if src_name is None:
                continue
            # eligible targets: members whose every variant is outside output top-10
            eligible = [m for m in members if m != src_name
                        and min(int((lg > lg[v]).sum()) for v in variant_ids(m)) >= 10]
            for tgt_name in rng.sample(eligible, min(8, len(eligible))):
                tgt_tok = min(variant_ids(tgt_name), key=lambda v: int((lg > lg[v]).sum()))
                for centered in (False, True):
                    for alpha in (1.0, 2.0):
                        edits = core.swap_clamped(kit, ids, lays, [src_tok], [tgt_tok],
                                                  alpha=alpha, centered=centered)
                        lg2 = core.logits_with(kit, ids, edits)[-1]
                        tr = min(int((lg2 > lg2[v]).sum()) for v in variant_ids(tgt_name))
                        rows.append(dict(fmt=fi, cat=cat, src=src_name, tgt=tgt_name,
                                         centered=centered, alpha=alpha,
                                         tgt_rank_post=tr,
                                         top1=tr == 0, top5=tr < 5))
            print(".", end="", flush=True)
    print()
    n_pairs = len({(r["fmt"], r["cat"], r["tgt"]) for r in rows})
    print(f"{n_pairs} (format, category, target) swaps")
    for centered in (False, True):
        for alpha in (1.0, 2.0):
            sel = [r for r in rows if r["centered"] == centered and r["alpha"] == alpha]
            if sel:
                print(f"  {'centered' if centered else 'raw     '} a={alpha}: "
                      f"top-1 {100*sum(r['top1'] for r in sel)/len(sel):.0f}%  "
                      f"top-5 {100*sum(r['top5'] for r in sel)/len(sel):.0f}%  "
                      f"median post rank {statistics.median(r['tgt_rank_post'] for r in sel)}")
    json.dump(rows, open(f"/tiny-jlens/gpt2/results/c1_swap_{MODEL}.json", "w"))

# ---------------- inject ----------------

if PHASE == "inject":
    # Frame diversity: multiple report-eliciting frames and multiple
    # noun-expecting control frames (pools.INJECT_*_FRAMES; index 0 = the
    # original pair). Same injection protocol for both kinds.
    CONCEPTS = pools.INJECT_CONCEPTS
    lays = [l for l in kit.layers if 0.55 * kit.n_layers <= l <= 1.0 * kit.n_layers]
    # the low end (0.15-0.25) is where report and control frames separate;
    # by 0.5 every noun-licensing frame blurts the injected concept
    STRENGTHS = (0.0, 0.15, 0.175, 0.25, 0.5, 1.0, 2.0)
    FRAME_SETS = [("report", pools.INJECT_REPORT_FRAMES),
                  ("control", pools.INJECT_CONTROL_FRAMES)]
    # inject at every position EXCEPT the last 3 (the readout anchor is never
    # steered directly — the report must be carried there by the model), as in
    # the paper's inject-on-the-user-turn protocol
    n_by_frame = {(kind, i): kit.encode(f).shape[1]
                  for kind, frames in FRAME_SETS for i, f in enumerate(frames)}
    rows = []
    for w in CONCEPTS:
        vids = variant_ids(w)
        if not vids:
            continue
        for strength in STRENGTHS:
            for kind, frames in FRAME_SETS:
                for i, frame in enumerate(frames):
                    n = n_by_frame[(kind, i)]
                    edits = [core.steer(kit, l, vids[0], strength,
                                        positions=list(range(n - 3)),
                                        centered=True) for l in lays] if strength else []
                    _, lg = anchor_state(frame, edits)
                    rank = min(int((lg > lg[v]).sum()) for v in vids)
                    rows.append(dict(word=w, strength=strength, kind=kind,
                                     frame=i, rank=rank))
        print(".", end="", flush=True)
    print()
    n_words = len({r["word"] for r in rows})
    n_rep = len(pools.INJECT_REPORT_FRAMES)
    n_ctl = len(pools.INJECT_CONTROL_FRAMES)
    print(f"{n_words} concepts x {n_rep} report frames + {n_ctl} controls; "
          f"steering centered lens vector at layers {lays} "
          f"(all positions except the last 3)")
    print("strength   report top-5 (pooled)   control top-5 (pooled)   per-report-frame top-5")
    for s in STRENGTHS:
        rep = [r for r in rows if r["strength"] == s and r["kind"] == "report"]
        ctl = [r for r in rows if r["strength"] == s and r["kind"] == "control"]
        per_frame = " ".join(
            f"{sum(r['rank'] < 5 for r in rep if r['frame'] == i):>3}/{n_words}"
            for i in range(n_rep))
        print(f"  {s:4.2f}     {sum(r['rank']<5 for r in rep):>4}/{len(rep)}"
              f"                 {sum(r['rank']<5 for r in ctl):>4}/{len(ctl)}"
              f"              {per_frame}")
    json.dump(rows, open(f"/tiny-jlens/gpt2/results/c1_inject_{MODEL}.json", "w"))
