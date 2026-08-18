"""C5b — whole-J-space ablation battery, centered gauge.

At every prompt position (band L7–L10), project out the span of the top-k
CENTERED lens vectors (readout ranking; the raw version removes the shared
gauge axis instead and has no dose — see the README gauge note), sparing the clean
top-10 output tokens per position (the paper's protection rule).

Battery (capability-filtered per model; shallow/flexible assignment fixed by
task analysis BEFORE any ablation was run):
  shallow:   wikitext next-token agreement with the clean model; copy;
             continuation-stays-in-language
  flexible:  two-hop (lang_capital), passage->country, category report

Controls: (a) projecting out k random directions per layer (same
dimensionality), (b) Gaussian noise addition matched to the true ablation's
removed norm.

Run:  python experiments/71_ablation.py [model] [k]
"""

import json
import sys

sys.path.insert(0, "/tiny-jlens/gpt2")

import datasets
import torch

import core
import pools

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 10
kit = core.Kit(MODEL)
tok = kit.tokenizer
if len(sys.argv) > 3:  # explicit band "lo:hi" (band-matched comparisons)
    _lo, _hi = map(int, sys.argv[3].split(":"))
    LAYS = [l for l in kit.layers if _lo <= l <= _hi]
else:
    LAYS = [l for l in kit.layers if 0.55 * kit.n_layers <= l <= 1.0 * kit.n_layers]
ARTICLES = {"a", "an", "the", "called", '"', "'", ""}
torch.manual_seed(0)


def variant_ids(word: str) -> list[int]:
    out = []
    for w in {word, word.capitalize(), word.lower(), word.upper()}:
        for form in (" " + w, w):
            ids = tok(form, add_special_tokens=False)["input_ids"]
            if len(ids) == 1 and ids[0] not in out:
                out.append(ids[0])
    return out


def protection(ids: torch.Tensor) -> dict[int, set[int]]:
    lg = kit.model_logits(ids)
    top = lg.topk(10, dim=-1).indices
    return {p: set(top[p].tolist()) for p in range(ids.shape[1])}


REMOVED = []  # (norm_removed, norm_h) pairs recorded by the true ablation


def ablate_edits(ids, mode: str) -> list:
    """mode: 'ablate' (centered top-k), 'randproj', 'noise'."""
    n = ids.shape[1]
    pos = list(range(n))
    if mode == "ablate":
        prot = protection(ids)
        edits = []
        for l in LAYS:
            Jl = kit.lens.jacobians[l]

            def fn(h, pp, l=l, Jl=Jl):
                out = h.clone()
                scores = kit.U_eff @ (Jl @ h.T)
                for j in range(h.shape[0]):
                    s = scores[:, j].clone()
                    p = int(pp[j])
                    if p in prot:
                        s[list(prot[p])] = -torch.inf
                    topt = s.topk(K).indices.tolist()
                    W = kit.vectors(l, topt, centered=True)
                    Q, _ = torch.linalg.qr(W.T)
                    removed = Q @ (Q.T @ h[j])
                    REMOVED.append((removed.norm().item(), h[j].norm().item()))
                    out[j] = h[j] - removed
                return out

            edits.append(core.Edit(l, fn, pos))
        return edits
    if mode == "randproj":
        edits = []
        for l in LAYS:
            R = torch.randn(K, kit.d_model, device="cuda")
            Q, _ = torch.linalg.qr(R.T)

            def fn(h, pp, Q=Q):
                return h - (h @ Q) @ Q.T

            edits.append(core.Edit(l, fn, pos))
        return edits
    if mode == "noise":
        frac = (sum(a for a, b in REMOVED) / sum(b for a, b in REMOVED)) if REMOVED else 0.15
        edits = []
        for l in LAYS:
            def fn(h, pp, frac=frac):
                z = torch.randn_like(h)
                z = z / z.norm(dim=1, keepdim=True) * (h.norm(dim=1, keepdim=True) * frac)
                return h + z

            edits.append(core.Edit(l, fn, pos))
        return edits
    return []


@torch.no_grad()
def graded_top1(ids, edits=()):
    edits = list(edits)
    for _ in range(3):
        lg = core.logits_with(kit, ids, edits)[-1]
        t = int(lg.argmax())
        if tok.decode([t]).strip().lower() in ARTICLES:
            ids = torch.cat([ids, torch.tensor([[t]], device=ids.device)], dim=1)
            continue
        return t
    return t


# ---------------- task definitions ----------------

cap = json.load(open(f"/tiny-jlens/gpt2/results/capability_{MODEL}.json"))
ds = datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
wiki = [r["text"].strip() for r in ds if len(r["text"].strip()) > 600][:6]

COPY_SENTS = pools.C2_SENTENCES + ["The train arrived at the station ten minutes late."]


def task_wikitext(mode):
    """top-1 agreement with the clean model over positions 16..N."""
    agree = n = 0
    for t in wiki:
        ids = kit.encode(t[:1200])[:, :96]
        clean = kit.model_logits(ids).argmax(dim=-1)
        lg = core.logits_with(kit, ids, ablate_edits(ids, mode)) if mode else None
        pred = lg.argmax(dim=-1) if mode else clean
        for p in range(16, ids.shape[1] - 1):
            agree += int(pred[p] == clean[p])
            n += 1
    return agree / n


def task_copy(mode):
    """teacher-forced accuracy on the copy span."""
    ok = n = 0
    for s in COPY_SENTS:
        text = f'Copy the sentence.\nSentence: "{s}"\nCopy: "{s}'
        ids = kit.encode(text)
        start = text.rindex(s)
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
        span = [i for i, (a, b) in enumerate(enc["offset_mapping"][0].tolist())
                if a >= start and b > a]
        lg = core.logits_with(kit, ids, ablate_edits(ids, mode))
        for i in span[:-1]:
            ok += int(lg[i - 1].argmax() == ids[0, i])
            n += 1
    return ok / n


def task_contlang(mode):
    passages = [p for p in pools.c5_passages() + pools.EXTRA_PASSAGES]
    ok = n = 0
    for p in passages:
        ids = kit.encode(p["text"])
        cont = core.generate_with(kit, ids, ablate_edits(ids, mode), max_new_tokens=24)
        ok += int(pools.classify_language(cont) == p["category"])
        n += 1
    return ok / n


def task_twohop(mode):
    items = [it for it in pools.twohop_items(tok)
             if it.family == "lang_capital"
             and it.prompt in {r["prompt"] for r in cap["twohop"] if r["twohop"]}]
    if not items:
        return float("nan")
    ok = 0
    for it in items:
        ids = kit.encode(it.prompt)
        t = graded_top1(ids, ablate_edits(ids, mode))
        ok += int(t in variant_ids(it.answer))
    return ok / len(items)


def task_country(mode):
    COUNTRY = {"French": "France", "German": "Germany", "Spanish": "Spain", "Italian": "Italy"}
    ok = n = 0
    for p in pools.c5_passages() + pools.EXTRA_PASSAGES:
        text = pools.LANG_COUNTRY_FEWSHOT.format(passage=p["text"])
        ids = kit.encode(text)
        t = graded_top1(ids, ablate_edits(ids, mode))
        ok += int(t in variant_ids(COUNTRY[p["category"]]))
        n += 1
    return ok / n


def task_report(mode):
    cats = pools.report_categories(tok, with_additions=True)
    usable = [c for c, r in cap["report"].items() if r["valid"]]
    if not usable:
        return float("nan")
    ok = 0
    for cat in usable:
        ids = kit.encode(pools.REPORT_FEWSHOT.format(cat=cat))
        t = graded_top1(ids, ablate_edits(ids, mode))
        ok += int(t in {v for m in cats[cat] for v in variant_ids(m)})
    return ok / len(usable)


TASKS = [("wikitext", task_wikitext, "shallow"), ("copy", task_copy, "shallow"),
         ("cont_lang", task_contlang, "shallow"), ("twohop", task_twohop, "flexible"),
         ("country", task_country, "flexible"), ("report", task_report, "flexible")]

results = {}
for mode in (None, "ablate", "randproj", "noise"):
    label = mode or "clean"
    results[label] = {}
    for name, fn, kind in TASKS:
        score = fn(mode)
        results[label][name] = score
        print(f"{label:9s} {name:10s} ({kind:8s}) {score:.2f}", flush=True)
    if mode == "ablate" and REMOVED:
        frac = sum(a for a, b in REMOVED) / sum(b for a, b in REMOVED)
        print(f"   [ablation removed {frac:.1%} of residual norm on average]")

print(f"\nretention vs clean (k={K}, layers {LAYS}, centered):")
print(f"{'task':10s} {'kind':8s} {'ablate':>7} {'randproj':>9} {'noise':>7}")
for name, fn, kind in TASKS:
    c = results["clean"][name]
    line = f"{name:10s} {kind:8s}"
    for m in ("ablate", "randproj", "noise"):
        line += f" {results[m][name] / c if c else float('nan'):>7.2f}" if c else "    n/a"
    print(line)
suffix2 = f"_L{LAYS[0]}-{LAYS[-1]}" if len(sys.argv) > 3 else ""
json.dump(results, open(f"/tiny-jlens/gpt2/results/c5b_{MODEL}_k{K}{suffix2}.json", "w"))
