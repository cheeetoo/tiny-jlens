"""C1c — privilege test for the report channel (matched-norm probe split).

For each top1-valid report category: build member concept probes (3 "about
the {m}" frames, last position, member-centered within category), split each
probe into its J-span part (centered gradient pursuit, k=16) and the
complement, then matched-norm swap source->target along each part at the
report anchor. Conditions: full / J / nonJ / nonJ with the union of the two
members' GP atoms (plus their own coordinates) clamped. Success = target in
graded top-5, best over alpha in {1,2}. Privilege = J-component swaps flip
the report far more than matched-norm non-J swaps, and clamping the J
coordinates kills what the non-J swaps achieve.

Run:  python experiments/31_c1c.py [model]
"""

import json
import random
import sys

sys.path.insert(0, "/tiny-jlens/gpt2")

import torch

import core
import pools

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
kit = core.Kit(MODEL)
tok = kit.tokenizer
rng = random.Random(1234)

cap = json.load(open(f"results/capability_{MODEL}.json"))
cats_all = pools.report_categories(tok)
valid = [c for c, r in cap["report"].items() if r["valid"]]
print(f"{MODEL} top1-valid cats:", valid)

FRAMES = ["She told me all about the {m}.",
          "The article was mainly about the {m}.",
          "He wrote a story about the {m}."]
LAYS = [l for l in kit.layers if 0.55 * kit.n_layers <= l <= 1.0 * kit.n_layers]
ART = {"a", "an", "the", "called", '"', "'", ""}


def variant_ids(word):
    out = []
    for w in {word, word.capitalize(), word.lower(), word.upper()}:
        for form in (" " + w, w):
            ids = tok(form, add_special_tokens=False)["input_ids"]
            if len(ids) == 1 and ids[0] not in out:
                out.append(ids[0])
    return out


@torch.no_grad()
def anchor(prompt, edits=()):
    ids = kit.encode(prompt)
    for _ in range(3):
        lg = core.logits_with(kit, ids, list(edits))[-1]
        t = int(lg.argmax())
        if tok.decode([t]).strip().lower() in ART:
            ids = torch.cat([ids, torch.tensor([[t]], device=ids.device)], dim=1)
            continue
        return ids, lg
    return ids, lg


res = {"full": [], "J": [], "nonJ": [], "nonJ_clamped": []}
for cat in valid:
    members = list(cats_all[cat])
    cvec = {}
    for m in members:
        hs = {l: [] for l in LAYS}
        for fr in FRAMES:
            r = kit.residuals(kit.encode(fr.format(m=m)), LAYS)
            for l in LAYS:
                hs[l].append(r[l][-1])
        cvec[m] = {l: torch.stack(v).mean(0) for l, v in hs.items()}
    gmean = {l: torch.stack([cvec[m][l] for m in members]).mean(0) for l in LAYS}
    probe = {m: {l: cvec[m][l] - gmean[l] for l in LAYS} for m in members}
    split = {}
    for m in members:
        split[m] = {}
        for l in LAYS:
            sel, _, recon = core.gradient_pursuit(kit, probe[m][l], l, 16, centered=True)
            split[m][l] = (recon, probe[m][l] - recon, sel.tolist())
    ids, lg = anchor(pools.REPORT_FEWSHOT.format(cat=cat))
    src = next((m for m in members if int(lg.argmax()) in variant_ids(m)), None)
    if src is None:
        continue
    elig = [m for m in members if m != src and variant_ids(m)
            and min(int((lg > lg[v]).sum()) for v in variant_ids(m)) >= 10]
    for tgt in rng.sample(elig, min(4, len(elig))):
        for cond in res:
            best = 99999
            for alpha in (1.0, 2.0):
                edits = []
                for l in LAYS:
                    full = probe[tgt][l] - probe[src][l]
                    d = {"full": full,
                         "J": split[tgt][l][0] - split[src][l][0],
                         "nonJ": split[tgt][l][1] - split[src][l][1],
                         "nonJ_clamped": split[tgt][l][1] - split[src][l][1]}[cond]
                    d = d * (alpha * full.norm() / d.norm().clamp_min(1e-8))
                    edits.append(core.add_delta(d, l, None))
                if cond == "nonJ_clamped":
                    ct = sorted(set(split[src][LAYS[-1]][2]) | set(split[tgt][LAYS[-1]][2])
                                | {variant_ids(src)[0], variant_ids(tgt)[0]})
                    edits += core.clamp_coords(kit, ids, LAYS, ct, centered=True)
                _, lg2 = anchor(pools.REPORT_FEWSHOT.format(cat=cat), edits)
                tr = min(int((lg2 > lg2[v]).sum()) for v in variant_ids(tgt))
                best = min(best, tr)
            res[cond].append(best < 5)
    print(".", end="", flush=True)

n = len(res["full"])
print(f"\nC1c at {MODEL} n={n}:")
for c, v in res.items():
    print(f"  {c:14s} {sum(v)}/{n}")
json.dump({k: [bool(x) for x in v] for k, v in res.items()},
          open(f"results/c1c_addendum_{MODEL}.json", "w"))
