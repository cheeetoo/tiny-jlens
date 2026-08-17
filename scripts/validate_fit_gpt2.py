"""Validate our fitting pipeline against the Neuronpedia prefit gpt2-small lens.

Comparisons:
1. Per-layer Pearson correlation of J_l entries: ours (A+B merged) vs prefit,
   and A vs B (sampling-noise ceiling on the same corpus).
2. Readout agreement: top-10 lens tokens per (layer, position) on held-out
   wikitext prompts — Jaccard overlap ours-vs-prefit vs A-vs-B.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch, transformers, jlens
from tinyjlens.corpus import wikitext_prompts

lensA = jlens.JacobianLens.load("runs/gpt2-fitA/lensA.pt")
lensB = jlens.JacobianLens.load("runs/gpt2-fitB/lensB.pt")
ours = jlens.JacobianLens.merge([lensA, lensB])
theirs = jlens.JacobianLens.load("lenses/gpt2-small/gpt2_jacobian_lens.pt")
print(f"ours: {ours}\ntheirs: {theirs}")

def corr(X, Y):
    x, y = X.flatten().float(), Y.flatten().float()
    x, y = x - x.mean(), y - y.mean()
    return float((x @ y) / (x.norm() * y.norm() + 1e-9))

print(f"\n{'layer':>5} {'corr(ours,theirs)':>18} {'corr(A,B)':>10}")
for l in ours.source_layers:
    print(f"{l:>5} {corr(ours.jacobians[l], theirs.jacobians[l]):>18.4f} "
          f"{corr(lensA.jacobians[l], lensB.jacobians[l]):>10.4f}")

# readout agreement on held-out prompts
tok = transformers.AutoTokenizer.from_pretrained("gpt2")
hf = transformers.AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.bfloat16).cuda()
model = jlens.from_hf(hf, tok)
held_out = wikitext_prompts(12, skip=5000)

def topk_sets(lens, prompt, layers, k=10):
    ll, _, _ = lens.apply(model, prompt, layers=layers, max_seq_len=96)
    return {l: ll[l].topk(k, dim=-1).indices for l in layers}

layers = [1, 3, 5, 7, 9, 10]
jac_ot, jac_ab = {l: [] for l in layers}, {l: [] for l in layers}
for p in held_out:
    t_ours, t_theirs = topk_sets(ours, p, layers), topk_sets(theirs, p, layers)
    t_A, t_B = topk_sets(lensA, p, layers), topk_sets(lensB, p, layers)
    for l in layers:
        for pos in range(0, t_ours[l].shape[0], 7):
            a, b = set(t_ours[l][pos].tolist()), set(t_theirs[l][pos].tolist())
            jac_ot[l].append(len(a & b) / len(a | b))
            a, b = set(t_A[l][pos].tolist()), set(t_B[l][pos].tolist())
            jac_ab[l].append(len(a & b) / len(a | b))

print(f"\n{'layer':>5} {'top10 Jaccard ours-vs-theirs':>28} {'A-vs-B':>8}")
for l in layers:
    m = lambda xs: sum(xs) / len(xs)
    print(f"{l:>5} {m(jac_ot[l]):>28.3f} {m(jac_ab[l]):>8.3f}")
