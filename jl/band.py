"""The structural statistics behind the choice of workspace band for gpt2-small.

Three analyses, all reported in PROTOCOL.md ("Appendix: the workspace band"):

  stats     paper Fig 28 metrics, over 48 wikitext-103 validation sequences x 128 tokens: lens-vs-
            model top-1/top-10 agreement, excess kurtosis of the lens readout, top-1
            autocorrelation (lags 1-3) against a shuffled null, effective dimensionality of the
            centered dictionary, and linear CKA between layer dictionaries (see `cka` for why
            linear CKA is uninformative here).      -> results/band/band.json
  cka       dictionary similarity between layers, controlling for the dominant direction.  Even
            after vocabulary-mean centering one principal component carries 26-37% of each
            layer's dictionary variance at layers 0-10 (2% at 11), so plain linear CKA is ~1
            everywhere.  Variants: drop the top-5/20 PCs, or mean squared canonical correlation
            between top-r subspaces.  Result: smooth drift with depth, no block structure.
                                                    -> results/band/cka.json
  mlp_gain  paper Fig 32: the output norm of block L+1's MLP on a unit direction (after that
            block's pre-LN), normalized by the median over isotropic random unit directions;
            2000 J-lens vectors per layer, raw and centered, with the block's neuron output
            directions as a second control.         -> results/band/mlp_gain.json

Run:  python -m jl.band [stats|cka|mlp_gain]   (default: all three)
CPU is fine: about 2 + 4 + 1 minutes.  DEVICE=cuda uses the GPU.
"""
from __future__ import annotations

import json
import random
import sys
import time

import torch

import jl

LAYERS = list(range(12))


def _seeded(lm=None):
    torch.set_num_threads(1)
    random.seed(0)
    torch.manual_seed(0)
    return lm or jl.Lensed()


# =============================================================================== Fig 28 metrics
def stats():
    """Layer-wise lens statistics on wikitext-103 validation text."""
    from datasets import load_dataset

    lm = _seeded()
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    texts = [t for t in ds["text"] if len(t) > 600]
    random.shuffle(texts)
    seqs = []
    for t in texts[:48]:
        ids = lm.tok(t, add_special_tokens=False).input_ids[:127]
        seqs.append(torch.tensor([[lm.bos] + ids]))

    t0 = time.time()
    st = {L: dict(top1=0, top10=0, n=0, kurt=[], auto=[], null=[]) for L in LAYERS}
    for ids in seqs:
        res = lm.residuals(ids, LAYERS)
        out = lm.logits(ids)
        model_top1 = out[:-1].argmax(-1)                       # prediction at each position
        for L in LAYERS:
            lg = lm.lens_logits(res[L], L)                     # [T, vocab]
            lens_top = lg[:-1].topk(10).indices
            st[L]["top1"] += int((lens_top[:, 0] == model_top1).sum())
            st[L]["top10"] += int((lens_top == model_top1[:, None]).any(1).sum())
            st[L]["n"] += len(model_top1)
            z = (lg - lg.mean(-1, keepdim=True)) / lg.std(-1, keepdim=True)
            st[L]["kurt"].append(float(((z ** 4).mean(-1) - 3).mean()))
            t1 = lg.argmax(-1)
            T = len(t1)
            for d in [1, 2, 3]:
                st[L]["auto"].append(float((t1[:-d] == t1[d:]).float().mean()))
                perm = t1[torch.randperm(T)]
                st[L]["null"].append(float((perm[:-d] == perm[d:]).float().mean()))
    print("forward stats done", round(time.time() - t0), "s")

    # dictionary geometry on a 4000-token vocab subsample: effective dimensionality + CKA
    idx = torch.randperm(lm.vocab)[:4000]
    D = {L: lm.V(L)[idx] for L in LAYERS}

    def effdim(V):
        s = torch.linalg.svdvals(V - V.mean(0))
        c = (s ** 2).cumsum(0) / (s ** 2).sum()
        return float((c < 0.9).sum() + 1) / V.shape[1]

    def cka_gram(A, B):
        A = A - A.mean(0)
        B = B - B.mean(0)
        K = A @ A.T
        Lk = B @ B.T
        return float((K * Lk).sum() / ((K * K).sum().sqrt() * (Lk * Lk).sum().sqrt()))

    ed = {L: effdim(D[L]) for L in LAYERS}
    C = [[cka_gram(D[a], D[b]) for b in LAYERS] for a in LAYERS]
    print("\n L  top1  top10  exKurt  autocorr  null   effdim(90%var)")
    for L in LAYERS:
        s = st[L]
        print(f'{L:2d}  {s["top1"]/s["n"]:.2f}  {s["top10"]/s["n"]:.2f}   '
              f'{sum(s["kurt"])/len(s["kurt"]):6.1f}    {sum(s["auto"])/len(s["auto"]):.3f}   '
              f'{sum(s["null"])/len(s["null"]):.3f}   {ed[L]:.2f}')
    print("\nCKA between layer dictionaries (rows/cols = layers 0..11)")
    for a in LAYERS:
        print(f"{a:2d} " + " ".join(f"{C[a][b]:.2f}" for b in LAYERS))
    json.dump(dict(stats={L: dict(top1=s["top1"] / s["n"], top10=s["top10"] / s["n"],
                                  kurt=sum(s["kurt"]) / len(s["kurt"]),
                                  auto=sum(s["auto"]) / len(s["auto"]),
                                  null=sum(s["null"]) / len(s["null"])) for L, s in st.items()},
                   effdim=ed, cka=C),
              open(jl.results_dir("band") / "band.json", "w"))


# =============================================================================== dictionary CKA
def cka():
    """Dictionary similarity between layers, controlling for the dominant direction."""
    lm = _seeded()
    idx = torch.randperm(lm.vocab)[:3000]
    D, SV = {}, {}
    for L in LAYERS:
        V = lm.V(L)[idx]
        V = V - V.mean(0)
        D[L] = V
        SV[L] = torch.linalg.svd(V, full_matrices=False)

    def cka_feat(A, B):   # linear CKA, feature-space form
        return float((A.T @ B).norm() ** 2 / ((A.T @ A).norm() * (B.T @ B).norm()))

    def whiten(L, r):
        return SV[L][0][:, :r]

    def drop_top(L, k):
        U, S, Vh = SV[L]
        S = S.clone()
        S[:k] = 0
        return U @ torch.diag(S) @ Vh

    W = {r: {L: whiten(L, r) for L in LAYERS} for r in [50, 100, 300]}
    T5 = {L: drop_top(L, 5) for L in LAYERS}
    T20 = {L: drop_top(L, 20) for L in LAYERS}
    out = {}
    for name, f in [("linear", lambda a, b: cka_feat(D[a], D[b])),
                    ("drop_top5", lambda a, b: cka_feat(T5[a], T5[b])),
                    ("drop_top20", lambda a, b: cka_feat(T20[a], T20[b])),
                    ("mean_cca_r50", lambda a, b: cka_feat(W[50][a], W[50][b])),
                    ("mean_cca_r100", lambda a, b: cka_feat(W[100][a], W[100][b])),
                    ("mean_cca_r300", lambda a, b: cka_feat(W[300][a], W[300][b]))]:
        M = [[f(a, b) for b in LAYERS] for a in LAYERS]
        out[name] = M
        print("\n" + name, flush=True)
        for a in LAYERS:
            print(f"{a:2d} " + " ".join(f"{M[a][b]:.2f}" for b in LAYERS), flush=True)
    print("\nlayer: top-1 PC share, top-5 PC share, #PCs for 90% var")
    for L in LAYERS:
        s = SV[L][1] ** 2
        c = s.cumsum(0) / s.sum()
        print(L, f"{float(s[0]/s.sum()):.2f}", f"{float(s[:5].sum()/s.sum()):.2f}", int((c < 0.9).sum()) + 1)
    json.dump(out, open(jl.results_dir("band") / "cka.json", "w"))


# =============================================================================== MLP gain
def mlp_gain():
    """Paper Fig 32: the MLP gain of J-lens directions, against random and neuron directions."""
    lm = _seeded()
    blocks = lm.m.layers  # HF GPT2Block modules

    def mlp_out(block, V):            # V [n, d] unit directions
        return block.mlp(block.ln_2(V)).norm(dim=-1)

    out = {}
    idx = torch.randperm(lm.vocab)[:2000]
    with torch.no_grad():
        for L in range(11):
            blk = blocks[L + 1]
            rnd = torch.randn(2000, lm.d, device=lm.device)
            rnd = rnd / rnd.norm(dim=-1, keepdim=True)
            base = mlp_out(blk, rnd).median()
            Vc = lm.V(L)[idx]
            Vc = Vc / Vc.norm(dim=-1, keepdim=True)
            Vr = (lm.U @ lm.J[L])[idx]
            Vr = Vr / Vr.norm(dim=-1, keepdim=True)
            g_c = (mlp_out(blk, Vc) / base).median().item()
            g_r = (mlp_out(blk, Vr) / base).median().item()
            # neuron output directions of the same block, the paper's comparison set
            W = blk.mlp.c_proj.weight  # [d_ff, d] in HF Conv1D
            Wn = W[torch.randperm(W.shape[0])[:2000]]
            Wn = Wn / Wn.norm(dim=-1, keepdim=True)
            g_n = (mlp_out(blk, Wn) / base).median().item()
            out[L] = dict(centered=g_c, raw=g_r, neuron=g_n)
            print(f"L{L:2d}  centered J-lens {g_c:.2f}   raw J-lens {g_r:.2f}   neuron dirs {g_n:.2f}", flush=True)
    json.dump(out, open(jl.results_dir("band") / "mlp_gain.json", "w"))


if __name__ == "__main__":
    which = sys.argv[1:] or ["stats", "cka", "mlp_gain"]
    for name in which:
        print(f"===== {name}", flush=True)
        {"stats": stats, "cka": cka, "mlp_gain": mlp_gain}[name]()
