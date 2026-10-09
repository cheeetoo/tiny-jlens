"""The structural statistics behind the choice of workspace band for gpt2-small.

Four analyses, all reported in PROTOCOL.md ("Appendix: the workspace band"):

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
  fig28     paper Fig 28 as the paper plots it, on the same sequences as `stats`: top-k agreement
            for k = 1..128, percentiles over positions of the readout's excess kurtosis, top-1
            autocorrelation as the gain in lens log-probability over a position-shuffled null at
            offsets 1..32 (and again using only top-1 tokens that occur nowhere in the sequence's
            input, for the real pairs and the null alike), and the fraction of dimensions needed for 90..99.5% of the
            variance of the full centered dictionary.
                                                    -> results/band/fig28.json

Run:  python -m jl.band [stats|cka|mlp_gain|fig28]   (default: all four)
CPU is fine: about 2 + 4 + 1 + 2 minutes.  DEVICE=cuda uses the GPU.

The Gemma 3 models of jl.control (gemma-270m, gemma-270m-it, gemma-1b, gemma-1b-it), with their
Neuronpedia lenses, run `cka` and `fig28` the same way:
      python -m jl.band --model gemma-1b-it cka fig28     -> results/band/gemma-1b-it/{cka,fig28}.json
For every model, `cka` and `fig28` also compute the dictionary statistics (CKA, effective
dimensionality) with the final norm's gain folded into the unembedding (keys ending `_gain`): the
direction each token actually reads.  In Gemma this matters (see `GemmaLensed`); in GPT-2 the CKA
shows no blocks either way.  NSEQ sets the number of sequences (default 48).
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

import torch

import jl

LAYERS = list(range(12))


class GemmaLensed:
    """A Gemma 3 model from jl.control behind the interface these statistics use (jl.Lensed's): its
    residuals and logits, the lens readout at every layer (the model's own output at the last layer,
    whose lens is the identity), and the centered J-lens dictionary (U - mean_t U) J_L, read in rows
    (`rows`) or as its d x d Gram matrix (`gram`), since the 262k-row dictionary is too big to hold at
    every layer.  `gain=True` folds the final RMSNorm's gain (1 + w) into U, giving the direction each
    token actually reads; in Gemma that gain varies about 50x across dimensions.  GPT-2's dictionary
    (jl.Lensed.V) leaves its ln_f gain out, and so does the default here."""

    def __init__(self, name):
        from jl import control

        A = control.load(name)
        self.name, self.A, self.m = name, A, A.m
        self.tok, self.device, self.bos, self.d, self.n_layers = A.tok, A.device, A.bos, A.d, A.n_layers
        self.J = dict(A.J)
        self.J[self.n_layers - 1] = torch.eye(self.d, device=self.device)
        gain = 1 + self.m._final_norm.weight.detach().float()
        self.U = {False: A.U, True: A.U * gain}
        self.vocab = A.U.shape[0]
        self._G = {}

    def residuals(self, ids, layers):
        return self.A.residuals(ids, layers)

    def logits(self, ids):
        return self.A.logits(ids)

    def lens_logits(self, h, L):
        return self.m.unembed(h @ self.J[L].T).float()

    def rows(self, L, idx, gain=False):
        U = self.U[gain]
        return (U[idx.to(U.device)] - U.mean(0)) @ self.J[L]

    def gram(self, L, gain=False):
        """V^T V of the full centered dictionary, in float64 on the CPU (MPS has no float64)."""
        if gain not in self._G:
            U = self.U[gain].cpu().double()
            U = U - U.mean(0)
            self._G[gain] = U.T @ U
        J = self.J[L].cpu().double()
        return J.T @ self._G[gain] @ J


def _layers(lm):
    return list(range(lm.n_layers)) if isinstance(lm, GemmaLensed) else LAYERS


def _out(lm):
    return jl.results_dir("band" if not isinstance(lm, GemmaLensed) else f"band/{lm.name}")


def _gpt2_gained(lm):
    """GPT-2's unembedding with its ln_f gain folded in."""
    return lm.U * lm.m._final_norm.weight.detach().float()


def _rows(lm, L, idx, gain=False):
    """Rows `idx` of the centered J-lens dictionary at layer L (`gain`: the final norm's gain folded in)."""
    if isinstance(lm, GemmaLensed):
        return lm.rows(L, idx, gain)
    if not gain:
        return lm.V(L)[idx]
    U = _gpt2_gained(lm)
    return (U[idx] - U.mean(0)) @ lm.J[L]


def _gram(lm, L, gain=False):
    """V^T V of the full centered J-lens dictionary at layer L, in float64."""
    if isinstance(lm, GemmaLensed):
        return lm.gram(L, gain)
    if not gain:
        V = lm.V(L).double()
        return V.T @ V
    U = _gpt2_gained(lm).double()
    U = U - U.mean(0)
    J = lm.J[L].double()
    return J.T @ (U.T @ U) @ J


def _seeded(lm=None):
    torch.set_num_threads(1)
    random.seed(0)
    torch.manual_seed(0)
    return lm or jl.Lensed()


def _wikitext(lm):
    """48 (NSEQ) wikitext-103 validation sequences of 128 tokens (BOS + 127)."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    texts = [t for t in ds["text"] if len(t) > 600]
    random.shuffle(texts)
    seqs = []
    for t in texts[:int(os.environ.get("NSEQ", 48))]:
        ids = lm.tok(t, add_special_tokens=False).input_ids[:127]
        seqs.append(torch.tensor([[lm.bos] + ids], device=lm.device))
    return seqs


# =============================================================================== Fig 28 metrics
def stats():
    """Layer-wise lens statistics on wikitext-103 validation text."""
    lm = _seeded()
    seqs = _wikitext(lm)

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
def cka(lm=None):
    """Dictionary similarity between layers, controlling for the dominant direction."""
    lm = _seeded(lm)
    layers = _layers(lm)
    idx = torch.randperm(lm.vocab)[:3000]
    out = {}
    for gain in [False, True]:
        D, SV = {}, {}
        for L in layers:
            V = _rows(lm, L, idx, gain).cpu()
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

        W = {r: {L: whiten(L, r) for L in layers} for r in [50, 100, 300]}
        T5 = {L: drop_top(L, 5) for L in layers}
        T20 = {L: drop_top(L, 20) for L in layers}
        sfx = "_gain" if gain else ""
        for name, f in [("linear", lambda a, b: cka_feat(D[a], D[b])),
                        ("drop_top5", lambda a, b: cka_feat(T5[a], T5[b])),
                        ("drop_top20", lambda a, b: cka_feat(T20[a], T20[b])),
                        ("mean_cca_r50", lambda a, b: cka_feat(W[50][a], W[50][b])),
                        ("mean_cca_r100", lambda a, b: cka_feat(W[100][a], W[100][b])),
                        ("mean_cca_r300", lambda a, b: cka_feat(W[300][a], W[300][b]))]:
            M = [[f(a, b) for b in layers] for a in layers]
            out[name + sfx] = M
            print("\n" + name + sfx, flush=True)
            for a in layers:
                print(f"{a:2d} " + " ".join(f"{M[a][b]:.2f}" for b in layers), flush=True)
        print(f"\nlayer: top-1 PC share, top-5 PC share, #PCs for 90% var{sfx}")
        for L in layers:
            s = SV[L][1] ** 2
            c = s.cumsum(0) / s.sum()
            print(L, f"{float(s[0]/s.sum()):.2f}", f"{float(s[:5].sum()/s.sum()):.2f}", int((c < 0.9).sum()) + 1)
    json.dump(out, open(_out(lm) / "cka.json", "w"))


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


# =============================================================================== Fig 28 as plotted
def fig28(lm=None):
    """Paper Fig 28's four panels, with the paper's series (top-k, percentiles, offsets, thresholds)."""
    import numpy as np

    lm = _seeded(lm)
    layers = _layers(lm)
    seqs = _wikitext(lm)
    KS, PCTS, DS, FVE = [1, 2, 4, 8, 16, 32, 64, 128], [1, 10, 25, 50, 75, 90, 99], \
        [1, 2, 4, 8, 16, 32], [0.9, 0.95, 0.98, 0.99, 0.995]
    hit = {L: {k: 0 for k in KS} for L in layers}
    n = 0
    kurt = {L: [] for L in layers}
    # [real, null] lists of log-probs, for all pairs and for pairs whose token isn't in the input
    ac = {L: {d: dict(all=([], []), not_input=([], []), n=0, n_input=0) for d in DS} for L in layers}
    t0 = time.time()
    for si, ids in enumerate(seqs):
        if si:
            print(f"  sequence {si}/{len(seqs)} ({time.time() - t0:.0f}s)", flush=True)
        x = ids[0]
        res = lm.residuals(ids, layers)
        model_top1 = lm.logits(ids)[:-1].argmax(-1)
        n += len(model_top1)
        for L in layers:
            lg = lm.lens_logits(res[L], L)                     # [T, vocab]
            top = lg[:-1].topk(max(KS)).indices
            for k in KS:
                hit[L][k] += int((top[:, :k] == model_top1[:, None]).any(1).sum())
            z = (lg - lg.mean(-1, keepdim=True)) / lg.std(-1, keepdim=True)
            kurt[L] += ((z ** 4).mean(-1) - 3).tolist()        # one value per position
            lp = torch.log_softmax(lg, -1)
            t1 = lg.argmax(-1)
            T = len(t1)
            for d in DS:
                # log p at t+d of the top-1 token at t, and of the top-1 token at a random position
                tok_real, tok_null = t1[:-d], t1[torch.randint(T, (T - d,)).to(t1.device)]
                real = lp[d:].gather(1, tok_real[:, None])[:, 0]
                null = lp[d:].gather(1, tok_null[:, None])[:, 0]
                # same rule for both: drop tokens that occur anywhere in the sequence's input
                keep_r = ~torch.isin(tok_real, x)
                keep_n = ~torch.isin(tok_null, x)
                a = ac[L][d]
                a["all"][0].extend(real.tolist())
                a["all"][1].extend(null.tolist())
                a["not_input"][0].extend(real[keep_r].tolist())
                a["not_input"][1].extend(null[keep_n].tolist())
                a["n"] += T - d
                a["n_input"] += int((~keep_r).sum())
    print("forward done", round(time.time() - t0), "s", flush=True)

    def effdim(gain):
        eff = {}
        for L in layers:
            ev = torch.linalg.eigvalsh(_gram(lm, L, gain)).flip(0)
            c = (ev.cumsum(0) / ev.sum()).cpu().numpy()
            eff[L] = {f: float((np.searchsorted(c, f) + 1) / lm.d) for f in FVE}
        return {f: [eff[L][f] for L in layers] for f in FVE}

    gain = lambda L, d, which: float(np.mean(ac[L][d][which][0]) - np.mean(ac[L][d][which][1]))
    out = dict(
        topk={k: [hit[L][k] / n for L in layers] for k in KS},
        kurtosis={p: [float(np.percentile(kurt[L], p)) for L in layers] for p in PCTS},
        autocorr={d: [gain(L, d, "all") for L in layers] for d in DS},
        autocorr_not_input={d: [gain(L, d, "not_input") for L in layers] for d in DS},
        frac_top1_is_input={d: [ac[L][d]["n_input"] / ac[L][d]["n"] for L in layers] for d in DS},
        effdim=effdim(False),
        effdim_gain=effdim(True),
    )
    for key in ["autocorr", "autocorr_not_input", "frac_top1_is_input", "effdim", "effdim_gain"]:
        print(key)
        for s, ys in out[key].items():
            print(f"  {s:>5} " + " ".join(f"{v:6.2f}" for v in ys))
    json.dump(out, open(_out(lm) / "fig28.json", "w"), indent=1)


if __name__ == "__main__":
    args = sys.argv[1:]
    model = None
    if "--model" in args:
        i = args.index("--model")
        model = args[i + 1]
        del args[i:i + 2]
    lm = GemmaLensed(model) if model and model != "gpt2" else None
    which = args or (["stats", "cka", "mlp_gain", "fig28"] if lm is None else ["cka", "fig28"])
    for name in which:
        print(f"===== {name}", flush=True)
        if lm is not None and name not in ("cka", "fig28"):
            raise SystemExit(f"{name} runs on GPT-2 only")
        {"stats": stats, "cka": cka, "mlp_gain": mlp_gain, "fig28": fig28}[name](*([lm] if lm is not None else []))
