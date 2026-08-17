"""Layer-band analysis: locate the workspace band (paper §4.1).

Metrics per layer:
- excess kurtosis of lens readout logits (nonrandomness of the readout)
- top-k agreement between lens readout and the model's actual top-1 next token
- autocorrelation of the top-1 lens token across positions, vs a shuffled null
- effective dimensionality of the J-lens vector set W_U J_l
- CKA similarity of J-lens vector geometry between layers
- top-1 agreement between J-lens and logit lens
"""

from __future__ import annotations

import torch


@torch.no_grad()
def readout_stats(kit, prompts: list[str], layers: list[int], *,
                  max_seq_len: int = 96, skip_first: int = 8) -> dict:
    """Corpus statistics of lens readouts.

    Returns dict of per-layer lists: kurtosis, top{1,5,10}_next_agree,
    autocorr (Δ=1) vs null, logitlens_top1_agree.
    """
    stats = {
        "kurtosis": {l: [] for l in layers},
        "top1_next": {l: [] for l in layers},
        "top5_next": {l: [] for l in layers},
        "top10_next": {l: [] for l in layers},
        "autocorr_top1": {l: [] for l in layers},
        "null_top1": {l: [] for l in layers},
        "logitlens_top1_agree": {l: [] for l in layers},
    }
    final = kit.model.n_layers - 1
    for prompt in prompts:
        ids = kit.model.encode(prompt, max_length=max_seq_len)
        resid = kit.residuals(ids, layers + [final])
        model_logits = kit.model.unembed(resid[final]).float()
        model_top1 = model_logits.argmax(-1)  # [seq]
        seq = model_top1.shape[0]
        sl = slice(skip_first, seq - 1)
        for l in layers:
            lens_logits = kit.lens_logits(resid[l], l)  # [seq, vocab]
            # excess kurtosis over vocab, mean over positions
            x = lens_logits[sl]
            mu = x.mean(-1, keepdim=True)
            sd = x.std(-1, keepdim=True).clamp_min(1e-6)
            z = (x - mu) / sd
            kurt = (z**4).mean(-1) - 3.0
            stats["kurtosis"][l].append(kurt.mean().item())
            topk = lens_logits.topk(10, dim=-1).indices  # [seq, 10]
            hit1 = (topk[sl, :1] == model_top1[sl, None]).any(-1).float().mean().item()
            hit5 = (topk[sl, :5] == model_top1[sl, None]).any(-1).float().mean().item()
            hit10 = (topk[sl] == model_top1[sl, None]).any(-1).float().mean().item()
            stats["top1_next"][l].append(hit1)
            stats["top5_next"][l].append(hit5)
            stats["top10_next"][l].append(hit10)
            top1 = topk[:, 0]
            agree = (top1[sl][:-1] == top1[sl][1:]).float().mean().item()
            perm = top1[sl][torch.randperm(top1[sl].shape[0])]
            null = (perm[:-1] == perm[1:]).float().mean().item()
            stats["autocorr_top1"][l].append(agree)
            stats["null_top1"][l].append(null)
            # logit lens comparison (identity transport)
            ll_top1 = kit.model.unembed(resid[l]).float().argmax(-1)
            stats["logitlens_top1_agree"][l].append(
                (ll_top1[sl] == topk[sl, 0]).float().mean().item()
            )
    return stats


@torch.no_grad()
def effective_dim(kit, layers: list[int], *, n_tokens: int = 8192,
                  shares=(0.5, 0.8, 0.9, 0.99)) -> dict:
    """Fraction of dims needed to capture variance shares of {v_t} = U_s J_l."""
    g = torch.Generator().manual_seed(0)
    sample = torch.randperm(kit.U_eff.shape[0], generator=g)[:n_tokens].cuda()
    out = {}
    for l in layers:
        V = kit.U_eff[sample] @ kit.lens.jacobians[l]  # [N, d]
        V = V - V.mean(0, keepdim=True)
        s = torch.linalg.svdvals(V)
        var = s**2 / (s**2).sum()
        cum = var.cumsum(0)
        out[l] = {str(sh): float((cum < sh).sum() + 1) / V.shape[1] for sh in shares}
    return out


@torch.no_grad()
def cka_matrix(kit, layers: list[int], *, n_tokens: int = 2048) -> torch.Tensor:
    """Linear CKA between layers of the J-lens vector geometry."""
    g = torch.Generator().manual_seed(0)
    sample = torch.randperm(kit.U_eff.shape[0], generator=g)[:n_tokens].cuda()
    feats = []
    for l in layers:
        V = kit.U_eff[sample] @ kit.lens.jacobians[l]
        V = V - V.mean(0, keepdim=True)
        feats.append(V)
    n = len(layers)
    M = torch.zeros(n, n)
    for i in range(n):
        for j in range(i, n):
            X, Y = feats[i], feats[j]
            xty = (X.T @ Y).norm() ** 2
            xtx = (X.T @ X).norm()
            yty = (Y.T @ Y).norm()
            M[i, j] = M[j, i] = (xty / (xtx * yty + 1e-9)).item()
    return M
