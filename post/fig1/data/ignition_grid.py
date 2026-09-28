"""Full layer x mixture grids for the ignition sweep (paper Fig 29 B/C), for Figure 1.

Same sweep as `jl.followups.ignition` (16 country pairs x 40 of the paper's sentences, 21 mixing
weights), but it keeps the whole curves rather than summary statistics.  As in the paper's
released figure data, the x-axis is alpha minus the trial's threshold, on [-0.5, 0.5]; we take the
threshold to be where the final layer's share crosses 0.5.  Curves are held flat past alpha = 0 or 1,
then averaged over trials.

Writes post/fig1/data/ignition_grid.json.  Run from the repo root:
    PYTHONPATH=. python post/fig1/data/ignition_grid.py      (about 3 minutes on CPU)
"""
import itertools
import json
import pathlib
import random

import numpy as np
import torch
from jlens.hooks import ActivationRecorder

import jl

OUT = pathlib.Path(__file__).resolve().parent / "ignition_grid.json"  # read by post/fig1/build.py
ALPHAS = np.array([i / 20 for i in range(21)])
D_GRID = np.round(np.linspace(-0.5, 0.5, 101), 3)  # alpha - threshold, as in the paper's data


def threshold(share_final):
    """Where the final-layer share first crosses 0.5."""
    s = share_final
    t = 0.5
    for i in range(1, len(s)):
        if (s[i - 1] - 0.5) * (s[i] - 0.5) <= 0 and s[i] != s[i - 1]:
            t = ALPHAS[i - 1] + (0.5 - s[i - 1]) / (s[i] - s[i - 1]) * (ALPHAS[i] - ALPHAS[i - 1])
            break
    return float(t)


@torch.no_grad()
def main():
    lm = jl.Lensed()
    D = json.load(open(jl.REF_DATA / "ignition.json"))
    countries = [c for c in D["countries_12"] if lm.is_single(" " + c)]
    pairs = random.Random(0).sample(list(itertools.combinations(countries, 2)), 16)
    carriers = D["ctx_templates"][:40]
    wte = lm.m._embed_tokens.weight
    layers = list(range(lm.n_layers))

    share_full, share_J, rank = [], [], []  # each: per trial [n_layer, len(D_GRID)]
    thresholds = []
    for A, B in pairs:
        ta, tb = lm.tid(" " + A), lm.tid(" " + B)
        ea, eb = wte[ta], wte[tb]
        Q = {}
        for L in layers:
            V = torch.stack([lm.v(L, ta), lm.v(L, tb)])
            Q[L] = torch.linalg.qr(V.T)[0]
        for tmpl in carriers:
            pre, post = tmpl.split("{W}")
            pre_ids = lm.tok(pre.rstrip(), add_special_tokens=False).input_ids
            post_ids = lm.tok(post, add_special_tokens=False).input_ids if post else []
            ids = [lm.bos] + pre_ids + [ta] + post_ids
            w = 1 + len(pre_ids)
            emb = wte[torch.tensor(ids, device=lm.device)][None].repeat(len(ALPHAS), 1, 1)
            for i, a in enumerate(ALPHAS):
                emb[i, w] = (1 - a) * eb + a * ea
            with ActivationRecorder(lm.m.layers, at=layers) as rec:
                lm.m._text_module(inputs_embeds=emb, use_cache=False)
                acts = torch.stack([rec.activations[L][:, w].float() for L in layers], 1)  # [a, L, d]

            sf, sj, rk = [], [], []
            for L in layers:
                h = acts[:, L]
                d = h[-1] - h[0]
                sf.append((((h - h[0]) @ d) / d.norm() ** 2).cpu().numpy())
                hp = (h - h[0]) @ Q[L]
                sj.append(((hp @ hp[-1]) / hp[-1].norm().clamp_min(1e-9) ** 2).cpu().numpy())
                lg = lm.lens_logits(h, L)                                   # [a, vocab]
                r = (lg[:, None, :] > lg[:, [ta, tb]][:, :, None]).sum(-1) + 1  # [a, 2]
                rk.append(r.min(1).values.float().cpu().numpy())
            t = threshold(sf[-1])
            thresholds.append(t)
            x = ALPHAS - t
            share_full.append([np.interp(D_GRID, x, y) for y in sf])
            share_J.append([np.interp(D_GRID, x, y) for y in sj])
            rank.append([np.exp(np.interp(D_GRID, x, np.log(y))) for y in rk])
        print(A, B, flush=True)

    sf, sj, rk = (np.array(x) for x in (share_full, share_J, rank))  # [trial, L, d_grid]
    json.dump(dict(
        pairs=pairs, n_carriers=len(carriers), d_grid=D_GRID.tolist(), layers=layers,
        share_full=sf.mean(0).tolist(),
        share_J=sj.mean(0).tolist(),
        rank_median=np.median(rk, 0).tolist(),
        rank_geomean=np.exp(np.log(rk).mean(0)).tolist(),
        threshold_median=float(np.median(thresholds)),
    ), open(OUT, "w"))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
