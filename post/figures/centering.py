"""Figure 2: raw vs centered J-lens vectors in gpt2-small.

(a) schematic of the shared component, (b) pairwise cosine histogram at layer 8 (real data),
(c) headline interventions with raw vs centered vectors (from results/followups/variants.json).

Run from the repo root:  python post/figures/centering.py
"""
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch

import jl

OUT = pathlib.Path(__file__).resolve().parent
MAROON, SLATE, GRAY = "#800000", "#33658a", "#8a8580"
BG = "#fbfbf9"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG})

torch.manual_seed(0)
lm = jl.Lensed(device="cpu")
L = 8
sub = torch.randperm(lm.vocab)[:3000]
raw = lm.U @ lm.J[L]
cen = raw - raw.mean(0, keepdim=True)


def pairwise(M):
    Mn = torch.nn.functional.normalize(M[sub], dim=1)
    G = Mn @ Mn.T
    iu = torch.triu_indices(len(sub), len(sub), 1)
    return G[iu[0], iu[1]].numpy()


pw_raw, pw_cen = pairwise(raw), pairwise(cen)
per_layer = []
for LL in range(12):
    r = lm.U @ lm.J[LL]
    per_layer.append(dict(layer=LL, raw=float(pairwise(r).mean()), centered=float(pairwise(r - r.mean(0)).mean())))
json.dump(dict(layer8=dict(raw_mean=float(pw_raw.mean()), centered_mean=float(pw_cen.mean())), per_layer=per_layer),
          open(OUT / "centering_stats.json", "w"), indent=1)

var = json.load(open(jl.RESULTS / "followups/variants.json")) if (jl.RESULTS / "followups/variants.json").exists() else None

fig, axes = plt.subplots(1, 3 if var else 2, figsize=(15 if var else 10.5, 4.2),
                         gridspec_kw=dict(width_ratios=[1, 1.25, 1.25] if var else [1, 1.25]))

# (a) schematic: raw vectors = a shared component plus a small token-specific part
ax = axes[0]
rng = np.random.default_rng(3)
n = 12
mu = np.array([0.95, 0.55])
spec_ang = np.deg2rad(np.linspace(0, 360, n, endpoint=False) + rng.normal(0, 8, n))
spec = 0.32 * np.stack([np.cos(spec_ang), np.sin(spec_ang)], 1)
o1 = np.array([-1.35, -0.1])
for sp in spec:
    ax.annotate("", xy=o1 + mu + sp, xytext=o1, arrowprops=dict(arrowstyle="-|>", color=MAROON, lw=1.3, alpha=0.6))
ax.annotate("", xy=o1 + mu, xytext=o1, arrowprops=dict(arrowstyle="-|>", color="#222222", lw=2.4), zorder=5)
ax.text(o1[0] + 0.5, o1[1] - 0.28, "raw", ha="center", color=MAROON, fontsize=11)
ax.text(o1[0] + 0.62, o1[1] + 0.18, "mean", color="#222222", fontsize=9.5, ha="left")
o2 = np.array([1.25, 0.45])
for sp in spec:
    ax.annotate("", xy=o2 + 1.9 * sp, xytext=o2, arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.3, alpha=0.85))
ax.text(o2[0], o2[1] - 0.83, "centered (raw minus mean)", ha="center", color=SLATE, fontsize=11)
ax.set_xlim(-1.7, 2.3)
ax.set_ylim(-0.55, 1.2)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("(a) Schematic", fontsize=11)

# (b) histogram
ax = axes[1]
bins = np.linspace(-0.7, 1.0, 86)
ax.hist(pw_raw, bins=bins, color=MAROON, alpha=0.75, label=f"raw (mean {pw_raw.mean():.2f})")
ax.hist(pw_cen, bins=bins, color=SLATE, alpha=0.75, label=f"centered (mean {abs(pw_cen.mean()):.2f})")
ax.set_xlabel("Cosine similarity between two tokens' J-lens vectors")
ax.set_ylabel("Token pairs")
ax.set_title("(b) Layer 8, 3,000 random tokens", fontsize=11)
ax.legend(loc="upper left", frameon=False)

# (c) interventions
if var:
    ax = axes[2]
    rows = [("Verbal report swap\n(target becomes top-1)", "c1b_top1", None),
            ("Two-hop swap\n(target answer top-1)", "c3_top1", None),
            ("Two-hop accuracy after\nJ-space ablation", "c5_twohop_J", "c5_twohop_R")]
    x = np.arange(len(rows))
    w = 0.36
    ax.bar(x - w / 2, [var["raw"][k] for _, k, _ in rows], w, color=MAROON, label="raw")
    ax.bar(x + w / 2, [var["centered"][k] for _, k, _ in rows], w, color=SLATE, label="centered")
    ax.set_xticks(x, [r for r, _, _ in rows], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of trials")
    ax.set_title("(c) Interventions with each set of vectors", fontsize=11)
    ax.legend(frameon=False, loc="upper center", ncol=2)
    ax.set_ylim(0, 1.15)

fig.tight_layout()
fig.savefig(OUT / "fig2_centering.png", dpi=200)
print("saved", OUT / "fig2_centering.png", pw_raw.mean(), pw_cen.mean())
print(per_layer)
