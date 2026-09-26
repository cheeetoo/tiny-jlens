import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler
S = str(__import__("pathlib").Path(__file__).resolve().parent)
CREAM = "#fbfbf9"; MAROON = "#800000"; SLATE = "#33658a"; ORANGE = "#d45d00"; TEAL = "#3a7d6a"; GOLD = "#c99000"; GRAY = "#8a8580"
plt.rcParams.update({"figure.facecolor": CREAM, "axes.facecolor": CREAM, "savefig.facecolor": CREAM,
                     "axes.prop_cycle": cycler(color=[MAROON, SLATE, ORANGE, TEAL, GOLD, GRAY]),
                     "savefig.dpi": 200})
D = np.load(f"{S}/geom.npz"); per = json.load(open(f"{S}/per_layer.json"))
labels = list(D["labels"]); L = 8

import transformers
tok = transformers.AutoTokenizer.from_pretrained("openai-community/gpt2")
labels = [str(x) for x in D["labels"] if str(x) in (" the", " of", " apple", " dog")]
lab_ids = [tok(x, add_special_tokens=False).input_ids[0] for x in labels]
sub = D["sub"]

def wedge(L, which):
    """x = component along the vocabulary-mean direction, y = norm of the orthogonal remainder."""
    n = D[f"L{L}_{'raw' if which == 'raw' else 'cen'}_norm"]; c = D[f"L{L}_cos_mu_{which}"]
    return n * c, n * np.sqrt(np.clip(1 - c ** 2, 0, None))

def scatter_pair(axr, axc, L, annotate=True, equal=True):
    xr, yr = wedge(L, "raw"); xc, yc = wedge(L, "cen"); mu = per[L]["mu_norm"]
    axr.scatter(xr[sub], yr[sub], s=4, alpha=0.25, color=MAROON, linewidths=0)
    axc.scatter(xc[sub], yc[sub], s=4, alpha=0.25, color=SLATE, linewidths=0)
    axr.annotate("", xy=(mu, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    axr.text(mu / 2, -0.9, "vocabulary mean", ha="center", va="top", fontsize=9)
    if annotate:
        for i, s in enumerate(labels):
            t = lab_ids[i]
            axr.annotate(repr(s), (xr[t], yr[t]), fontsize=8, xytext=(4, 2), textcoords="offset points")
            axc.annotate(repr(s), (xc[t], yc[t]), fontsize=8, xytext=(4, 2), textcoords="offset points")
            axr.scatter(xr[t], yr[t], s=14, color="black", zorder=3)
            axc.scatter(xc[t], yc[t], s=14, color="black", zorder=3)
    for ax in (axr, axc):
        ax.axhline(0, color=GRAY, lw=0.8); ax.axvline(0, color=GRAY, lw=0.8)
        if equal: ax.set_aspect("equal")
        ax.set_xlabel("component along the vocabulary-mean direction")
        ax.set_ylim(-2.5, 13.5)
    axr.set_ylabel("norm of the component orthogonal to it")
    axr.set_title(f"raw  v_t = J_{L}ᵀ w_t"); axc.set_title(f"centered  v_t − mean_t' v_t'")

# ---- v1: two-panel scatter, shared axes
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.4), sharex=True, sharey=True, constrained_layout=True)
scatter_pair(a, b, L)
fig.suptitle(f"J-lens vectors at layer {L}, gpt2-small (4000 random tokens; angle from the x-axis = angle to the mean)")
fig.savefig(f"{S}/v1_scatter_L{L}.png")

# ---- v2: histograms of pairwise cosine, raw vs centered
fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
bins = np.linspace(-0.4, 1, 71)
ax.hist(D[f"L{L}_pw_raw"], bins=bins, alpha=0.7, label=f"raw (mean {per[L]['pw_raw_mean']:.2f})")
ax.hist(D[f"L{L}_pw_cen"], bins=bins, alpha=0.7, label=f"centered (mean {per[L]['pw_cen_mean']:.2f})")
ax.set_xlabel("cosine similarity between two J-lens vectors"); ax.set_ylabel("token pairs")
ax.set_title(f"Pairwise cosine of J-lens vectors, layer {L}, gpt2-small"); ax.legend()
fig.savefig(f"{S}/v2_hist_L{L}.png")

# ---- v3: per-layer
Ls = [p["L"] for p in per]
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
a.plot(Ls, [p["pw_raw_mean"] for p in per], "o-", label="raw")
a.plot(Ls, [p["pw_cen_mean"] for p in per], "o-", label="centered")
a.set_xlabel("layer"); a.set_ylabel("mean pairwise cosine"); a.set_title("Mean pairwise cosine of J-lens vectors"); a.legend()
a.set_xticks(Ls); a.axvspan(6.5, 9.5, color=GRAY, alpha=0.15, lw=0)
b.plot(Ls, [p["mu_norm"] for p in per], "o-", label="‖vocabulary mean‖")
b.plot(Ls, [p["mean_cen_norm"] for p in per], "o-", label="mean ‖v_t − mean‖")
b.plot(Ls, [p["mean_raw_norm"] for p in per], "o-", color=GRAY, label="mean ‖v_t‖ (raw)")
b.set_xlabel("layer"); b.set_ylabel("norm"); b.set_title("Size of the common component"); b.legend()
b.set_xticks(Ls); b.axvspan(6.5, 9.5, color=GRAY, alpha=0.15, lw=0)
fig.suptitle("gpt2-small; layer 11 is the lens target (J_11 = I); band 7–9 shaded")
fig.savefig(f"{S}/v3_layers.png")

# ---- v4: schematic + scatter pair
fig = plt.figure(figsize=(13, 3.9), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1, 1])
s = fig.add_subplot(gs[0]); a = fig.add_subplot(gs[1]); b = fig.add_subplot(gs[2], sharex=a, sharey=a)
s.set_axis_off(); s.set_xlim(0, 1); s.set_ylim(0, 1)
box = dict(boxstyle="round,pad=0.4", fc=CREAM, ec="black", lw=1)
steps = [(0.9, "unembedding row of token t\n$w_t \\in \\mathbb{R}^{768}$   (row t of $W_U$)"),
         (0.66, "pull back through the lens Jacobian\n$v_t = J_L^{\\top} w_t$"),
         (0.42, "raw J-lens vector $v_t$\n(all $v_t$ share a large common component)"),
         (0.18, "subtract the vocabulary mean\n$v_t \\leftarrow v_t - \\frac{1}{|V|}\\sum_{t'} v_{t'}$")]
for y, txt in steps:
    s.text(0.5, y, txt, ha="center", va="center", fontsize=9.5, bbox=box)
for y0, y1 in [(0.9, 0.66), (0.66, 0.42), (0.42, 0.18)]:
    s.annotate("", xy=(0.5, y1 + 0.085), xytext=(0.5, y0 - 0.085), arrowprops=dict(arrowstyle="->", lw=1.2, color="black"))
s.text(0.5, 0.02, "readout $W_U\\,\\mathrm{ln_f}(J_L h)$ is unchanged by the last step", ha="center", va="center", fontsize=8.5, color=GRAY)
s.set_title("construction")
scatter_pair(a, b, L)
fig.savefig(f"{S}/v4_schematic_scatter_L{L}.png")

# ---- v5: cosine-to-mean histogram (single panel, simplest)
fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
bins = np.linspace(-1, 1, 81)
ax.hist(D[f"L{L}_cos_mu_raw"], bins=bins, alpha=0.7, label="raw")
ax.hist(D[f"L{L}_cos_mu_cen"], bins=bins, alpha=0.7, label="centered")
ax.set_xlabel("cosine of v_t with the vocabulary-mean direction"); ax.set_ylabel("tokens (full vocabulary)")
ax.set_title(f"J-lens vectors and the mean direction, layer {L}, gpt2-small"); ax.legend()
fig.savefig(f"{S}/v5_cos_to_mean_L{L}.png")
print("done")
