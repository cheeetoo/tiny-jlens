import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler
S = str(__import__("pathlib").Path(__file__).resolve().parent)
CREAM = "#fbfbf9"; MAROON = "#800000"; SLATE = "#33658a"; GRAY = "#8a8580"
plt.rcParams.update({"figure.facecolor": CREAM, "axes.facecolor": CREAM, "savefig.facecolor": CREAM,
                     "axes.prop_cycle": cycler(color=[MAROON, SLATE, "#d45d00", "#3a7d6a", "#c99000", GRAY]),
                     "savefig.dpi": 200})
D = np.load(f"{S}/geom.npz"); per = json.load(open(f"{S}/per_layer.json")); L = 8
rng = np.random.default_rng(0)
cr, cc = D[f"L{L}_cos_mu_raw"], D[f"L{L}_cos_mu_cen"]
th_r, th_c = np.arccos(np.clip(cr, -1, 1)), np.arccos(np.clip(cc, -1, 1))

# ---- v6: fan of unit arrows at their angle to the mean direction
pick = rng.choice(len(cr), 400, replace=False)
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True, subplot_kw=dict(aspect="equal"))
for ax, th, col, name in [(a, th_r, MAROON, "raw"), (b, th_c, SLATE, "centered")]:
    for t in th[pick]:
        ax.annotate("", xy=(np.cos(t), np.sin(t)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col, alpha=0.18, lw=0.8, mutation_scale=8))
    ax.annotate("", xy=(1.15, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6, mutation_scale=12))
    ax.text(1.17, 0, "vocabulary\nmean", va="center", fontsize=9)
    ax.set_xlim(-1.2, 1.55); ax.set_ylim(-0.08, 1.1); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{name} J-lens directions, layer {L}")
fig.suptitle("400 random tokens; each arrow is a unit vector drawn at its angle to the vocabulary mean")
fig.savefig(f"{S}/v6_fan_L{L}.png")

# ---- v7: polar histogram of the angle to the mean, full vocabulary
fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True, subplot_kw=dict(projection="polar"))
bins = np.linspace(0, np.pi, 61)
for th, name in [(th_r, "raw"), (th_c, "centered")]:
    h, _ = np.histogram(th, bins=bins)
    ax.bar((bins[:-1] + bins[1:]) / 2, h, width=np.diff(bins), alpha=0.7, label=name, linewidth=0)
ax.set_thetamin(0); ax.set_thetamax(180); ax.set_yticks([])
ax.set_xticks(np.deg2rad([0, 30, 60, 90, 120, 150, 180])); ax.set_xticklabels(["0°", "30°", "60°", "90°", "120°", "150°", "180°"])
ax.set_title(f"Angle of each J-lens vector to the vocabulary mean, layer {L} (all 50257 tokens)", pad=14)
ax.legend(loc="lower left", bbox_to_anchor=(0.85, 0.0))
fig.savefig(f"{S}/v7_polar_L{L}.png")

# ---- v8: literal 2-D linear projection of unit directions (plane = mean direction + top PC orthogonal to it)
sub = D["sub"]; pr, pc = D[f"L{L}_proj_raw_mu"], D[f"L{L}_proj_cen_mu"]
ur = pr / D[f"L{L}_raw_norm"][sub][:, None]; uc = pc / D[f"L{L}_cen_norm"][sub][:, None]
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True, subplot_kw=dict(aspect="equal"))
for ax, u, col, name in [(a, ur, MAROON, "raw"), (b, uc, SLATE, "centered")]:
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color=GRAY, lw=0.8))
    ax.scatter(u[:, 0], u[:, 1], s=5, alpha=0.3, color=col, linewidths=0)
    ax.axhline(0, color=GRAY, lw=0.6); ax.axvline(0, color=GRAY, lw=0.6)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("along the vocabulary-mean direction"); ax.set_title(f"{name} unit directions, layer {L}")
a.set_ylabel("along the top PC orthogonal to the mean")
fig.suptitle("Linear projection of unit J-lens vectors onto one fixed 2-D plane (4000 random tokens; circle = unit length)")
fig.savefig(f"{S}/v8_projection_L{L}.png")
print("done")
