import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S = str(__import__("pathlib").Path(__file__).resolve().parent)
CREAM = "#fbfbf9"; MAROON = "#800000"; SLATE = "#33658a"; GRAY = "#8a8580"
plt.rcParams.update({"figure.facecolor": CREAM, "axes.facecolor": CREAM, "savefig.facecolor": CREAM, "savefig.dpi": 200})
D = np.load(f"{S}/geom.npz"); L = 8
rng = np.random.default_rng(1)
th_r = np.arccos(np.clip(D[f"L{L}_cos_mu_raw"], -1, 1)); th_c = np.arccos(np.clip(D[f"L{L}_cos_mu_cen"], -1, 1))
N = 40
pick = rng.choice(len(th_r), N, replace=False)
sign = rng.choice([-1, 1], N)          # the sign of the off-mean part is arbitrary in 2-D; drawn at random

def draw(ax, th, col, full):
    for t, s in zip(th[pick], sign):
        t = t * (s if full else 1)
        ax.annotate("", xy=(np.cos(t), np.sin(t)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col, alpha=0.65, lw=1.3, mutation_scale=11))
    ax.plot(0, 0, "o", color="black", ms=4)
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15 if full else -0.15, 1.15); ax.set_aspect("equal"); ax.set_axis_off()

for full, name in [(False, "half"), (True, "full")]:
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 5 if full else 3), constrained_layout=True)
    draw(a, th_r, MAROON, full); draw(b, th_c, SLATE, full)
    a.set_title("raw J-lens directions"); b.set_title("after subtracting the vocabulary mean")
    fig.savefig(f"{S}/v9_arrows_{name}_L{L}.png")
print("done")
