"""One figure for criterion 4 (flexible generalization).  Run after run.py.

A  case study:  France->China across the four country functions, one fixed swap (Fig 18)
B  swap success: raw 192 vs the capable subset, alpha=1/2, vs the paper (Fig 19 left)
C  by category:  capable-subset success and workspace loading side by side
D  4x4 grids:    two functions that follow the swap, two that do not (Fig 68)
E  loading:      per-cell loading vs swap success — the link the paper reports (Fig 19 right)
F  summary:      headline numbers vs the paper
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
out = json.load(open(f"{HERE}/results/results.json"))
sw = out["swap"]
CATS = ["countries", "months", "animals", "numbers"]
CATCOL = dict(countries="tab:blue", months="tab:green", animals="tab:orange", numbers="tab:red")
clean = lambda r: r["distinct"] and not r["echo"]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0, 0
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return p, max(0.0, c - h), min(1.0, c + h)


fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))

# ---- A: case study France->China across country functions ------------------
ax = axes[0, 0]
case = out["case"]
funcs = [c["function"] for c in case]
y = np.arange(len(funcs))[::-1]
for yi, c in zip(y, case):
    col = "tab:blue" if c["follows"] else "tab:gray"
    ax.plot([c["target_rank_clean"], c["target_rank_swapped"]], [yi, yi], "-", color=col, lw=1.5, zorder=1)
    ax.scatter(c["target_rank_clean"], yi, s=40, color="tab:gray", zorder=3)
    ax.scatter(c["target_rank_swapped"], yi, s=40, color=col, zorder=3, marker="D")
    ax.text(1.4, yi + 0.16, f"→{c['swapped_top1'].strip()}", fontsize=8, color=col, va="bottom")
ax.set_yticks(y); ax.set_yticklabels(funcs)
ax.set_xscale("log"); ax.set_xlim(0.7, 3000)
ax.axvline(1, color="k", ls="--", lw=0.7)
ax.set_xlabel(f"rank of {case[0]['target']}'s answer  (◇ = after swap; left = better)")
nfollow = sum(c["follows"] for c in case)
ax.set_title(f"A  case study: one swap {case[0]['source']}→{case[0]['target']}, {nfollow}/4 functions follow (Fig 18)",
             loc="left", fontsize=10)

# ---- B: overall swap success, raw vs capable, vs paper ---------------------
ax = axes[0, 1]
def succ(flt, al):
    rows = [r for r in sw if flt(r)]
    k = sum(r["subadd"][str(al)]["hit"] for r in rows)
    return k, len(rows), *wilson(k, len(rows))
groups = [("all 192\n(raw)", lambda r: True),
          ("capable\n(target gated)", lambda r: r["target_gated"] and clean(r))]
x = np.arange(len(groups)); w = 0.36
for j, al in enumerate([1.0, 2.0]):
    ps, los, his, ns = [], [], [], []
    for _, flt in groups:
        k, n, p, lo, hi = succ(flt, al); ps.append(p); los.append(p - lo); his.append(hi - p); ns.append((k, n))
    bars = ax.bar(x + (j - 0.5) * w, ps, w, label=f"α={al:.0f}", color=["tab:blue", "tab:cyan"][j])
    ax.errorbar(x + (j - 0.5) * w, ps, yerr=[los, his], fmt="none", ecolor="k", capsize=3)
    for xi, p, (k, n) in zip(x + (j - 0.5) * w, ps, ns):
        ax.text(xi, p + 0.02, f"{k}/{n}", ha="center", fontsize=8)
# paper reference lines (its raw 192 rates)
ax.axhline(76 / 192, color="tab:blue", ls=":", lw=1); ax.text(1.5, 76/192 + 0.005, "paper α1 40%", fontsize=7, color="tab:blue", ha="right")
ax.axhline(101 / 192, color="tab:cyan", ls=":", lw=1); ax.text(1.5, 101/192 + 0.005, "paper α2 53%", fontsize=7, color="tab:cyan", ha="right")
ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups])
ax.set_ylabel("swap → target answer at top-1"); ax.set_ylim(0, 0.62); ax.legend(fontsize=8, loc="upper left")
ax.set_title("B  swap success: raw grid vs capable subset (Fig 19 left)", loc="left", fontsize=10)

# ---- C: by category — capable-subset success and loading -------------------
ax = axes[0, 2]
lo = out["loading"]
cat_succ, cat_n, cat_load = {}, {}, {}
for cat in CATS:
    rows = [r for r in sw if r["category"] == cat and r["target_gated"] and clean(r)]
    k = sum(r["subadd"]["1.0"]["hit"] for r in rows)
    cat_succ[cat] = k / len(rows) if rows else 0; cat_n[cat] = (k, len(rows))
    vals = [x["loading"] for x in lo if x["category"] == cat]
    cat_load[cat] = sum(vals) / len(vals)
x = np.arange(len(CATS))
ax.bar(x, [cat_succ[c] for c in CATS], color=[CATCOL[c] for c in CATS], alpha=0.75)
for xi, c in zip(x, CATS):
    k, n = cat_n[c]
    ax.text(xi, cat_succ[c] + 0.02, f"{k}/{n}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(CATS, fontsize=9); ax.set_ylim(0, 0.85)
ax.set_ylabel("swap success (capable subset, α=1)")
ax2 = ax.twinx()
ax2.plot(x, [cat_load[c] for c in CATS], "k--o", lw=1.5, ms=5)
ax2.set_ylabel("workspace loading (cos)", color="k"); ax2.set_ylim(0, 0.20)
ax.set_title("C  by category: success (bars) vs loading (line)", loc="left", fontsize=10)

# ---- D: 4x4 grids, two functions that follow, two that do not (Fig 68) -----
ax = axes[1, 0]; ax.axis("off")
grid_funcs = [("countries", "language"), ("numbers", "square"),
              ("months", "next_month"), ("numbers", "successor")]
cats_by_name = {c["name"]: c for c in json.load(open("/tiny-jlens/ref/jacobian-lens/data/experiments/flexible-generalization.json"))["categories"]}
bykey = {(r["category"], r["function"], r["source"], r["target"]): r for r in sw}
sub = fig.add_gridspec(2, 2, left=0.045, right=0.31, top=0.42, bottom=0.06, hspace=0.75, wspace=0.55)
for gi, (cat, fn) in enumerate(grid_funcs):
    a = fig.add_subplot(sub[gi // 2, gi % 2])
    args = cats_by_name[cat]["args"]
    M = np.full((4, 4), np.nan)  # NaN (masked, grey) = diagonal baseline or non-scored pair
    for i, s in enumerate(args):
        for jx, t in enumerate(args):
            if s == t:
                continue
            r = bykey.get((cat, fn, s, t))
            if r is not None and clean(r):
                M[i, jx] = 1.0 if r["subadd"]["1.0"]["hit"] else 0.0
    cmap = matplotlib.colormaps["RdYlGn"].copy(); cmap.set_bad("0.85")
    a.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=1, aspect="equal")
    a.set_xticks(range(4)); a.set_yticks(range(4))
    a.set_xticklabels([x[:4] for x in args], fontsize=6, rotation=90)
    a.set_yticklabels([x[:4] for x in args], fontsize=6)
    k = int(np.nansum(M[(M == 1)])); n = int(np.sum((M == 0) | (M == 1)))
    a.set_title(f"{cat[:4]}/{fn}\n{k}/{n} hit", fontsize=7)
axes[1, 0].set_title("D  grids: retrieval funcs follow (green), successor-type don't (red)  (Fig 68)",
                     loc="left", fontsize=10)

# ---- E: loading vs success, per cell (Fig 19 right) ------------------------
ax = axes[1, 1]
loadmap = {(x["category"], x["function"], x["arg"]): x["loading"] for x in lo}
cell = {}
for r in sw:
    if r["target_gated"] and clean(r):
        key = (r["category"], r["function"], r["source"])
        c = cell.setdefault(key, [0, 0]); c[0] += r["subadd"]["1.0"]["hit"]; c[1] += 1
for key, (k, n) in cell.items():
    jit = (hash(key) % 100 - 50) / 3000
    ax.scatter(loadmap[key], k / n + jit, s=28, color=CATCOL[key[0]], alpha=0.8,
               edgecolor="k", linewidth=0.3)
handles = [plt.Line2D([], [], marker="o", ls="", color=CATCOL[c], label=c) for c in CATS]
ax.legend(handles=handles, fontsize=7, loc="center right")
ax.set_xlabel("workspace loading (cos)"); ax.set_ylabel("swap success for that source cell")
ax.set_ylim(-0.08, 1.08)
ax.set_title("E  loading does NOT predict success here (see F)", loc="left", fontsize=10)

# ---- F: summary text -------------------------------------------------------
ax = axes[1, 2]; ax.axis("off")
def rate(flt, al):
    rows = [r for r in sw if flt(r)]
    return sum(r["subadd"][str(al)]["hit"] for r in rows), len(rows)
raw1 = rate(lambda r: True, 1.0); raw2 = rate(lambda r: True, 2.0)
cap1 = rate(lambda r: r["target_gated"] and clean(r), 1.0)
xs = [loadmap[k] for k in cell]; ys = [cell[k][0] / cell[k][1] for k in cell]
def spearman(xs, ys):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for i, idx in enumerate(o): r[idx] = i
        return r
    rx, ry = rank(xs), rank(ys); n = len(xs)
    return 1 - 6 * sum((a - b) ** 2 for a, b in zip(rx, ry)) / (n * (n * n - 1))
lines = [
    "gpt2-small (124M base)      vs   paper (Sonnet 4.5)",
    "",
    f"case study: {sum(c['follows'] for c in case)}/4 country functions follow one swap",
    "",
    f"swap → target answer at top-1 (subtract-and-add):",
    f"   raw 192 grid   α1 {raw1[0]}/{raw1[1]} ({raw1[0]/raw1[1]:.0%})    paper 76/192 (40%)",
    f"                  α2 {raw2[0]}/{raw2[1]} ({raw2[0]/raw2[1]:.0%})    paper 101/192 (53%)",
    f"   capable subset α1 {cap1[0]}/{cap1[1]} ({cap1[0]/cap1[1]:.0%})     (≈ paper's 40%)",
    "",
    "α=2 overshoots (names the argument) → α2 ≤ α1,",
    "   opposite to the paper.",
    "",
    "loading: countries/animals high, numbers lowest",
    f"   (matches paper's ordering), but cell-level",
    f"   loading→success Spearman = {spearman(xs, ys):+.2f} (n={len(xs)}):",
    "   success tracks function TYPE, not loading —",
    "   retrieval follows, successor-type doesn't.",
    "",
    f"floor (bare templates): {out['floor']['cap']}/{out['floor']['n']} answerable",
]
ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9, transform=ax.transAxes)

fig.suptitle("Criterion 4 — flexible generalization — gpt2-small", y=1.00, fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.99])
os.makedirs(f"{HERE}/figures", exist_ok=True)
fig.savefig(f"{HERE}/figures/criterion4.png", dpi=150, bbox_inches="tight")
print("wrote figures/criterion4.png")
