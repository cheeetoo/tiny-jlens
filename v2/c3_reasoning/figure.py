"""One figure for criterion 3 (internal reasoning).  Run after run.py.

A  readout:   median lens rank of the unspoken intermediate vs controls, per layer
B  case:      France->China, clean vs swapped top-5 next-token log-probs (Fig 13)
C  swap:      swap_answer's output rank, no-swap vs after-swap (Fig 15 left)
D  depth:     effect of swapping intermediate vs answer, per layer (Fig 15 right)
E  privilege: swap_answer top-1 per component, matched magnitude (Fig 16)
F  summary:   headline numbers vs the paper
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
out = json.load(open(f"{HERE}/results/results.json"))
band = out["band"]
LAYERS = list(range(11))


def gk(d, l):  # get by layer key (JSON stringifies int keys)
    return d[str(l)] if str(l) in d else d[l]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0, 0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return p, max(0.0, c - h), min(1.0, c + h)


fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))

# ---- A: readout, median rank per layer -----------------------------------
ax = axes[0, 0]
styles = dict(intermediate=("tab:blue", "-", "unspoken intermediate"),
              answer=("tab:green", "--", "answer (motor)"),
              arg=("tab:gray", ":", "surface cue (echo)"),
              null=("tab:red", "-.", "random country (null)"))
for name, (col, ls, lab) in styles.items():
    med = [np.median([gk(r["ranks"][name], l) for r in out["readout"]]) for l in LAYERS]
    ax.plot(LAYERS, med, ls, color=col, label=lab, lw=2)
ax.axvspan(band[0] - 0.5, band[-1] + 0.5, color="gold", alpha=0.15, zorder=0)
ax.set_yscale("log"); ax.invert_yaxis()
ax.axhline(10, color="k", lw=0.6, ls=":")
ax.set_xlabel("layer"); ax.set_ylabel("median lens rank (log, top = better)")
ax.legend(fontsize=8, loc="lower left")
ax.set_title("A  the unspoken intermediate surfaces in the band", loc="left", fontsize=10)

# ---- B: case study, the answer flip (source_answer vs target_answer) ------
ax = axes[0, 1]
c = out["case"]
if c:
    al = c["answer_lp"]
    labels = [f"{c['source_answer']}\n(clean answer)", f"{c['target_answer']}\n(swap target)"]
    clean = [al["source"]["clean"], al["target"]["clean"]]
    swapped = [al["source"]["swapped"], al["target"]["swapped"]]
    x = np.arange(2)
    ax.bar(x - 0.2, clean, width=0.38, color="tab:gray", label="clean")
    ax.bar(x + 0.2, swapped, width=0.38, color="tab:blue", label="after swap")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("next-token log-prob")
    ax.legend(fontsize=8, loc="lower center")
    ax.set_title(f"B  case study: swap {c['source']}→{c['target']} flips {c['source_answer']}→{c['target_answer']}",
                 loc="left", fontsize=10)

# ---- C: swap, output rank no-swap vs after-swap --------------------------
ax = axes[0, 2]
sub = out["swap"]
rng = np.random.default_rng(1)
jit = rng.uniform(-0.08, 0.08, len(sub))
for x, dx in zip(sub, jit):
    ax.plot([dx, 1 + dx], [x["before"], x["after"]], color="gray", lw=0.3, alpha=0.25, zorder=1)
ax.scatter(jit, [x["before"] for x in sub], s=6, color="gray", zorder=3, label="no swap")
ax.scatter(1 + jit, [x["after"] for x in sub], s=6, color="tab:blue", zorder=3, label="after swap")
ax.axhline(1, color="k", ls="--", lw=0.7)
ax.set_yscale("log"); ax.invert_yaxis()
ax.set_xlim(-0.3, 1.3); ax.set_xticks([0, 1]); ax.set_xticklabels(["no swap", "after swap"])
ax.set_ylabel("swap_answer's output rank")
top1 = np.mean([x["hit"] for x in sub])
ax.legend(fontsize=8, loc="lower right")
ax.set_title(f"C  intermediate swap redirects the answer ({top1:.0%} top-1, n={len(sub)})", loc="left", fontsize=10)

# ---- D: depth control -----------------------------------------------------
ax = axes[1, 0]
DL = sorted({int(k) for r in out["depth"] for k in r["effect"]["interm"]})
mi = [np.mean([gk(r["effect"]["interm"], l) for r in out["depth"]]) for l in DL]
ma = [np.mean([gk(r["effect"]["answer"], l) for r in out["depth"]]) for l in DL]
si = [np.std([gk(r["effect"]["interm"], l) for r in out["depth"]]) / np.sqrt(len(out["depth"])) for l in DL]
sa = [np.std([gk(r["effect"]["answer"], l) for r in out["depth"]]) / np.sqrt(len(out["depth"])) for l in DL]
ax.axvspan(band[0] - 0.5, band[-1] + 0.5, color="gold", alpha=0.15, zorder=0)
ax.errorbar(DL, mi, yerr=si, color="tab:blue", lw=2, label="swap intermediate")
ax.errorbar(DL, ma, yerr=sa, color="tab:green", lw=2, ls="--", label="swap answer")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("layer of single-layer swap"); ax.set_ylabel("log-prob pushed onto swap_answer")
ax.legend(fontsize=8, loc="upper left")
ax.set_title("D  the intermediate is used before the answer (Fig 15 right)", loc="left", fontsize=10)

# ---- E: privileging -------------------------------------------------------
ax = axes[1, 1]
conds = [("raw_lens", "raw J-lens\nswap", "tab:blue"), ("full", "full\nprobe", "tab:purple"),
         ("jpart", "J-space\ncomponent", "tab:blue"), ("nonj", "non-J\ncomponent", "tab:orange"),
         ("nonj_clamp", "non-J,\nclamped", "tab:orange")]
n = len(out["privilege"])
byfam = defaultdict(list)
for r in out["privilege"]:
    byfam[r["intermediate"]].append(r)
for i, (key, lab, col) in enumerate(conds):
    p, lo, hi = wilson(sum(r[key] for r in out["privilege"]), n)
    ax.bar(i, p, color=col, alpha=0.6, zorder=1)
    ax.errorbar(i, p, yerr=[[max(0, p - lo)], [max(0, hi - p)]], color="k", capsize=3, zorder=2)
    ints = sorted(byfam)
    for j, c2 in enumerate(ints):
        ax.scatter(i + (j - len(ints) / 2) * 0.012, np.mean([r[key] for r in byfam[c2]]),
                   s=5, color="k", alpha=0.4, zorder=3)
    ax.text(i, hi + 0.03, f"{p:.0%}", ha="center", fontsize=9)
ax.set_xticks(range(len(conds))); ax.set_xticklabels([c[1] for c in conds], fontsize=8)
ax.set_ylim(0, 1.12); ax.set_ylabel("swap_answer reaches top-1")
ax.set_title(f"E  only the J-space component carries the effect (n={n})", loc="left", fontsize=10)

# ---- F: summary text ------------------------------------------------------
ax = axes[1, 2]; ax.axis("off")
sw_all = np.mean([x["hit"] for x in out["swap"]])
vf = out["variance_fraction"]
jshare = np.median([x["frac"] for x in vf if x["layer"] == band[-1]])
uns = [r for r in out["readout"] if r["int_out_rank"] >= 10]
int_band = np.mean([min(gk(r["ranks"]["intermediate"], l) for l in band) < 10 for r in uns])
lines = [
    "gpt2-small (124M base)   vs   paper",
    "",
    f"E1  unspoken intermediate in lens     {int_band:.0%}   top-10 at some band layer",
    f"E3  swap → answer flips (top-1)         {sw_all:.0%}      Haiku 54 / Sonnet 70 / Opus 70",
    "E5  swap_answer top-1, matched magnitude:",
    f"      raw J-lens swap                 {np.mean([r['raw_lens'] for r in out['privilege']]):.0%}      raw 60% (Sonnet)",
    f"      J-space component               {np.mean([r['jpart'] for r in out['privilege']]):.0%}      J    61%",
    f"      non-J-space component           {np.mean([r['nonj'] for r in out['privilege']]):.0%}       non-J 28%",
    f"      non-J, J coords clamped         {np.mean([r['nonj_clamp'] for r in out['privilege']]):.0%}       clamp 6%",
    f"      J-space share of probe var      {jshare:.0%}      paper 10-15%",
    "",
    f"floor  paper's own prompts (no frame): {out['floor']['cap']}/{out['floor']['n']} answerable",
]
ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9.5, transform=ax.transAxes)

fig.suptitle("Criterion 3 — internal reasoning — gpt2-small", y=1.00, fontsize=13)
fig.tight_layout()
os.makedirs(f"{HERE}/figures", exist_ok=True)
fig.savefig(f"{HERE}/figures/criterion3.png", dpi=150, bbox_inches="tight")
print("wrote figures/criterion3.png")
