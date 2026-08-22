"""One figure for criterion 5.  Run after run.py.

A  S2a, light ablation: per task, J-space ablation vs matched-norm random control (normalized to
   clean).  The flexible task collapses under the J ablation but not the norm-matched control;
   automatic tasks survive both -- the selectivity signature.
B  S2a, strength curve: J-ablated score (solid) and matched-norm random (dashed) across
   light/medium/heavy.  Two-hop collapses fastest; automatic tasks last.
C  S2b: the same language swap under two tasks, by strength.  The deliberate report follows the
   swap where the automatic continuation keeps its own language.
D  S1: fraction of activation variance captured by the top-K J-space directions vs a same-size
   random dictionary (a structured subframe, not a variance-dominating subspace).
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
STR = ["light", "medium", "heavy"]
TASK_COLOR = {"two_hop": "tab:red", "one_hop": "tab:orange",
              "induction": "tab:blue", "pretrain_match": "tab:green"}
TASK_LABEL = {"two_hop": "two-hop\n(flexible)", "one_hop": "one-hop\nrecall",
              "induction": "induction\n(automatic)", "pretrain_match": "next-token\n(automatic)"}

rows = out["battery"]["acc"] + [out["battery"]["pretrain"]]
fig, axes = plt.subplots(2, 3, figsize=(17, 9.2))

# ---- A: light ablation, J vs matched-norm random, per task
ax = axes[0][0]
x = np.arange(len(rows))
jj = [r["light_J"] for r in rows]
rr = [r["light_R"] for r in rows]
ax.bar(x - 0.2, jj, 0.38, color=[TASK_COLOR[r["task"]] for r in rows], label="J-space ablation")
ax.bar(x + 0.2, rr, 0.38, color="lightgray", edgecolor="gray", label="matched-norm random")
ax.axhline(1.0, color="k", lw=0.6, ls=":")
for i, r in enumerate(rows):
    ax.text(i, max(r["light_J"], r["light_R"]) + 0.03, f"gap {r['light_R']-r['light_J']:.2f}",
            ha="center", fontsize=8, color="k")
ax.set_xticks(x); ax.set_xticklabels([TASK_LABEL[r["task"]] for r in rows], fontsize=8)
ax.set_ylim(0, 1.15); ax.set_ylabel("task score (clean = 1.0)")
ax.legend(fontsize=8, loc="lower left")
ax.set_title("A  light ablation: the J-subspace specifically breaks the flexible task", loc="left", fontsize=10)

# ---- B: strength curve, J (solid) vs random (dashed)
ax = axes[0][1]
xs = np.arange(len(STR))
for r in rows:
    c = TASK_COLOR[r["task"]]
    ax.plot(xs, [r[f"{s}_J"] for s in STR], "-o", color=c, lw=2, label=TASK_LABEL[r["task"]].replace("\n", " "))
    ax.plot(xs, [r[f"{s}_R"] for s in STR], "--", color=c, lw=1, alpha=0.6)
ax.set_xticks(xs); ax.set_xticklabels([f"{s}\n{out['strengths'][s]}" for s in STR], fontsize=8)
ax.set_ylim(0, 1.05); ax.set_ylabel("task score (clean = 1.0)")
ax.set_xlabel("ablation strength (band layers)")
ax.legend(fontsize=7.5, loc="upper right")
ax.set_title("B  solid = J-space ablation, dashed = matched-norm random", loc="left", fontsize=10)

# ---- C: S2b, report flip vs continuation hold, by alpha
ax = axes[0][2]
lg = out["language"]
alphas = lg["alphas"]
rf = [np.mean([t["report_flip"] for t in lg["trials"] if t["alpha"] == a]) for a in alphas]
ch = [np.mean([t["cont_hold"] for t in lg["trials"] if t["alpha"] == a]) for a in alphas]
ax.plot(alphas, rf, "-o", color="tab:purple", lw=2, label="deliberate report\nflips to swapped language")
ax.plot(alphas, ch, "-s", color="tab:blue", lw=2, label="automatic continuation\nkeeps its own language")
ax.set_ylim(0, 1.05); ax.set_xlabel("swap strength  α")
ax.set_ylabel("fraction of swaps")
n = len([t for t in lg["trials"] if t["alpha"] == alphas[0]])
ax.legend(fontsize=8, loc="center left")
ax.set_title(f"C  same language swap, two tasks (n={n})", loc="left", fontsize=10)

# ---- D: S1, variance vs K, J vs random dictionary
ax = axes[1][0]
cap = out["capacity"]
K = np.arange(1, cap["kmax"] + 1)
for pl in cap["per_layer"]:
    ax.plot(K, np.array(pl["frac_J"]) * 100, "-", label=f"L{pl['layer']} J-space")
for pl in cap["per_layer"][:1]:
    ax.plot(K, np.array(pl["frac_R"]) * 100, "--", color="gray", label="random dict (same size)")
ax.axvline(25, color="k", lw=0.6, ls=":")
ax.set_xlabel("K (number of directions)"); ax.set_ylabel("% of activation variance captured")
ax.legend(fontsize=8, loc="lower right")
occ = ", ".join(f"L{pl['layer']} K={pl['occupancy']}" for pl in cap["per_layer"])
ax.set_title(f"D  the J-space is a small structured slice (occupancy {occ})", loc="left", fontsize=9)

# ---- E: vs-paper summary + floor, as text
ax = axes[1][1]; ax.axis("off")
b = out["battery"]

def light_sd(task):
    r = next(x for x in rows if x["task"] == task)
    return r["light_R"] - r["light_J"]


txt = [
    "gpt2-small (124M base)        vs   paper (Sonnet 4.5)",
    "",
    "S2a  J-space ablation is selective (light ablation):",
    "  selective damage = random - J score",
    f"    two-hop  (flexible)   +{light_sd('two_hop'):.2f}   multihop -> ~0 (Fig 22)",
    f"    one-hop  (recall)     +{light_sd('one_hop'):.2f}   recall: intermediate",
    f"    induction(automatic)  +{light_sd('induction'):.2f}   parsing/extract survive",
    f"    next-token(automatic) +{light_sd('pretrain_match'):.2f}   pretrain match preserved",
    "",
    "S2b  same language latent, two tasks (alpha=1.5):",
    f"    report flips {rf[alphas.index(1.5)]:.0%}   continuation holds {ch[alphas.index(1.5)]:.0%}",
    "    paper: report follows ~always, continuation unmoved",
    "    (ours: attenuated -- see PROTOCOL)",
    "",
    "S1  small subset: top-25 J-space ~ 30% of variance",
    "    (paper: excess over random <10%; occupancy ~25)",
]
ax.text(0.0, 1.0, "\n".join(txt), va="top", ha="left", family="monospace", fontsize=9.5)

# ---- F: floor note
ax = axes[1][2]; ax.axis("off")
fl = out["floor"]
ftxt = ["floor  line-length counting (§3.5.1, Fig 21)", "",
        "count token in band J-space top-25:"]
for cond in ("none", "direct", "letter"):
    ftxt.append(f"    {cond:7s} {fl[cond]['count_in_band']}/{fl[cond]['n']} passages")
ftxt += ["", "gpt2-small cannot track character counts;",
         "the count never enters the J-space under any",
         "task (as with the criterion-2 math floor).", "",
         "dropped, no base-model form:",
         "  - experiential reports (§3.5.3)",
         "  - naming-vs-avoiding (App., instruction-only)"]
ax.text(0.0, 1.0, "\n".join(ftxt), va="top", ha="left", family="monospace", fontsize=9.5)

fig.suptitle("Criterion 5 — selectivity — gpt2-small", y=1.005, fontsize=13)
fig.tight_layout()
os.makedirs(f"{HERE}/figures", exist_ok=True)
fig.savefig(f"{HERE}/figures/criterion5.png", dpi=150, bbox_inches="tight")
print("wrote figures/criterion5.png")
