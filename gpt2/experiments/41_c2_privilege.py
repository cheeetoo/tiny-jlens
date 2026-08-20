"""C2 privilege — instructions write to the J-space specifically.

The dissociation (paper §A "directed modulation affects the J-space more
than other representations"): a think-about-w instruction should move w's
LENS measure at the transcription tokens, while leaving a J-orthogonalized
probe for w unmoved; a real stimulus (a sentence actually containing w)
should move the probe strongly.

Probe for w: mean-difference of residuals over sentences containing w
(3 frames), centered across the 12 words; J-orthogonalized by removing its
centered-gauge gradient-pursuit component (k=16). All measures are paired
per (word, sentence) against the no-instruction baseline.

Run:  python experiments/41_c2_privilege.py [model]
"""

import json
import sys

sys.path.insert(0, "/tiny-jlens/gpt2")

import torch

import core
import pools

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
kit = core.Kit(MODEL)
tok = kit.tokenizer

FRAMES = [
    "She painted a picture of the {w} last summer.",
    "Everyone stopped to look at the {w} near the road.",
    "His favorite photograph showed a {w} in the evening light.",
    "The postcard on the shelf showed a {w} at sunset.",
    "The children drew a {w} on the classroom whiteboard.",
    "An old book on the table had a {w} on its cover.",
]
REAL_FRAME = "The {w} appeared suddenly at the edge of town."
LAYS = [l for l in kit.layers if 0.55 * kit.n_layers <= l <= 1.0 * kit.n_layers]


def word_ids(w):
    out = []
    for x in {w, w.capitalize()}:
        for f in (" " + x, x):
            ids = tok(f, add_special_tokens=False)["input_ids"]
            if len(ids) == 1 and ids[0] not in out:
                out.append(ids[0])
    return out


# single-token filter, as everywhere else (a word with no single-token form
# has no lens readout — its logsumexp would be -inf)
WORDS = [w for w in pools.C2_WORDS if word_ids(w)]


# ---- probes ----
resid_by_w = {}
for w in WORDS:
    hs = {l: [] for l in LAYS}
    for fr in FRAMES:
        ids = kit.encode(fr.format(w=w))
        r = kit.residuals(ids, LAYS)
        for l in LAYS:
            hs[l].append(r[l][-1])
    resid_by_w[w] = {l: torch.stack(v).mean(0) for l, v in hs.items()}
grand = {l: torch.stack([resid_by_w[w][l] for w in WORDS]).mean(0) for l in LAYS}

probes, probes_orth, jshare = {}, {}, []
for w in WORDS:
    probes[w], probes_orth[w] = {}, {}
    for l in LAYS:
        p = resid_by_w[w][l] - grand[l]
        _, _, recon = core.gradient_pursuit(kit, p, l, 16, centered=True)
        probes[w][l] = p / p.norm()
        q = p - recon
        probes_orth[w][l] = q / q.norm()
        jshare.append((recon.norm() ** 2 / p.norm() ** 2).item())
print(f"probe J-component share: median {sorted(jshare)[len(jshare)//2]:.1%}")


def span_measures(text: str, sentence: str, w: str):
    """(lens_logprob, orth_proj) of w over the transcription span: the lens
    measure is best-POSITION best-layer (a held word occupies specific slots,
    so span-averaging dilutes it — measured 2026-08-19: mean-over-span deltas
    are ~0/slightly negative on long sentences while best-position deltas are
    uniformly positive, matching 40's rank measure); the probe measure stays
    mean over span and layers (a property probe is diffuse)."""
    start = text.find(sentence, text.find(sentence) + 1)
    if start < 0:
        start = text.find(sentence)  # real condition: sentence appears once
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
    ids = enc["input_ids"].cuda()
    span = [i for i, (a, b) in enumerate(enc["offset_mapping"][0].tolist())
            if a >= start and b <= start + len(sentence) and b > a]
    resid = kit.residuals(ids, LAYS)
    vids = word_ids(w)
    lens_lp = max(
        torch.logsumexp(kit.lens_logits(resid[l][span], l).log_softmax(-1)[:, vids],
                        dim=-1).max().item()
        for l in LAYS)
    orth = sum(float((resid[l][span] @ probes_orth[w][l]).mean())
               for l in LAYS) / len(LAYS)
    return lens_lp, orth


rows = []
for w in WORDS:
    for s in pools.C2_SENTENCES:
        rec = dict(word=w, sentence=s)
        for cond in ("think", "dont", "base"):
            text = pools.C2_TEMPLATES[cond].format(s=s, w=w, s_open=s)
            rec[cond] = span_measures(text, s, w)
        real_s = REAL_FRAME.format(w=w)
        text = pools.C2_TEMPLATES["base"].format(s=real_s, w=w, s_open=real_s)
        rec["real"] = span_measures(text, real_s, w)
        rows.append(rec)

def paired_z(cond):
    d = torch.tensor([r[cond][0] - r["base"][0] for r in rows])
    dz = torch.tensor([r[cond][1] - r["base"][1] for r in rows])
    n = len(rows) ** 0.5
    return (d.mean() / (d.std() / n)).item(), (dz.mean() / (dz.std() / n)).item()

print(f"n={len(rows)} paired (word, sentence) trials; paired t vs base "
      f"(mean/(sd/sqrt n) — NOT a baseline-SD z; magnitudes in analyze.py) "
      f"(lens log-prob | J-orthogonalized probe):")
for cond in ("think", "dont", "real"):
    zl, zp = paired_z(cond)
    print(f"  {cond:6s} lens t {zl:+6.1f}    orth-probe t {zp:+6.1f}")
json.dump(rows, open(f"/tiny-jlens/gpt2/results/c2_privilege_{MODEL}.json", "w"))
