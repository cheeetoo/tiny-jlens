"""C2 privilege, property form (the paper's own design, ported): a header
CLAIMS an English sentence is French. The claim should raise French in the
J-lens at the sentence tokens, while leaving a J-orthogonalized
French-vs-English property probe unmoved; a real French sentence moves the
probe. (Word-level materials cannot make this dissociation — a mentioned
word contaminates every channel; see 41.)

Run:  python experiments/42_imagine.py [model]
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
LAYS = [l for l in kit.layers if 0.55 * kit.n_layers <= l <= 1.0 * kit.n_layers]

ENGLISH = [
    "The kitchen window was open all afternoon.",
    "A gray cat slept on the warm stone steps.",
    "The letters were stacked neatly on the desk.",
    "Rain kept falling on the empty market square.",
    "The last train left the station before dark.",
    "Her brother fixed the fence behind the barn.",
]
FRENCH = [
    "La fenêtre de la cuisine est restée ouverte tout l'après-midi.",
    "Un chat gris dormait sur les marches de pierre chaude.",
    "Les lettres étaient posées sur le bureau près de la porte.",
    "La pluie tombait sur la place vide du marché.",
    "Le dernier train a quitté la gare avant la nuit.",
    "Son frère a réparé la clôture derrière la grange.",
]
NEUTRAL = "Here is a sentence: {s}"
CLAIM = "The following sentence is written in French: {s}"

# ---- property probe: French vs English, from held-out C5a passages ----
fr_pass = [p["text"] for p in pools.c5_passages() + pools.EXTRA_PASSAGES
           if p["category"] == "French"]
en_pass = ["The old man walked through the forest every morning while the birds sang in the trees,",
           "The city woke slowly under the morning sun as people crossed the square with baskets of bread,",
           "The sea was calm that morning, and the fishing boats returned to the harbor one by one,"]


def mean_final(texts):
    out = {l: [] for l in LAYS}
    for t in texts:
        r = kit.residuals(kit.encode(t), LAYS)
        for l in LAYS:
            out[l].append(r[l][-1])
    return {l: torch.stack(v).mean(0) for l, v in out.items()}


mf, me = mean_final(fr_pass), mean_final(en_pass)
probe, probe_orth = {}, {}
for l in LAYS:
    p = mf[l] - me[l]
    _, _, recon = core.gradient_pursuit(kit, p, l, 16, centered=True)
    probe[l] = p / p.norm()
    q = p - recon
    probe_orth[l] = q / q.norm()
    print(f"L{l}: French-probe J-share {recon.norm()**2 / p.norm()**2:.1%}")

fr_ids = [kit.tok_id(" French"), kit.tok_id("French")]


def measures(text: str, sentence: str):
    start = text.index(sentence)
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
    ids = enc["input_ids"].cuda()
    span = [i for i, (a, b) in enumerate(enc["offset_mapping"][0].tolist())
            if a >= start and b > a]
    resid = kit.residuals(ids, LAYS)
    lens_lp = max(
        torch.logsumexp(kit.lens_logits(resid[l][span], l).log_softmax(-1)[:, fr_ids],
                        dim=-1).mean().item() for l in LAYS)
    orth = sum(float((resid[l][span] @ probe_orth[l]).mean()) for l in LAYS) / len(LAYS)
    full = sum(float((resid[l][span] @ probe[l]).mean()) for l in LAYS) / len(LAYS)
    return lens_lp, orth, full


rows = []
for s_en, s_fr in zip(ENGLISH, FRENCH):
    rows.append(dict(
        neutral=measures(NEUTRAL.format(s=s_en), s_en),
        claim=measures(CLAIM.format(s=s_en), s_en),
        real=measures(NEUTRAL.format(s=s_fr), s_fr)))

def paired_z(cond, i):
    d = torch.tensor([r[cond][i] - r["neutral"][i] for r in rows])
    return (d.mean() / (d.std() / len(rows) ** 0.5)).item()

print(f"\nn={len(rows)} sentences; paired z vs neutral header "
      f"(lens 'French' | J-orth property probe | full probe):")
for cond in ("claim", "real"):
    print(f"  {cond:6s} lens z {paired_z(cond, 0):+6.1f}   "
          f"orth-probe z {paired_z(cond, 1):+6.1f}   full-probe z {paired_z(cond, 2):+6.1f}")
json.dump(rows, open(f"/tiny-jlens/gpt2/results/c2_imagine_{MODEL}.json", "w"))
