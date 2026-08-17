"""C2c — Directed modulation is J-space-specific (paper appendix
"Directed modulation affects the J-space more than other representations").

For each property category:
  neutral:   "Here is a sentence: {stim}"            (baseline)
  imagine:   "Imagine that the following ... {stim}" (instruction)
  real:      neutral framing, stimulus genuinely has the property

Measures at stimulus positions x band layers:
  lens:  label-token lens log-softmax score
  probe: projection on a mean-difference property probe with its J-space
         component (gradient pursuit k=25) projected out

Expectation: imagine moves lens >> probe; real moves probe.

Usage: python scripts/c2c_modulation_privilege.py --lens runs/smollm2-135m-it/lens.pt
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import transformers

import jlens

from tinyjlens.lensops import LensKit
from tinyjlens.prompts import build_raw, variant_token_ids

PRESENT = ["The dog runs across the yard.", "She walks to the store every day.",
           "He plays the piano in the evening.", "The children laugh at the clown.",
           "Birds sing in the morning.", "The chef cooks dinner for the guests.",
           "My brother drives a red car.", "The teacher explains the lesson.",
           "Rain falls on the roof.", "The cat sleeps on the sofa.",
           "Farmers grow wheat in the fields.", "The train arrives at noon."]
PAST = ["The dog ran across the yard.", "She walked to the store yesterday.",
        "He played the piano last night.", "The children laughed at the clown.",
        "Birds sang in the morning.", "The chef cooked dinner for the guests.",
        "My brother drove a red car.", "The teacher explained the lesson.",
        "Rain fell on the roof.", "The cat slept on the sofa.",
        "Farmers grew wheat in the fields.", "The train arrived at noon."]
ENGLISH = ["The sun set behind the hills and golden light covered the fields.",
           "Once upon a time there was a small village by the sea.",
           "The market was full of people buying bread and cheese.",
           "She opened the window and looked at the quiet street below.",
           "The old man told stories about his youth in the mountains.",
           "A cool wind blew through the trees along the river.",
           "The library was silent except for the turning of pages.",
           "Children played in the park until the evening came.",
           "The baker took warm bread out of the oven at dawn.",
           "Boats rested in the harbor under a grey autumn sky.",
           "The garden smelled of roses after the summer rain.",
           "He wrote a long letter to his sister in the city."]
FRENCH = ["Le soleil se couchait derrière les collines et la lumière dorée couvrait les champs.",
          "Il était une fois un petit village au bord de la mer.",
          "Le marché était plein de gens qui achetaient du pain et du fromage.",
          "Elle a ouvert la fenêtre et a regardé la rue tranquille.",
          "Le vieil homme racontait des histoires de sa jeunesse dans les montagnes.",
          "Un vent frais soufflait à travers les arbres le long de la rivière.",
          "La bibliothèque était silencieuse sauf le bruit des pages.",
          "Les enfants jouaient dans le parc jusqu'au soir.",
          "Le boulanger sortait du pain chaud du four à l'aube.",
          "Les bateaux se reposaient dans le port sous un ciel gris.",
          "Le jardin sentait les roses après la pluie d'été.",
          "Il a écrit une longue lettre à sa sœur en ville."]
LOWER_WORDS = ["mountain", "bicycle", "elephant", "wonderful", "garden", "library",
               "morning", "yellow", "thunder", "village", "picture", "kitchen"]
UPPER_WORDS = [w.upper() for w in LOWER_WORDS]

CATEGORIES = {
    "tense": {
        "label_words": ["past"],
        "imagine": "Imagine that the following sentence happened in the past tense. {stim}",
        "neutral": "Here is a sentence. {stim}",
        "neg_stims": PRESENT, "pos_stims": PAST,
    },
    "language": {
        "label_words": ["French"],
        "imagine": "Imagine that the following text is written in French. {stim}",
        "neutral": "Here is a text. {stim}",
        "neg_stims": ENGLISH, "pos_stims": FRENCH,
    },
    "caps": {
        "label_words": ["capital", "caps", "uppercase"],
        "imagine": "Imagine that the following word is written in all capital letters. The word is {stim}.",
        "neutral": "Here is a word. The word is {stim}.",
        "neg_stims": LOWER_WORDS, "pos_stims": UPPER_WORDS,
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--lens", default="runs/smollm2-135m-it/lens.pt")
    ap.add_argument("--band", default=None)
    ap.add_argument("--k-gp", type=int, default=25)
    ap.add_argument("--out", default="runs/c2c_privilege.json")
    args = ap.parse_args()

    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    hf = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda()
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(args.lens)
    kit = LensKit.build(model, lens)
    n_layers = model.n_layers
    if args.band:
        lo, hi = map(int, args.band.split(":"))
    else:
        lo, hi = int(n_layers * 0.4), int(n_layers * 0.87)
    band = [l for l in range(lo, hi + 1) if l in lens.source_layers]

    results = {"band": [band[0], band[-1]], "categories": {}}

    for cat, spec in CATEGORIES.items():
        label_ids = []
        for w in spec["label_words"]:
            label_ids += variant_token_ids(tok, w)
        label_ids = list(dict.fromkeys(label_ids))

        def stim_measures(framing: str, stim: str, probe_dirs=None):
            text = framing.format(stim=stim)
            bp = build_raw(tok, text)
            a, b = bp.find_span(stim if len(stim) < 60 else stim[:60])
            resid = kit.residuals(bp.input_ids, band)
            lens_scores, probe_scores = [], []
            for l in band:
                lg = torch.log_softmax(kit.lens_logits(resid[l][a:b], l), dim=-1)
                lens_scores.append(float(lg[:, label_ids].max(dim=1).values.mean()))
                if probe_dirs is not None:
                    p = probe_dirs[l]
                    probe_scores.append(float((resid[l][a:b] @ p).mean()))
            return (sum(lens_scores) / len(lens_scores),
                    sum(probe_scores) / len(probe_scores) if probe_scores else None)

        # 1) build the probe from real stimuli under neutral framing
        pos_resid = {l: [] for l in band}
        neg_resid = {l: [] for l in band}
        for stims, store in ((spec["pos_stims"], pos_resid), (spec["neg_stims"], neg_resid)):
            for stim in stims:
                text = spec["neutral"].format(stim=stim)
                bp = build_raw(tok, text)
                a, b = bp.find_span(stim if len(stim) < 60 else stim[:60])
                resid = kit.residuals(bp.input_ids, band)
                for l in band:
                    store[l].append(resid[l][a:b].mean(0))
        probe_dirs = {}
        for l in band:
            diff = torch.stack(pos_resid[l]).mean(0) - torch.stack(neg_resid[l]).mean(0)
            _, _, recon = kit.gradient_pursuit(diff, l, args.k_gp)
            ortho = diff - recon
            probe_dirs[l] = ortho / ortho.norm().clamp_min(1e-8)

        # 2) three conditions on the negative stimuli (real = positive stimuli)
        conds = {"neutral": [], "imagine": [], "real": []}
        for stim in spec["neg_stims"]:
            conds["neutral"].append(stim_measures(spec["neutral"], stim, probe_dirs))
            conds["imagine"].append(stim_measures(spec["imagine"], stim, probe_dirs))
        for stim in spec["pos_stims"]:
            conds["real"].append(stim_measures(spec["neutral"], stim, probe_dirs))

        import statistics
        base_lens = [x[0] for x in conds["neutral"]]
        base_probe = [x[1] for x in conds["neutral"]]
        mu_l, sd_l = statistics.mean(base_lens), statistics.stdev(base_lens)
        mu_p, sd_p = statistics.mean(base_probe), statistics.stdev(base_probe)

        def z(vals, mu, sd):
            zs = [(v - mu) / (sd + 1e-9) for v in vals]
            return (statistics.mean(zs),
                    statistics.stdev(zs) / max(len(zs) - 1, 1) ** 0.5 if len(zs) > 1 else 0.0)

        summary = {}
        for cond in ("imagine", "real"):
            lz = z([x[0] for x in conds[cond]], mu_l, sd_l)
            pz = z([x[1] for x in conds[cond]], mu_p, sd_p)
            summary[cond] = {"lens_z": [round(v, 2) for v in lz],
                             "probe_z": [round(v, 2) for v in pz]}
        results["categories"][cat] = summary
        print(cat, summary)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
