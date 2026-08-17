"""C2 revisited — internal state control at 135M, mirroring the user's LW
protocol (arXiv-of-record: "Internal state control is a general property of
LLMs") side-by-side with the J-lens criterion.

Protocol (per the post):
  think:      Write '{sentence}' Think about {word} while you write the sentence.
  dont_think: Write '{sentence}' Don't think about {word} while you write the sentence.
  mention:    There is such thing as {word}. Write '{sentence}'
  baseline:   Write '{sentence}'
Teacher-forced transcription; measurement at the transcribed-sentence tokens.

Measures, at EVERY layer (0..n-1, motor layers included):
  probe:  cos(residual, concept_vector) where concept_vector = last-position
          residual of "Tell me about {word}." minus the mean over a pool of
          words; null = distribution of cosines for unrelated concept vectors
          at the same positions (95% CI -> above-baseline criterion)
  lens:   J-lens rank and log-softmax of the word's tokens
Blurt controls: (a) free-generation transcription must not contain the word,
(b) the model's actual next-token top-10 at carrier positions must not
contain the word (held ≠ about-to-be-said).

Usage: python scripts/c2_probe_control.py --lens runs/smollm2-135m-it/lens.pt
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

WORDS = ["orange", "lightning", "ocean", "truck", "guitar", "castle",
         "penguin", "volcano", "mirror", "honey", "winter", "robot"]
NULL_WORDS = ["door", "cloud", "pencil", "river", "shoe", "candle", "tiger",
              "bridge", "salad", "trumpet", "island", "blanket", "ladder",
              "garden", "rocket", "spider", "butter", "violin", "desert",
              "hammer", "pillow", "circus", "anchor", "lantern", "meadow",
              "engine", "bottle", "forest", "camera", "throne"]
SENTENCES = ["The old painting hung crookedly on the wall.",
             "The sun is shining brightly today in town.",
             "A gentle breeze moved the curtains in the empty room."]

CONDS = {
    "think": "Write '{s}' Think about {w} while you write the sentence.",
    "dont_think": "Write '{s}' Don't think about {w} while you write the sentence.",
    "mention": "There is such thing as {w}. Write '{s}'",
    "baseline": "Write '{s}'",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--lens", default="runs/smollm2-135m-it/lens.pt")
    ap.add_argument("--out", default="runs/c2_probe_control.json")
    ap.add_argument("--raw-format", action="store_true",
                    help="plain-text prompts instead of chat template (base models)")
    ap.add_argument("--confirm", action="store_true",
                    help="held-out confirmatory words/sentences (frozen addendum)")
    args = ap.parse_args()

    global WORDS, SENTENCES
    if args.confirm:
        WORDS = ["piano", "glacier", "dolphin", "umbrella", "tunnel", "cherry",
                 "magnet", "saddle", "comet", "harbor", "walnut", "cannon"]
        SENTENCES = ["The library closed early because of the storm.",
                     "Three birds landed on the fence near the gate.",
                     "She folded the letter and placed it in the drawer."]
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    hf = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda()
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(args.lens)
    kit = LensKit.build(model, lens)
    layers = lens.source_layers
    n_layers = model.n_layers

    def render(user, forced):
        if args.raw_format:
            return user + " " + forced
        return tok.apply_chat_template([{"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True) + forced

    # ---------- concept vectors (post's recipe) ----------
    def concept_resid(word):
        text = render(f"Tell me about {word}.", "")
        bp = build_raw(tok, text)
        r = kit.residuals(bp.input_ids, list(layers))
        return {l: r[l][-1] for l in layers}

    pool = {}
    for w in WORDS + NULL_WORDS:
        pool[w] = concept_resid(w)
    mean_all = {l: torch.stack([pool[w][l] for w in pool]).mean(0) for l in layers}
    cvec = {w: {l: (pool[w][l] - mean_all[l]) for l in layers} for w in pool}
    for w in cvec:
        for l in layers:
            cvec[w][l] = cvec[w][l] / cvec[w][l].norm().clamp_min(1e-8)

    # ---------- trials ----------
    results = {"trials": [], "model": args.model}
    for w in WORDS:
        wids = variant_token_ids(tok, w)
        for s in SENTENCES:
            for cond, tmpl in CONDS.items():
                user = tmpl.format(s=s, w=w)
                full = render(user, "'" + s + "'")
                bp = build_raw(tok, full)
                a, b = bp.find_span(s, occurrence=full.count(s) - 1)
                resid = kit.residuals(bp.input_ids, list(layers) + [n_layers - 1])
                out_logits = kit.model.unembed(resid[n_layers - 1][a:b]).float()
                out_top10 = out_logits.topk(10, dim=-1).indices
                blurt = bool((out_top10[..., None] == torch.tensor(wids, device=out_top10.device)).any())
                row = {"word": w, "sentence": s[:20], "cond": cond, "blurt_top10": blurt,
                       "probe_cos": {}, "null_mean": {}, "null_p95": {},
                       "lens_rank": {}, "lens_lp": {}}
                for l in layers:
                    span = resid[l][a:b].float()
                    span_n = span / span.norm(dim=1, keepdim=True).clamp_min(1e-8)
                    row["probe_cos"][l] = float((span_n @ cvec[w][l]).max())
                    nulls = torch.stack([ (span_n @ cvec[nw][l]).max() for nw in NULL_WORDS ])
                    row["null_mean"][l] = float(nulls.mean())
                    row["null_p95"][l] = float(nulls.quantile(0.95))
                    ranks = kit.lens_ranks_of(span, l, wids)
                    row["lens_rank"][l] = int(ranks.min())
                    lg = torch.log_softmax(kit.lens_logits(span, l), dim=-1)
                    row["lens_lp"][l] = float(lg[:, wids].max())
                results["trials"].append(row)
        print(f"{w}: done")

    # ---------- summaries ----------
    import statistics
    def per_layer(metric, cond):
        out = {}
        for l in layers:
            vals = [t[metric][l] for t in results["trials"] if t["cond"] == cond]
            out[l] = statistics.mean(vals)
        return out

    print("\nper-layer probe cosine (think / dont / mention / baseline / null95):")
    pt, pd, pm, pb = (per_layer("probe_cos", c) for c in ("think", "dont_think", "mention", "baseline"))
    n95 = per_layer("null_p95", "think")
    for l in layers:
        flag = " <-- above null95" if pt[l] > n95[l] else ""
        print(f"  L{l:>2}: {pt[l]:+.3f} / {pd[l]:+.3f} / {pm[l]:+.3f} / {pb[l]:+.3f} / {n95[l]:+.3f}{flag}")

    print("\nper-layer mean lens rank (think / dont / baseline):")
    lt, ld, lb = (per_layer("lens_rank", c) for c in ("think", "dont_think", "baseline"))
    for l in layers:
        print(f"  L{l:>2}: {lt[l]:>8.0f} / {ld[l]:>8.0f} / {lb[l]:>8.0f}")

    blurts = sum(t["blurt_top10"] for t in results["trials"] if t["cond"] == "think")
    n_think = sum(1 for t in results["trials"] if t["cond"] == "think")
    print(f"\nblurt (word in model's actual top-10 next-token preds at carrier): {blurts}/{n_think} think trials")

    with open(args.out, "w") as f:
        json.dump(results, f)
    print("saved", args.out)


if __name__ == "__main__":
    main()
