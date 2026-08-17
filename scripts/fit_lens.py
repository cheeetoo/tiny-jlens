"""Fit a Jacobian lens on an HF model using Anthropic's reference implementation.

Usage:
  python scripts/fit_lens.py HuggingFaceTB/SmolLM2-135M-Instruct out/smollm2-135m-it \
      --n-prompts 1000 [--skip-prompts 0] [--dim-batch 128] [--compile]

Recipe mirrors the Neuronpedia prefit lenses (lenses/*/config.yaml): wikitext-103,
128-token sequences, target = final layer, skip_first = 16, bf16.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import transformers

import jlens
from tinyjlens.corpus import wikitext_prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out_dir")
    ap.add_argument("--n-prompts", type=int, default=1000)
    ap.add_argument("--skip-prompts", type=int, default=0)
    ap.add_argument("--dim-batch", type=int, default=128)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--target-layer", type=int, default=None)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--tag", default="lens")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    jlens.configure_logging()

    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16
    ).cuda()
    model = jlens.from_hf(hf, tok, compile=args.compile)
    print(model, flush=True)

    prompts = wikitext_prompts(args.n_prompts, skip=args.skip_prompts)
    t0 = time.time()
    lens = jlens.fit(
        model,
        prompts,
        target_layer=args.target_layer,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        checkpoint_path=os.path.join(args.out_dir, f"{args.tag}_ckpt.pt"),
        checkpoint_every=25,
    )
    lens.save(os.path.join(args.out_dir, f"{args.tag}.pt"))
    meta = {
        "model": args.model,
        "n_prompts": args.n_prompts,
        "skip_prompts": args.skip_prompts,
        "dim_batch": args.dim_batch,
        "max_seq_len": args.max_seq_len,
        "target_layer": args.target_layer,
        "skip_first": jlens.fitting.SKIP_FIRST_N_POSITIONS,
        "corpus": "Salesforce/wikitext:wikitext-103-raw-v1:train (min200/max2000 chars, non-header)",
        "dtype": "bfloat16",
        "wall_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out_dir, f"{args.tag}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("saved", os.path.join(args.out_dir, f"{args.tag}.pt"), meta, flush=True)


if __name__ == "__main__":
    main()
