"""Fit a Jacobian lens with Anthropic's reference implementation (`jlens`).

Recipe = the Neuronpedia prefit-lens recipe that produced lenses/gpt2-small
(see its config.yaml): wikitext-103-raw-v1 train rows (non-header, >=200
chars, truncated to 2000 chars), 128-token sequences, skip_first=16,
target = final layer, bf16 model. Our pipeline was validated day 1 against
the authors' released gpt2-small lens (per-layer r = 0.993-0.9996).

Usage:
  python fit_lens.py gpt2-medium ../lenses/gpt2-medium --n-prompts 1000
"""

from __future__ import annotations

import argparse
import json
import os
import time

import datasets
import torch
import transformers

import jlens


def wikitext_prompts(n: int, *, max_chars: int = 2000, min_chars: int = 200) -> list[str]:
    ds = datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    prompts: list[str] = []
    for row in ds:
        text = row["text"].strip()
        if len(text) < min_chars or text.startswith("="):
            continue
        prompts.append(text[:max_chars])
        if len(prompts) >= n:
            break
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out_dir")
    ap.add_argument("--n-prompts", type=int, default=1000)
    ap.add_argument("--dim-batch", type=int, default=128)
    ap.add_argument("--target-layer", type=int, default=None)
    ap.add_argument("--tag", default="lens")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    jlens.configure_logging()

    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    hf = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda()
    model = jlens.from_hf(hf, tok, compile=True)
    print(model, flush=True)

    prompts = wikitext_prompts(args.n_prompts)
    t0 = time.time()
    lens = jlens.fit(
        model,
        prompts,
        target_layer=args.target_layer,
        dim_batch=args.dim_batch,
        max_seq_len=128,
        checkpoint_path=os.path.join(args.out_dir, f"{args.tag}_ckpt.pt"),
        checkpoint_every=25,
    )
    lens.save(os.path.join(args.out_dir, f"{args.tag}.pt"))
    meta = {
        "model": args.model,
        "n_prompts": args.n_prompts,
        "dim_batch": args.dim_batch,
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
