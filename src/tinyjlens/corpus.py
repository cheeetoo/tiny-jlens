"""Fitting corpus: pretraining-like prompts from wikitext-103-raw-v1.

Mirrors the Neuronpedia prefit-lens recipe (see lenses/*/config.yaml):
dataset Salesforce/wikitext, config wikitext-103-raw-v1, split train,
text field `text`, max_chars 2000. Row filtering is not specified in their
config; we use a minimal deterministic rule (non-empty, non-header, >=200
chars, taken in split order) and rely on convergence of the corpus mean —
two converged fits on different samples of the same corpus agree (validated
against the Neuronpedia gpt2-small lens in scripts/validate_fit_gpt2.py).
"""

from __future__ import annotations

import datasets


def wikitext_prompts(
    n: int,
    *,
    max_chars: int = 2000,
    min_chars: int = 200,
    skip: int = 0,
    seed_offset_rows: int = 0,
) -> list[str]:
    """First `n` eligible rows of wikitext-103-raw-v1 train (after skipping
    `skip` eligible rows), truncated to `max_chars`.

    `skip` lets callers take disjoint slices (e.g. half-vs-half fits).
    """
    ds = datasets.load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train"
    )
    prompts: list[str] = []
    n_seen_eligible = 0
    for row in ds.skip(seed_offset_rows) if seed_offset_rows else ds:
        text = row["text"].strip()
        if len(text) < min_chars or text.startswith("="):
            continue
        n_seen_eligible += 1
        if n_seen_eligible <= skip:
            continue
        prompts.append(text[:max_chars])
        if len(prompts) >= n:
            break
    if len(prompts) < n:
        raise ValueError(f"only found {len(prompts)} eligible rows, wanted {n}")
    return prompts
