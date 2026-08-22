"""Prompt material for criterion 1, from the paper's released data
(ref/jacobian-lens/data/experiments/verbal-report.json: 14 categories x 14 candidate words).

Base-model frame (deviation; see PROTOCOL.md): the request is the last line of a 9-line list
whose first 8 lines are the same request for other categories with an answer filled in.  Demo
categories are the 8 following the queried one in the paper's order; a demo answer is that
category's first single-token candidate that is not also a candidate of the queried category.
The lens is read, and the swap graded, at the final ':'.
"""
from __future__ import annotations

import json

DATA = "/tiny-jlens/ref/jacobian-lens/data/experiments/verbal-report.json"
CANDIDATES: dict[str, list[str]] = json.load(open(DATA))["candidates"]
CATEGORIES: list[str] = list(CANDIDATES)
LINE = "Name a {cat}:"
K_SHOT = 8
CONCEPT_PROMPT = "Tell me about {concept}."  # §3.1 concept vectors


def members(lm, cat: str) -> list[str]:
    """The paper's candidates that are single GPT-2 tokens as ' Word', in the paper's order."""
    return [w for w in CANDIDATES[cat] if lm.is_single(" " + w)]


def ten(lm, cat: str) -> list[str]:
    """The paper's '10 candidate answers'."""
    return members(lm, cat)[:10]


def prompt(lm, cat: str) -> str:
    i = CATEGORIES.index(cat)
    demos = []
    for j in range(1, len(CATEGORIES)):
        c = CATEGORIES[(i + j) % len(CATEGORIES)]
        demos.append((c, next(w for w in members(lm, c) if w not in CANDIDATES[cat])))
        if len(demos) == K_SHOT:
            break
    return "".join(f"{LINE.format(cat=c)} {a}\n" for c, a in demos) + LINE.format(cat=cat)
