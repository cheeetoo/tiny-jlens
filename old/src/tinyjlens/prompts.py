"""Prompt construction with token-position bookkeeping.

Everything downstream needs to know *where* things are: the span of the user's
text, the position of a readout anchor (colon / open-quote), the span of a
carrier sentence, etc. We tokenize the full rendered string once with a fast
tokenizer and map character offsets to token indices.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BuiltPrompt:
    text: str
    input_ids: torch.Tensor  # [1, seq]
    offsets: list[tuple[int, int]]  # per-token char spans

    @property
    def n_tokens(self) -> int:
        return self.input_ids.shape[1]

    def char_to_token(self, char_idx: int) -> int:
        """Token index containing char_idx (or the nearest token starting after)."""
        for i, (a, b) in enumerate(self.offsets):
            if a <= char_idx < max(b, a + 1):
                return i
        for i, (a, b) in enumerate(self.offsets):
            if a >= char_idx:
                return i
        return len(self.offsets) - 1

    def find_span(self, sub: str, *, occurrence: int = 0) -> tuple[int, int]:
        """Token span [start, end) covering the given substring occurrence."""
        start = -1
        for _ in range(occurrence + 1):
            start = self.text.find(sub, start + 1)
            if start == -1:
                raise ValueError(f"substring {sub!r} (occ {occurrence}) not found")
        end_char = start + len(sub)
        t0 = self.char_to_token(start)
        t1 = self.char_to_token(end_char - 1) + 1
        return t0, t1

    def last_token_index(self) -> int:
        return self.n_tokens - 1


def build_raw(tokenizer, text: str, device="cuda") -> BuiltPrompt:
    enc = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    ids = enc["input_ids"].to(device)
    offsets = [tuple(x) for x in enc["offset_mapping"][0].tolist()]
    return BuiltPrompt(text=text, input_ids=ids, offsets=offsets)


def build_chat(tokenizer, user: str, *, assistant_prefill: str = "",
               system: str | None = None, device="cuda") -> BuiltPrompt:
    """Render a chat prompt (with generation prompt) plus an optional
    teacher-forced assistant prefill, tracking offsets over the full string."""
    msgs = []
    if system is not None:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    rendered = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    full = rendered + assistant_prefill
    return build_raw(tokenizer, full, device=device)


def single_token_id(tokenizer, word: str) -> int | None:
    """Token id if ` word` (with leading space) or `word` is a single token;
    prefers the leading-space form (sentence-medial), else None."""
    for form in (" " + word, word):
        ids = tokenizer(form, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            return ids[0]
    return None


def variant_token_ids(tokenizer, word: str) -> list[int]:
    """Single-token ids among surface variants of `word` (leading space,
    bare, capitalized, upper). Deduplicated, order-stable."""
    seen: list[int] = []
    for w in {word, word.capitalize(), word.lower(), word.upper()}:
        for form in (" " + w, w):
            ids = tokenizer(form, add_special_tokens=False)["input_ids"]
            if len(ids) == 1 and ids[0] not in seen:
                seen.append(ids[0])
    return seen
