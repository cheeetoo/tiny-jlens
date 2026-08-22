"""Residual-stream edits as forward hooks on transformer blocks.

An Edit says: at the output of block `layer`, at token `positions` (None = every position),
replace the residual rows h [n, d] by fn(h, positions).  Edits at the same layer are applied
in the order given (so e.g. a swap followed by a clamp means the clamp wins).  Batch size 1.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Edit:
    layer: int
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # (h[n,d], pos[n]) -> h'[n,d]
    positions: Sequence[int] | None = None


class Session:
    """Install edits for the duration of a `with` block."""

    def __init__(self, blocks: Sequence[nn.Module], edits: Sequence[Edit]):
        self.blocks = blocks
        self.by_layer: dict[int, list[Edit]] = {}
        for e in edits:
            self.by_layer.setdefault(e.layer, []).append(e)
        self._handles: list = []

    def _hook(self, layer: int):
        def hook(module, inputs, output):
            hidden = output if torch.is_tensor(output) else output[0]
            assert hidden.shape[0] == 1, "edits assume batch size 1"
            h = hidden[0]
            T = h.shape[0]
            for e in self.by_layer[layer]:
                if e.positions is None:
                    idx = torch.arange(T, device=h.device)
                else:
                    idx = torch.as_tensor(sorted(set(e.positions)), device=h.device)
                    idx = idx[idx < T]
                if len(idx):
                    h = h.index_copy(0, idx, e.fn(h[idx].float(), idx).to(h.dtype))
            h = h.unsqueeze(0)
            return h if torch.is_tensor(output) else (h,) + tuple(output[1:])

        return hook

    def __enter__(self):
        for layer in self.by_layer:
            self._handles.append(self.blocks[layer].register_forward_hook(self._hook(layer)))
        return self

    def __exit__(self, *exc):
        for hd in self._handles:
            hd.remove()
        self._handles = []
