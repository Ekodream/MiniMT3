from __future__ import annotations

import torch
from torch import nn


class Seq2SeqLoss(nn.Module):
    def __init__(self, pad_id: int):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
