from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from minimt3.symbolic.events import EventCodec


@dataclass
class LossOutput:
    loss: torch.Tensor
    family_losses: dict[str, float]


class WeightedSeq2SeqLoss(nn.Module):
    def __init__(
        self,
        codec: EventCodec,
        label_smoothing: float = 0.05,
        family_weights: dict[str, float] | None = None,
    ):
        super().__init__()
        self.codec = codec
        self.pad_id = codec.pad_id
        self.label_smoothing = label_smoothing
        family_weights = family_weights or {
            "PITCH": 1.35,
            "EOS": 1.5,
            "PEDAL": 1.2,
            "VELOCITY": 0.8,
            "SHIFT": 0.9,
        }
        weights = torch.tensor(codec.family_mask(family_weights), dtype=torch.float32)
        self.register_buffer("weights", weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> LossOutput:
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_target = target.reshape(-1)
        valid = flat_target.ne(self.pad_id)
        per_token = F.cross_entropy(
            flat_logits,
            flat_target,
            ignore_index=self.pad_id,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        token_weights = self.weights.to(flat_logits.device).gather(
            0, flat_target.clamp_min(0).clamp_max(self.weights.numel() - 1)
        )
        loss = (per_token * token_weights * valid).sum() / (token_weights * valid).sum().clamp_min(1.0)
        return LossOutput(loss=loss, family_losses=self._family_losses(per_token.detach(), flat_target, valid))

    def _family_losses(
        self,
        per_token: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        for family in ["SHIFT", "VELOCITY", "PITCH", "PEDAL", "EOS"]:
            ids = [
                token_id
                for token_id in range(self.codec.vocab_size)
                if self.codec.token_family(token_id) == family
            ]
            if not ids:
                continue
            mask = torch.zeros_like(valid)
            for token_id in ids:
                mask |= target.eq(token_id)
            mask &= valid
            if mask.any():
                out[family] = float(per_token[mask].mean().cpu())
        return out


Seq2SeqLoss = WeightedSeq2SeqLoss
