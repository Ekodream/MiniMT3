from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class DenseLossOutput:
    loss: torch.Tensor
    logs: dict[str, float]


class DenseAMTLoss(nn.Module):
    def __init__(
        self,
        onset_weight: float = 1.0,
        frame_weight: float = 0.5,
        offset_weight: float = 0.5,
        velocity_weight: float = 0.1,
        onset_pos_weight: float = 12.0,
        frame_pos_weight: float = 2.5,
        offset_pos_weight: float = 12.0,
    ):
        super().__init__()
        self.onset_weight = onset_weight
        self.frame_weight = frame_weight
        self.offset_weight = offset_weight
        self.velocity_weight = velocity_weight
        self.register_buffer("onset_pos_weight", torch.full((88,), onset_pos_weight))
        self.register_buffer("frame_pos_weight", torch.full((88,), frame_pos_weight))
        self.register_buffer("offset_pos_weight", torch.full((88,), offset_pos_weight))

    def forward(self, model_out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> DenseLossOutput:
        device = model_out["onset_logits"].device
        onset = batch["onset"].to(device, non_blocking=True)
        frame = batch["frame"].to(device, non_blocking=True)
        offset = batch["offset"].to(device, non_blocking=True)
        velocity = batch["velocity"].to(device, non_blocking=True)
        onset_mask = batch["onset_mask"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True).unsqueeze(-1).float()

        max_len = min(model_out["onset_logits"].shape[1], onset.shape[1])
        onset_logits = model_out["onset_logits"][:, :max_len]
        frame_logits = model_out["frame_logits"][:, :max_len]
        offset_logits = model_out["offset_logits"][:, :max_len]
        velocity_logits = model_out["velocity_logits"][:, :max_len]
        onset = onset[:, :max_len]
        frame = frame[:, :max_len]
        offset = offset[:, :max_len]
        velocity = velocity[:, :max_len]
        onset_mask = onset_mask[:, :max_len]
        valid = valid[:, :max_len]

        onset_loss = _masked_bce(onset_logits, onset, valid, self.onset_pos_weight)
        frame_loss = _masked_bce(frame_logits, frame, valid, self.frame_pos_weight)
        offset_loss = _masked_bce(offset_logits, offset, valid, self.offset_pos_weight)
        velocity_pred = torch.sigmoid(velocity_logits)
        velocity_denom = (onset_mask * valid).sum().clamp_min(1.0)
        velocity_loss = ((velocity_pred - velocity).abs() * onset_mask * valid).sum() / velocity_denom
        loss = (
            self.onset_weight * onset_loss
            + self.frame_weight * frame_loss
            + self.offset_weight * offset_loss
            + self.velocity_weight * velocity_loss
        )
        return DenseLossOutput(
            loss=loss,
            logs={
                "ONSET": float(onset_loss.detach().cpu()),
                "FRAME": float(frame_loss.detach().cpu()),
                "OFFSET": float(offset_loss.detach().cpu()),
                "VELOCITY": float(velocity_loss.detach().cpu()),
            },
        )


def _masked_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
        pos_weight=pos_weight.to(logits.device),
    )
    denom = (valid.sum() * logits.shape[-1]).clamp_min(1.0)
    return (raw * valid).sum() / denom
