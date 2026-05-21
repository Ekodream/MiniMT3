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
        onset_neg_weight: float = 1.0,
        frame_neg_weight: float = 1.0,
        offset_neg_weight: float = 1.0,
        pedal_weight: float = 0.0,
        pedal_pos_weight: float = 3.0,
        pedal_neg_weight: float = 1.0,
        duration_weight: float = 0.0,
        onset_focal_gamma_pos: float = 0.0,
        onset_focal_gamma_neg: float = 0.0,
        frame_focal_gamma_pos: float = 0.0,
        frame_focal_gamma_neg: float = 0.0,
        offset_focal_gamma_pos: float = 0.0,
        offset_focal_gamma_neg: float = 0.0,
    ):
        super().__init__()
        self.onset_weight = onset_weight
        self.frame_weight = frame_weight
        self.offset_weight = offset_weight
        self.velocity_weight = velocity_weight
        self.onset_neg_weight = onset_neg_weight
        self.frame_neg_weight = frame_neg_weight
        self.offset_neg_weight = offset_neg_weight
        self.pedal_weight = pedal_weight
        self.pedal_neg_weight = pedal_neg_weight
        self.duration_weight = duration_weight
        self.onset_focal_gamma_pos = onset_focal_gamma_pos
        self.onset_focal_gamma_neg = onset_focal_gamma_neg
        self.frame_focal_gamma_pos = frame_focal_gamma_pos
        self.frame_focal_gamma_neg = frame_focal_gamma_neg
        self.offset_focal_gamma_pos = offset_focal_gamma_pos
        self.offset_focal_gamma_neg = offset_focal_gamma_neg
        self.register_buffer("onset_pos_weight", torch.full((88,), onset_pos_weight))
        self.register_buffer("frame_pos_weight", torch.full((88,), frame_pos_weight))
        self.register_buffer("offset_pos_weight", torch.full((88,), offset_pos_weight))
        self.register_buffer("pedal_pos_weight", torch.full((1,), pedal_pos_weight))

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

        onset_loss = _masked_bce(
            onset_logits,
            onset,
            valid,
            self.onset_pos_weight,
            neg_weight=self.onset_neg_weight,
            gamma_pos=self.onset_focal_gamma_pos,
            gamma_neg=self.onset_focal_gamma_neg,
        )
        frame_loss = _masked_bce(
            frame_logits,
            frame,
            valid,
            self.frame_pos_weight,
            neg_weight=self.frame_neg_weight,
            gamma_pos=self.frame_focal_gamma_pos,
            gamma_neg=self.frame_focal_gamma_neg,
        )
        offset_loss = _masked_bce(
            offset_logits,
            offset,
            valid,
            self.offset_pos_weight,
            neg_weight=self.offset_neg_weight,
            gamma_pos=self.offset_focal_gamma_pos,
            gamma_neg=self.offset_focal_gamma_neg,
        )
        velocity_pred = torch.sigmoid(velocity_logits)
        velocity_denom = (onset_mask * valid).sum().clamp_min(1.0)
        velocity_loss = ((velocity_pred - velocity).abs() * onset_mask * valid).sum() / velocity_denom
        loss = (
            self.onset_weight * onset_loss
            + self.frame_weight * frame_loss
            + self.offset_weight * offset_loss
            + self.velocity_weight * velocity_loss
        )
        logs = {
            "ONSET": float(onset_loss.detach().cpu()),
            "FRAME": float(frame_loss.detach().cpu()),
            "OFFSET": float(offset_loss.detach().cpu()),
            "VELOCITY": float(velocity_loss.detach().cpu()),
        }
        if self.pedal_weight > 0.0 and "pedal_logits" in model_out and "pedal" in batch:
            pedal = batch["pedal"].to(device, non_blocking=True)[:, :max_len]
            pedal_logits = model_out["pedal_logits"][:, :max_len]
            pedal_loss = _masked_bce(
                pedal_logits,
                pedal,
                valid,
                self.pedal_pos_weight,
                neg_weight=self.pedal_neg_weight,
            )
            loss = loss + self.pedal_weight * pedal_loss
            logs["PEDAL"] = float(pedal_loss.detach().cpu())
        if self.duration_weight > 0.0 and "duration_logits" in model_out and "duration" in batch:
            duration = batch["duration"].to(device, non_blocking=True)[:, :max_len]
            duration_mask = batch["duration_mask"].to(device, non_blocking=True)[:, :max_len]
            duration_pred = torch.sigmoid(model_out["duration_logits"][:, :max_len])
            duration_denom = (duration_mask * valid).sum().clamp_min(1.0)
            duration_loss = (
                F.smooth_l1_loss(duration_pred, duration, reduction="none") * duration_mask * valid
            ).sum() / duration_denom
            loss = loss + self.duration_weight * duration_loss
            logs["DURATION"] = float(duration_loss.detach().cpu())
        return DenseLossOutput(
            loss=loss,
            logs=logs,
        )


def _masked_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    pos_weight: torch.Tensor,
    neg_weight: float = 1.0,
    gamma_pos: float = 0.0,
    gamma_neg: float = 0.0,
) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
        pos_weight=pos_weight.to(logits.device),
    )
    if neg_weight != 1.0 or gamma_pos > 0.0 or gamma_neg > 0.0:
        prob = torch.sigmoid(logits)
        weights = torch.ones_like(raw)
        if neg_weight != 1.0:
            weights = torch.where(target > 0.5, weights, weights * float(neg_weight))
        if gamma_pos > 0.0 or gamma_neg > 0.0:
            pos_focus = (1.0 - prob).clamp_min(1e-6).pow(float(gamma_pos))
            neg_focus = prob.clamp_min(1e-6).pow(float(gamma_neg))
            weights = weights * torch.where(target > 0.5, pos_focus, neg_focus)
        raw = raw * weights
    denom = (valid.sum() * logits.shape[-1]).clamp_min(1.0)
    return (raw * valid).sum() / denom
