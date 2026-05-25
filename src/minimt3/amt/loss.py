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
        duration_long_weight: float = 0.0,
        duration_long_target_min: float = 0.35,
        duration_bucket_weight: float = 0.0,
        duration_bucket_class_weights: list[float] | tuple[float, ...] | None = None,
        duration_consistency_weight: float = 0.0,
        teacher_onset_weight: float = 0.0,
        teacher_frame_weight: float = 0.0,
        teacher_offset_weight: float = 0.0,
        teacher_positive_only: bool = True,
        teacher_gt_gate: bool = False,
        teacher_gt_gate_frames: int = 1,
        onset_shift_weight: float = 0.0,
        offset_shift_weight: float = 0.0,
        onset_mass_weight: float = 0.0,
        offset_mass_weight: float = 0.0,
        frame_mass_weight: float = 0.0,
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
        self.duration_long_weight = duration_long_weight
        self.duration_long_target_min = duration_long_target_min
        self.duration_bucket_weight = duration_bucket_weight
        self.duration_bucket_class_weights = tuple(float(x) for x in duration_bucket_class_weights or ())
        self.duration_consistency_weight = duration_consistency_weight
        self.teacher_onset_weight = teacher_onset_weight
        self.teacher_frame_weight = teacher_frame_weight
        self.teacher_offset_weight = teacher_offset_weight
        self.teacher_positive_only = teacher_positive_only
        self.teacher_gt_gate = teacher_gt_gate
        self.teacher_gt_gate_frames = teacher_gt_gate_frames
        self.onset_shift_weight = onset_shift_weight
        self.offset_shift_weight = offset_shift_weight
        self.onset_mass_weight = onset_mass_weight
        self.offset_mass_weight = offset_mass_weight
        self.frame_mass_weight = frame_mass_weight
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
        if "sample_weight" in batch:
            sample_weight = batch["sample_weight"].to(device, non_blocking=True).float().view(-1, 1, 1)
            valid = valid * sample_weight

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
        if self.onset_mass_weight > 0.0:
            onset_mass_loss = _masked_mass_loss(onset_logits, onset, valid)
            loss = loss + self.onset_mass_weight * onset_mass_loss
            logs["ONSET_MASS"] = float(onset_mass_loss.detach().cpu())
        if self.offset_mass_weight > 0.0:
            offset_mass_loss = _masked_mass_loss(offset_logits, offset, valid)
            loss = loss + self.offset_mass_weight * offset_mass_loss
            logs["OFFSET_MASS"] = float(offset_mass_loss.detach().cpu())
        if self.frame_mass_weight > 0.0:
            frame_mass_loss = _masked_mass_loss(frame_logits, frame, valid)
            loss = loss + self.frame_mass_weight * frame_mass_loss
            logs["FRAME_MASS"] = float(frame_mass_loss.detach().cpu())
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
            duration_weight = torch.ones_like(duration)
            if self.duration_long_weight > 0.0:
                long_mask = (duration >= float(self.duration_long_target_min)).float()
                duration_weight = duration_weight + long_mask * float(self.duration_long_weight)
            duration_masked_weight = duration_mask * valid * duration_weight
            duration_denom = duration_masked_weight.sum().clamp_min(1.0)
            duration_loss = (
                F.smooth_l1_loss(duration_pred, duration, reduction="none") * duration_masked_weight
            ).sum() / duration_denom
            loss = loss + self.duration_weight * duration_loss
            logs["DURATION"] = float(duration_loss.detach().cpu())
        if (
            self.duration_bucket_weight > 0.0
            and "duration_bucket_logits" in model_out
            and "duration_bucket" in batch
        ):
            duration_bucket = batch["duration_bucket"].to(device, non_blocking=True)[:, :max_len]
            duration_bucket_mask = batch["duration_bucket_mask"].to(device, non_blocking=True)[:, :max_len]
            duration_bucket_logits = model_out["duration_bucket_logits"][:, :max_len]
            duration_bucket_loss = _masked_duration_bucket_ce(
                duration_bucket_logits,
                duration_bucket,
                duration_bucket_mask,
                valid,
                self.duration_bucket_class_weights,
            )
            loss = loss + self.duration_bucket_weight * duration_bucket_loss
            logs["DURATION_BUCKET"] = float(duration_bucket_loss.detach().cpu())
        if (
            self.duration_consistency_weight > 0.0
            and "duration_frame" in batch
            and "duration_mask" in batch
        ):
            duration_frame = batch["duration_frame"].to(device, non_blocking=True)[:, :max_len]
            duration_mask = batch["duration_mask"].to(device, non_blocking=True)[:, :max_len]
            consistency_loss = _duration_offset_consistency(offset_logits, duration_frame, duration_mask, valid)
            loss = loss + self.duration_consistency_weight * consistency_loss
            logs["DURATION_OFFSET_CONSISTENCY"] = float(consistency_loss.detach().cpu())
        if "teacher_mask" in batch:
            teacher_mask = batch["teacher_mask"].to(device, non_blocking=True)[:, :max_len]
            gt_targets = {"onset": onset, "frame": frame, "offset": offset}
            teacher_specs = (
                ("onset", self.teacher_onset_weight, "TEACHER_ONSET"),
                ("frame", self.teacher_frame_weight, "TEACHER_FRAME"),
                ("offset", self.teacher_offset_weight, "TEACHER_OFFSET"),
            )
            for key, weight, log_key in teacher_specs:
                teacher_key = f"teacher_{key}"
                logits_key = f"{key}_logits"
                if weight <= 0.0 or teacher_key not in batch or logits_key not in model_out:
                    continue
                teacher_target = batch[teacher_key].to(device, non_blocking=True)[:, :max_len]
                teacher_loss_mask = teacher_mask
                if self.teacher_gt_gate:
                    gate = _dilate_time_mask(
                        (gt_targets[key] > 0).float(),
                        radius=max(0, int(self.teacher_gt_gate_frames)),
                    )
                    teacher_loss_mask = teacher_loss_mask * gate
                teacher_loss = _masked_teacher_bce(
                    model_out[logits_key][:, :max_len],
                    teacher_target,
                    teacher_loss_mask,
                    valid,
                    positive_only=bool(self.teacher_positive_only),
                )
                loss = loss + float(weight) * teacher_loss
                logs[log_key] = float(teacher_loss.detach().cpu())
        if self.onset_shift_weight > 0.0 and "onset_shift_logits" in model_out and "onset_shift" in batch:
            onset_shift = batch["onset_shift"].to(device, non_blocking=True)[:, :max_len]
            onset_shift_mask = batch["onset_shift_mask"].to(device, non_blocking=True)[:, :max_len]
            onset_shift_pred = torch.tanh(model_out["onset_shift_logits"][:, :max_len])
            onset_shift_loss = _masked_regression(onset_shift_pred, onset_shift, onset_shift_mask, valid)
            loss = loss + self.onset_shift_weight * onset_shift_loss
            logs["ONSET_SHIFT"] = float(onset_shift_loss.detach().cpu())
        if self.offset_shift_weight > 0.0 and "offset_shift_logits" in model_out and "offset_shift" in batch:
            offset_shift = batch["offset_shift"].to(device, non_blocking=True)[:, :max_len]
            offset_shift_mask = batch["offset_shift_mask"].to(device, non_blocking=True)[:, :max_len]
            offset_shift_pred = torch.tanh(model_out["offset_shift_logits"][:, :max_len])
            offset_shift_loss = _masked_regression(offset_shift_pred, offset_shift, offset_shift_mask, valid)
            loss = loss + self.offset_shift_weight * offset_shift_loss
            logs["OFFSET_SHIFT"] = float(offset_shift_loss.detach().cpu())
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


def _masked_regression(
    pred: torch.Tensor,
    target: torch.Tensor,
    event_mask: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    mask = event_mask * valid
    denom = mask.sum().clamp_min(1.0)
    return (F.smooth_l1_loss(pred, target, reduction="none") * mask).sum() / denom


def _masked_mass_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Penalize clip-level probability mass drift to reduce dense false positives."""
    mask = valid.expand_as(logits)
    pred_mass = (torch.sigmoid(logits) * mask).sum(dim=(1, 2))
    target_mass = (target * mask).sum(dim=(1, 2))
    return F.smooth_l1_loss(torch.log1p(pred_mass), torch.log1p(target_mass), reduction="mean")


def _masked_duration_bucket_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    event_mask: torch.Tensor,
    valid: torch.Tensor,
    class_weights: tuple[float, ...],
) -> torch.Tensor:
    classes = int(logits.shape[-1])
    target = target.long().clamp(0, classes - 1)
    raw = F.cross_entropy(
        logits.reshape(-1, classes),
        target.reshape(-1),
        reduction="none",
    ).view_as(event_mask)
    mask = event_mask * valid
    if class_weights:
        weights = torch.ones(classes, device=logits.device, dtype=logits.dtype)
        for idx, value in enumerate(class_weights[:classes]):
            weights[idx] = float(value)
        mask = mask * weights[target]
    denom = mask.sum().clamp_min(1.0)
    return (raw * mask).sum() / denom


def _masked_teacher_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    teacher_mask: torch.Tensor,
    valid: torch.Tensor,
    positive_only: bool,
) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    mask = teacher_mask * valid
    if positive_only:
        mask = mask * (target > 0).float()
    denom = mask.sum().clamp_min(1.0)
    return (raw * mask).sum() / denom


def _dilate_time_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    batch, frames, pitches = mask.shape
    x = mask.permute(0, 2, 1).reshape(batch * pitches, 1, frames)
    pooled = F.max_pool1d(x, kernel_size=radius * 2 + 1, stride=1, padding=radius)
    return pooled.reshape(batch, pitches, frames).permute(0, 2, 1)


def _duration_offset_consistency(
    offset_logits: torch.Tensor,
    duration_frame: torch.Tensor,
    duration_mask: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    offset_prob = torch.sigmoid(offset_logits)
    frames = offset_prob.shape[1]
    positions = torch.arange(frames, device=offset_prob.device, dtype=offset_prob.dtype).view(1, frames, 1)
    reverse_prob = torch.flip(torch.cumsum(torch.flip(offset_prob, dims=[1]), dim=1), dims=[1])
    reverse_pos = torch.flip(torch.cumsum(torch.flip(offset_prob * positions, dims=[1]), dim=1), dims=[1])
    expected_rel = (reverse_pos - positions * reverse_prob) / reverse_prob.clamp_min(1e-6)
    expected_fraction = (expected_rel / max(1, frames - 1)).clamp(0.0, 1.0)
    mask = duration_mask * valid
    denom = mask.sum().clamp_min(1.0)
    raw = F.smooth_l1_loss(expected_fraction, duration_frame, reduction="none")
    return (raw * mask).sum() / denom
