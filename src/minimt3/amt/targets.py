from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from minimt3.symbolic.events import NoteEvent, PITCH_MAX, PITCH_MIN, load_midi_events

NUM_PITCHES = PITCH_MAX - PITCH_MIN + 1


@dataclass
class DenseTargetConfig:
    onset_width_frames: int = 1
    offset_width_frames: int = 1
    onset_soft_radius_frames: int = 0
    offset_soft_radius_frames: int = 0
    min_note_seconds: float = 0.02
    include_pedal: bool = False
    include_duration: bool = False
    include_duration_bucket: bool = False
    include_time_shifts: bool = False
    max_duration_seconds: float = 8.0
    duration_bucket_bounds: tuple[float, ...] | list[float] = (0.125, 0.5, 2.0)
    frame_strides: tuple[int, int] | list[int] = (2, 2)
    time_shift_clip_frames: float = 1.0
    supervision_margin_seconds: float = 0.0


def encoder_frame_count(feature_frames: int, strides: tuple[int, ...] | list[int] = (2, 2)) -> int:
    frames = int(feature_frames)
    for stride in strides:
        stride = max(1, int(stride))
        frames = (frames + stride - 1) // stride
    return max(1, frames)


def build_dense_targets(
    midi_path: str | Path,
    start: float,
    end: float,
    frames: int,
    cfg: DenseTargetConfig | None = None,
) -> dict[str, torch.Tensor]:
    cfg = cfg or DenseTargetConfig()
    duration = max(0.01, float(end) - float(start))
    notes, pedals, tie_notes = load_midi_events(midi_path, start=start, end=end, include_ties=True)
    onset = torch.zeros(frames, NUM_PITCHES)
    frame = torch.zeros(frames, NUM_PITCHES)
    offset = torch.zeros(frames, NUM_PITCHES)
    velocity = torch.zeros(frames, NUM_PITCHES)
    onset_mask = torch.zeros(frames, NUM_PITCHES)
    note_duration = torch.zeros(frames, NUM_PITCHES) if cfg.include_duration else None
    note_duration_frame = torch.zeros(frames, NUM_PITCHES) if cfg.include_duration else None
    duration_mask = torch.zeros(frames, NUM_PITCHES) if cfg.include_duration else None
    duration_bucket = torch.zeros(frames, NUM_PITCHES, dtype=torch.long) if cfg.include_duration_bucket else None
    duration_bucket_mask = torch.zeros(frames, NUM_PITCHES) if cfg.include_duration_bucket else None
    onset_shift = torch.zeros(frames, NUM_PITCHES) if cfg.include_time_shifts else None
    onset_shift_mask = torch.zeros(frames, NUM_PITCHES) if cfg.include_time_shifts else None
    offset_shift = torch.zeros(frames, NUM_PITCHES) if cfg.include_time_shifts else None
    offset_shift_mask = torch.zeros(frames, NUM_PITCHES) if cfg.include_time_shifts else None

    for note in notes:
        _write_note(
            note,
            onset,
            frame,
            offset,
            velocity,
            onset_mask,
            duration,
            cfg,
            write_onset=True,
            note_duration=note_duration,
            note_duration_frame=note_duration_frame,
            duration_mask=duration_mask,
            duration_bucket=duration_bucket,
            duration_bucket_mask=duration_bucket_mask,
            onset_shift=onset_shift,
            onset_shift_mask=onset_shift_mask,
            offset_shift=offset_shift,
            offset_shift_mask=offset_shift_mask,
        )
    for note in tie_notes:
        _write_note(
            note,
            onset,
            frame,
            offset,
            velocity,
            onset_mask,
            duration,
            cfg,
            write_onset=False,
            note_duration=note_duration,
            note_duration_frame=note_duration_frame,
            duration_mask=duration_mask,
            duration_bucket=duration_bucket,
            duration_bucket_mask=duration_bucket_mask,
            onset_shift=onset_shift,
            onset_shift_mask=onset_shift_mask,
            offset_shift=offset_shift,
            offset_shift_mask=offset_shift_mask,
        )
    targets = {
        "onset": onset,
        "frame": frame,
        "offset": offset,
        "velocity": velocity,
        "onset_mask": onset_mask,
    }
    if note_duration is not None and duration_mask is not None:
        targets["duration"] = note_duration
        if note_duration_frame is not None:
            targets["duration_frame"] = note_duration_frame
        targets["duration_mask"] = duration_mask
    if duration_bucket is not None and duration_bucket_mask is not None:
        targets["duration_bucket"] = duration_bucket
        targets["duration_bucket_mask"] = duration_bucket_mask
    if (
        onset_shift is not None
        and onset_shift_mask is not None
        and offset_shift is not None
        and offset_shift_mask is not None
    ):
        targets["onset_shift"] = onset_shift
        targets["onset_shift_mask"] = onset_shift_mask
        targets["offset_shift"] = offset_shift
        targets["offset_shift_mask"] = offset_shift_mask
    if cfg.include_pedal:
        pedal = torch.zeros(frames, 1)
        for item in pedals:
            start_idx = _time_to_frame(item.start, duration, frames)
            end_idx = max(start_idx + 1, _time_to_frame(item.end, duration, frames, end=True))
            pedal[start_idx:min(frames, end_idx), 0] = 1.0
        targets["pedal"] = pedal
    return targets


def _write_note(
    note: NoteEvent,
    onset: torch.Tensor,
    frame: torch.Tensor,
    offset: torch.Tensor,
    velocity: torch.Tensor,
    onset_mask: torch.Tensor,
    duration: float,
    cfg: DenseTargetConfig,
    write_onset: bool,
    note_duration: torch.Tensor | None = None,
    note_duration_frame: torch.Tensor | None = None,
    duration_mask: torch.Tensor | None = None,
    duration_bucket: torch.Tensor | None = None,
    duration_bucket_mask: torch.Tensor | None = None,
    onset_shift: torch.Tensor | None = None,
    onset_shift_mask: torch.Tensor | None = None,
    offset_shift: torch.Tensor | None = None,
    offset_shift_mask: torch.Tensor | None = None,
) -> None:
    if not (PITCH_MIN <= int(note.pitch) <= PITCH_MAX):
        return
    if note.end - note.start < cfg.min_note_seconds:
        return
    frames = onset.shape[0]
    pitch_idx = int(note.pitch) - PITCH_MIN
    start_idx = _time_to_frame(note.start, duration, frames)
    end_idx = max(start_idx + 1, _time_to_frame(note.end, duration, frames, end=True))
    end_idx = min(frames, end_idx)
    frame[start_idx:end_idx, pitch_idx] = 1.0
    if write_onset:
        onset_center = _time_to_frame_float(note.start, duration, frames)
        onset_indices = _write_boundary_target(
            onset,
            start_idx,
            pitch_idx,
            hard_width_frames=cfg.onset_width_frames,
            soft_radius_frames=cfg.onset_soft_radius_frames,
        )
        for idx in onset_indices:
            onset_mask[idx, pitch_idx] = 1.0
            velocity[idx, pitch_idx] = max(1, min(127, int(note.velocity))) / 127.0
            if onset_shift is not None and onset_shift_mask is not None:
                onset_shift[idx, pitch_idx] = _clip_shift(
                    onset_center - idx,
                    cfg.time_shift_clip_frames,
                )
                onset_shift_mask[idx, pitch_idx] = 1.0
        if note_duration is not None and duration_mask is not None:
            normalized = _duration_to_unit(note.end - note.start, cfg.max_duration_seconds)
            frame_fraction = max(0.0, min(1.0, (end_idx - start_idx) / max(1, frames - 1)))
            bucket_idx = _duration_bucket(note.end - note.start, cfg.duration_bucket_bounds)
            for idx in onset_indices:
                note_duration[idx, pitch_idx] = normalized
                if note_duration_frame is not None:
                    note_duration_frame[idx, pitch_idx] = frame_fraction
                duration_mask[idx, pitch_idx] = 1.0
                if duration_bucket is not None and duration_bucket_mask is not None:
                    duration_bucket[idx, pitch_idx] = bucket_idx
                    duration_bucket_mask[idx, pitch_idx] = 1.0
    if note.end < duration - 1e-4:
        offset_center = _time_to_frame_float(note.end, duration, frames)
        offset_idx = min(frames - 1, max(0, int(offset_center)))
        offset_indices = _write_boundary_target(
            offset,
            offset_idx,
            pitch_idx,
            hard_width_frames=cfg.offset_width_frames,
            soft_radius_frames=cfg.offset_soft_radius_frames,
        )
        if offset_shift is not None and offset_shift_mask is not None:
            for idx in offset_indices:
                offset_shift[idx, pitch_idx] = _clip_shift(
                    offset_center - idx,
                    cfg.time_shift_clip_frames,
                )
                offset_shift_mask[idx, pitch_idx] = 1.0


def _write_boundary_target(
    target: torch.Tensor,
    center_idx: int,
    pitch_idx: int,
    hard_width_frames: int,
    soft_radius_frames: int,
) -> list[int]:
    """Write a hard or triangular boundary target and return supervised frames."""
    frames = target.shape[0]
    if soft_radius_frames > 0:
        radius = int(soft_radius_frames)
        written: list[int] = []
        for idx in range(max(0, center_idx - radius), min(frames, center_idx + radius + 1)):
            distance = abs(idx - center_idx)
            value = 1.0 - (distance / float(radius + 1))
            target[idx, pitch_idx] = max(float(target[idx, pitch_idx]), value)
            written.append(idx)
        return written
    width = max(0, int(hard_width_frames))
    lo = max(0, center_idx - width)
    hi = min(frames, center_idx + width + 1)
    target[lo:hi, pitch_idx] = 1.0
    return list(range(lo, hi))


def _time_to_frame(seconds: float, duration: float, frames: int, end: bool = False) -> int:
    pos = max(0.0, min(1.0, float(seconds) / max(1e-6, duration)))
    raw = pos * frames
    idx = int(math.ceil(raw)) if end else int(raw)
    return min(frames - 1 if not end else frames, max(0, idx))


def _time_to_frame_float(seconds: float, duration: float, frames: int) -> float:
    pos = max(0.0, min(1.0, float(seconds) / max(1e-6, duration)))
    return max(0.0, min(float(frames - 1), pos * frames))


def _clip_shift(value: float, clip: float) -> float:
    clip = max(0.05, float(clip))
    return max(-clip, min(clip, float(value))) / clip


def _duration_to_unit(seconds: float, max_duration_seconds: float) -> float:
    max_duration_seconds = max(0.1, float(max_duration_seconds))
    return max(0.0, min(1.0, math.log1p(max(0.0, seconds)) / math.log1p(max_duration_seconds)))


def _duration_bucket(seconds: float, bounds: tuple[float, ...] | list[float]) -> int:
    value = max(0.0, float(seconds))
    for idx, bound in enumerate(bounds):
        if value < float(bound):
            return idx
    return len(bounds)
