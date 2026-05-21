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
    min_note_seconds: float = 0.02
    include_pedal: bool = False
    include_duration: bool = False
    max_duration_seconds: float = 8.0


def encoder_frame_count(feature_frames: int) -> int:
    first = (int(feature_frames) + 1) // 2
    return max(1, (first + 1) // 2)


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
    duration_mask = torch.zeros(frames, NUM_PITCHES) if cfg.include_duration else None

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
            duration_mask=duration_mask,
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
            duration_mask=duration_mask,
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
        targets["duration_mask"] = duration_mask
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
    duration_mask: torch.Tensor | None = None,
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
        lo = max(0, start_idx - cfg.onset_width_frames)
        hi = min(frames, start_idx + cfg.onset_width_frames + 1)
        onset[lo:hi, pitch_idx] = 1.0
        onset_mask[lo:hi, pitch_idx] = 1.0
        velocity[lo:hi, pitch_idx] = max(1, min(127, int(note.velocity))) / 127.0
        if note_duration is not None and duration_mask is not None:
            normalized = _duration_to_unit(note.end - note.start, cfg.max_duration_seconds)
            note_duration[lo:hi, pitch_idx] = normalized
            duration_mask[lo:hi, pitch_idx] = 1.0
    if note.end < duration - 1e-4:
        offset_idx = min(frames - 1, max(0, end_idx - 1))
        lo = max(0, offset_idx - cfg.offset_width_frames)
        hi = min(frames, offset_idx + cfg.offset_width_frames + 1)
        offset[lo:hi, pitch_idx] = 1.0


def _time_to_frame(seconds: float, duration: float, frames: int, end: bool = False) -> int:
    pos = max(0.0, min(1.0, float(seconds) / max(1e-6, duration)))
    raw = pos * frames
    idx = int(math.ceil(raw)) if end else int(raw)
    return min(frames - 1 if not end else frames, max(0, idx))


def _duration_to_unit(seconds: float, max_duration_seconds: float) -> float:
    max_duration_seconds = max(0.1, float(max_duration_seconds))
    return max(0.0, min(1.0, math.log1p(max(0.0, seconds)) / math.log1p(max_duration_seconds)))
