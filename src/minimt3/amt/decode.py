from __future__ import annotations

import torch

from minimt3.symbolic.events import NoteEvent, PITCH_MIN


def decode_dense_notes(
    outputs: dict[str, torch.Tensor],
    duration: float,
    onset_threshold: float = 0.45,
    frame_threshold: float = 0.35,
    offset_threshold: float = 0.35,
    min_note_seconds: float = 0.04,
    max_notes_per_second: float = 45.0,
    max_polyphony: int = 12,
    min_onset_gap_seconds: float = 0.06,
    min_frame_at_onset: float = 0.0,
) -> list[NoteEvent]:
    onset = torch.sigmoid(outputs["onset_logits"])[0].detach().cpu()
    frame = torch.sigmoid(outputs["frame_logits"])[0].detach().cpu()
    offset = torch.sigmoid(outputs["offset_logits"])[0].detach().cpu()
    velocity = torch.sigmoid(outputs["velocity_logits"])[0].detach().cpu()
    frames, pitches = onset.shape
    frame_seconds = float(duration) / max(1, frames)
    min_frames = max(1, int(round(min_note_seconds / max(1e-6, frame_seconds))))
    candidates: list[tuple[float, int, int]] = []
    max_notes = int(max_notes_per_second * max(0.1, duration))
    nms_frames = max(1, int(round(min_onset_gap_seconds / max(1e-6, frame_seconds))))

    for pitch_idx in range(pitches):
        peaks = _onset_peaks(onset[:, pitch_idx], onset_threshold)
        for start_idx in peaks:
            if min_frame_at_onset > 0.0 and float(frame[start_idx, pitch_idx]) < min_frame_at_onset:
                continue
            candidates.append((float(onset[start_idx, pitch_idx]), start_idx, pitch_idx))
    candidates.sort(key=lambda item: item[0], reverse=True)

    selected: list[tuple[float, int, int]] = []
    frame_counts: dict[int, int] = {}
    pitch_starts: dict[int, list[int]] = {}
    for score, start_idx, pitch_idx in candidates:
        if len(selected) >= max_notes:
            break
        if frame_counts.get(start_idx, 0) >= max_polyphony:
            continue
        if any(abs(start_idx - prev) <= nms_frames for prev in pitch_starts.get(pitch_idx, [])):
            continue
        selected.append((score, start_idx, pitch_idx))
        frame_counts[start_idx] = frame_counts.get(start_idx, 0) + 1
        pitch_starts.setdefault(pitch_idx, []).append(start_idx)

    notes: list[NoteEvent] = []
    for _, start_idx, pitch_idx in selected:
        end_idx = _find_note_end(
            frame[:, pitch_idx],
            offset[:, pitch_idx],
            start_idx,
            min_frames=min_frames,
            frame_threshold=frame_threshold,
            offset_threshold=offset_threshold,
        )
        vel = int(round(float(velocity[start_idx, pitch_idx]) * 127))
        notes.append(
            NoteEvent(
                pitch=PITCH_MIN + pitch_idx,
                start=start_idx * frame_seconds,
                end=max((start_idx + min_frames) * frame_seconds, end_idx * frame_seconds),
                velocity=max(1, min(127, vel)),
            )
        )
    notes.sort(key=lambda n: (n.start, -n.velocity, n.pitch))
    return notes


def _onset_peaks(values: torch.Tensor, threshold: float) -> list[int]:
    peaks: list[int] = []
    for i in range(values.numel()):
        value = float(values[i])
        if value < threshold:
            continue
        left = float(values[i - 1]) if i > 0 else -1.0
        right = float(values[i + 1]) if i + 1 < values.numel() else -1.0
        if value >= left and value >= right:
            if peaks and i - peaks[-1] <= 1 and value <= float(values[peaks[-1]]):
                continue
            peaks.append(i)
    return peaks


def _find_note_end(
    frame_values: torch.Tensor,
    offset_values: torch.Tensor,
    start_idx: int,
    min_frames: int,
    frame_threshold: float,
    offset_threshold: float,
) -> int:
    low_frame_streak = 0
    for i in range(start_idx + min_frames, frame_values.numel()):
        if float(offset_values[i]) >= offset_threshold:
            return i + 1
        if float(frame_values[i]) < frame_threshold:
            low_frame_streak += 1
            if low_frame_streak >= 2:
                return max(start_idx + min_frames, i - 1)
        else:
            low_frame_streak = 0
    return frame_values.numel()
