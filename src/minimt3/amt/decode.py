from __future__ import annotations

import math

import torch

from minimt3.symbolic.events import NoteEvent, PITCH_MIN, PedalEvent


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
    onset_frame_fusion_weight: float = 0.0,
    chord_onset_threshold: float | None = None,
    chord_frame_threshold: float = 0.35,
    chord_window_frames: int = 1,
    chord_score_ratio: float = 0.75,
    onset_peak_prominence: float = 0.0,
    max_notes_per_start_window: int | None = None,
    start_window_seconds: float = 0.08,
    use_duration_head: bool = True,
    max_duration_seconds: float = 8.0,
    duration_extension_weight: float = 1.0,
) -> list[NoteEvent]:
    onset = torch.sigmoid(outputs["onset_logits"])[0].detach().cpu()
    frame = torch.sigmoid(outputs["frame_logits"])[0].detach().cpu()
    offset = torch.sigmoid(outputs["offset_logits"])[0].detach().cpu()
    velocity = torch.sigmoid(outputs["velocity_logits"])[0].detach().cpu()
    duration_pred = None
    if use_duration_head and "duration_logits" in outputs:
        duration_pred = torch.sigmoid(outputs["duration_logits"])[0].detach().cpu()
    onset_for_peaks = onset
    if onset_frame_fusion_weight > 0.0:
        weight = max(0.0, min(1.0, float(onset_frame_fusion_weight)))
        onset_for_peaks = torch.maximum(onset, (1.0 - weight) * onset + weight * frame)
    frames, pitches = onset.shape
    frame_seconds = float(duration) / max(1, frames)
    min_frames = max(1, int(round(min_note_seconds / max(1e-6, frame_seconds))))
    candidates: list[tuple[float, int, int]] = []
    max_notes = int(max_notes_per_second * max(0.1, duration))
    nms_frames = max(1, int(round(min_onset_gap_seconds / max(1e-6, frame_seconds))))

    for pitch_idx in range(pitches):
        peaks = _onset_peaks(onset_for_peaks[:, pitch_idx], onset_threshold, prominence=onset_peak_prominence)
        for start_idx in peaks:
            if min_frame_at_onset > 0.0 and float(frame[start_idx, pitch_idx]) < min_frame_at_onset:
                continue
            score = float(onset_for_peaks[start_idx, pitch_idx])
            candidates.append((score, start_idx, pitch_idx))

    if chord_onset_threshold is not None:
        chord_threshold = float(chord_onset_threshold)
        frame_anchor_scores: dict[int, float] = {}
        for score, start_idx, _ in candidates:
            frame_anchor_scores[start_idx] = max(score, frame_anchor_scores.get(start_idx, 0.0))
        strong_frames = sorted(frame_anchor_scores)
        existing = {(start_idx, pitch_idx) for _, start_idx, pitch_idx in candidates}
        for strong_idx in strong_frames:
            lo = max(0, strong_idx - int(chord_window_frames))
            hi = min(frames, strong_idx + int(chord_window_frames) + 1)
            anchor_score = frame_anchor_scores.get(strong_idx, chord_threshold)
            adaptive_threshold = max(chord_threshold, anchor_score * max(0.0, min(1.0, chord_score_ratio)))
            for pitch_idx in range(pitches):
                if (strong_idx, pitch_idx) in existing:
                    continue
                local = onset_for_peaks[lo:hi, pitch_idx]
                if local.numel() <= 0:
                    continue
                local_value, local_offset = torch.max(local, dim=0)
                idx = lo + int(local_offset)
                if float(local_value) >= adaptive_threshold and float(frame[idx, pitch_idx]) >= chord_frame_threshold:
                    score = float(local_value) * 0.98
                    candidates.append((score, idx, pitch_idx))
                    existing.add((idx, pitch_idx))
    candidates.sort(key=lambda item: item[0], reverse=True)

    selected: list[tuple[float, int, int]] = []
    frame_counts: dict[int, int] = {}
    pitch_starts: dict[int, list[int]] = {}
    window_frames = max(0, int(round(start_window_seconds / max(1e-6, frame_seconds))))
    for score, start_idx, pitch_idx in candidates:
        if len(selected) >= max_notes:
            break
        if frame_counts.get(start_idx, 0) >= max_polyphony:
            continue
        if max_notes_per_start_window is not None and window_frames > 0:
            local_count = sum(1 for _, prev_start, _ in selected if abs(start_idx - prev_start) <= window_frames)
            if local_count >= max_notes_per_start_window:
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
        frame_end = max((start_idx + min_frames) * frame_seconds, end_idx * frame_seconds)
        end_seconds = frame_end
        if duration_pred is not None:
            predicted = _unit_to_duration(float(duration_pred[start_idx, pitch_idx]), max_duration_seconds)
            duration_end = start_idx * frame_seconds + predicted
            if duration_extension_weight >= 1.0:
                end_seconds = max(frame_end, duration_end)
            elif duration_extension_weight > 0.0:
                end_seconds = max(
                    frame_end,
                    frame_end * (1.0 - duration_extension_weight) + duration_end * duration_extension_weight,
                )
        vel = int(round(float(velocity[start_idx, pitch_idx]) * 127))
        notes.append(
            NoteEvent(
                pitch=PITCH_MIN + pitch_idx,
                start=start_idx * frame_seconds,
                end=end_seconds,
                velocity=max(1, min(127, vel)),
            )
        )
    notes.sort(key=lambda n: (n.start, -n.velocity, n.pitch))
    return notes


def _onset_peaks(values: torch.Tensor, threshold: float, prominence: float = 0.0) -> list[int]:
    peaks: list[int] = []
    for i in range(values.numel()):
        value = float(values[i])
        if value < threshold:
            continue
        left = float(values[i - 1]) if i > 0 else -1.0
        right = float(values[i + 1]) if i + 1 < values.numel() else -1.0
        if prominence > 0.0 and value - max(left, right) < prominence:
            continue
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


def _unit_to_duration(value: float, max_duration_seconds: float) -> float:
    max_duration_seconds = max(0.1, float(max_duration_seconds))
    value = max(0.0, min(1.0, float(value)))
    return math.expm1(value * math.log1p(max_duration_seconds))


def decode_dense_pedals(
    outputs: dict[str, torch.Tensor],
    duration: float,
    threshold: float = 0.5,
    min_pedal_seconds: float = 0.2,
    merge_gap_seconds: float = 0.08,
) -> list[PedalEvent]:
    if "pedal_logits" not in outputs:
        return []
    pedal = torch.sigmoid(outputs["pedal_logits"])[0, :, 0].detach().cpu()
    frames = pedal.numel()
    if frames == 0:
        return []
    frame_seconds = float(duration) / max(1, frames)
    min_frames = max(1, int(round(min_pedal_seconds / max(1e-6, frame_seconds))))
    regions: list[PedalEvent] = []
    start_idx: int | None = None
    for idx, value in enumerate(pedal):
        active = float(value) >= threshold
        if active and start_idx is None:
            start_idx = idx
        elif not active and start_idx is not None:
            if idx - start_idx >= min_frames:
                regions.append(PedalEvent(start_idx * frame_seconds, idx * frame_seconds))
            start_idx = None
    if start_idx is not None and frames - start_idx >= min_frames:
        regions.append(PedalEvent(start_idx * frame_seconds, frames * frame_seconds))
    if not regions:
        return []
    merged: list[PedalEvent] = []
    for pedal_event in regions:
        if merged and pedal_event.start - merged[-1].end <= merge_gap_seconds:
            merged[-1].end = max(merged[-1].end, pedal_event.end)
        else:
            merged.append(pedal_event)
    return merged
