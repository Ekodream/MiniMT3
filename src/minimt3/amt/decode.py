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
    time_shift_clip_frames: float = 1.0,
    consume_note_energy: bool = False,
    energy_neighbor_pitches: int = 1,
    energy_overlap_ratio: float = 0.50,
    infer_onsets_from_frame_diff: bool = False,
    frame_diff_n: int = 2,
    frame_diff_scale: float = 1.0,
    frame_diff_min_onset: float = 0.0,
    frame_diff_context_threshold: float = 0.0,
    frame_diff_context_window_frames: int = 0,
    frame_diff_context_min_pitches: int = 0,
) -> list[NoteEvent]:
    onset = torch.sigmoid(outputs["onset_logits"])[0].detach().cpu()
    frame = torch.sigmoid(outputs["frame_logits"])[0].detach().cpu()
    offset = torch.sigmoid(outputs["offset_logits"])[0].detach().cpu()
    velocity = torch.sigmoid(outputs["velocity_logits"])[0].detach().cpu()
    duration_pred = None
    if use_duration_head and "duration_logits" in outputs:
        duration_pred = torch.sigmoid(outputs["duration_logits"])[0].detach().cpu()
    onset_shift = None
    offset_shift = None
    if "onset_shift_logits" in outputs:
        onset_shift = torch.tanh(outputs["onset_shift_logits"])[0].detach().cpu()
    if "offset_shift_logits" in outputs:
        offset_shift = torch.tanh(outputs["offset_shift_logits"])[0].detach().cpu()
    onset_for_peaks = onset
    if onset_frame_fusion_weight > 0.0:
        weight = max(0.0, min(1.0, float(onset_frame_fusion_weight)))
        onset_for_peaks = torch.maximum(onset, (1.0 - weight) * onset + weight * frame)
    if infer_onsets_from_frame_diff:
        inferred = _inferred_onsets_from_frame_diff(
            onset_for_peaks,
            frame,
            n_diff=int(frame_diff_n),
            scale=float(frame_diff_scale),
        )
        inferred = _mask_inferred_onsets(
            inferred,
            onset,
            min_onset=float(frame_diff_min_onset),
            context_threshold=float(frame_diff_context_threshold),
            context_window_frames=int(frame_diff_context_window_frames),
            context_min_pitches=int(frame_diff_context_min_pitches),
        )
        onset_for_peaks = torch.maximum(onset_for_peaks, inferred)
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

    note_items: list[tuple[float, NoteEvent]] = []
    for _, start_idx, pitch_idx in selected:
        end_idx, offset_idx = _find_note_end(
            frame[:, pitch_idx],
            offset[:, pitch_idx],
            start_idx,
            min_frames=min_frames,
            frame_threshold=frame_threshold,
            offset_threshold=offset_threshold,
        )
        start_shift_frames = 0.0
        if onset_shift is not None:
            start_shift_frames = float(onset_shift[start_idx, pitch_idx]) * float(time_shift_clip_frames)
        start_seconds = (start_idx + start_shift_frames) * frame_seconds
        start_seconds = max(0.0, min(float(duration), start_seconds))
        frame_end = max((start_idx + min_frames) * frame_seconds, end_idx * frame_seconds)
        if offset_shift is not None and offset_idx is not None:
            offset_shift_frames = float(offset_shift[offset_idx, pitch_idx]) * float(time_shift_clip_frames)
            shifted_offset = (offset_idx + offset_shift_frames) * frame_seconds
            frame_end = max((start_idx + min_frames) * frame_seconds, shifted_offset)
        end_seconds = frame_end
        if duration_pred is not None:
            predicted = _unit_to_duration(float(duration_pred[start_idx, pitch_idx]), max_duration_seconds)
            duration_end = start_seconds + predicted
            if duration_extension_weight >= 1.0:
                end_seconds = max(frame_end, duration_end)
            elif duration_extension_weight > 0.0:
                end_seconds = max(
                    frame_end,
                    frame_end * (1.0 - duration_extension_weight) + duration_end * duration_extension_weight,
                )
        vel = int(round(float(velocity[start_idx, pitch_idx]) * 127))
        end_seconds = max(start_seconds + min_note_seconds, min(float(duration), end_seconds))
        note_items.append(
            (
                float(onset_for_peaks[start_idx, pitch_idx]),
                NoteEvent(
                pitch=PITCH_MIN + pitch_idx,
                start=start_seconds,
                end=end_seconds,
                velocity=max(1, min(127, vel)),
                ),
            )
        )
    if consume_note_energy:
        notes = _consume_duplicate_energy(
            note_items,
            neighbor_pitches=int(energy_neighbor_pitches),
            overlap_ratio=float(energy_overlap_ratio),
        )
    else:
        notes = [note for _, note in note_items]
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


def _inferred_onsets_from_frame_diff(
    onsets: torch.Tensor,
    frames: torch.Tensor,
    n_diff: int = 2,
    scale: float = 1.0,
) -> torch.Tensor:
    """Infer onset-like peaks from rapid frame-energy increases."""
    n_diff = max(1, int(n_diff))
    diffs = []
    for delta in range(1, n_diff + 1):
        padded = torch.cat([torch.zeros(delta, frames.shape[1]), frames], dim=0)
        diff = padded[delta:] - padded[:-delta]
        diff[:delta] = 0.0
        diffs.append(diff.clamp_min(0.0))
    inferred = torch.stack(diffs, dim=0).amin(dim=0)
    max_inferred = float(inferred.max())
    target_peak = max(0.6, float(onsets.max()))
    if max_inferred <= 1e-8:
        return torch.zeros_like(onsets)
    return inferred * (target_peak / max_inferred) * max(0.0, float(scale))


def _mask_inferred_onsets(
    inferred: torch.Tensor,
    onsets: torch.Tensor,
    min_onset: float,
    context_threshold: float,
    context_window_frames: int,
    context_min_pitches: int,
) -> torch.Tensor:
    direct = onsets >= float(min_onset) if min_onset > 0.0 else torch.ones_like(onsets, dtype=torch.bool)
    if context_threshold <= 0.0 or context_min_pitches <= 0:
        return torch.where(direct, inferred, torch.zeros_like(inferred))
    strong = onsets >= float(context_threshold)
    context_counts = strong.float().sum(dim=1, keepdim=True)
    window = max(0, int(context_window_frames))
    if window > 0:
        expanded = []
        for idx in range(strong.shape[0]):
            lo = max(0, idx - window)
            hi = min(strong.shape[0], idx + window + 1)
            expanded.append(context_counts[lo:hi].max(dim=0).values)
        context_counts = torch.stack(expanded, dim=0)
    chord_context = context_counts.expand_as(onsets) >= int(context_min_pitches)
    return torch.where(direct | chord_context, inferred, torch.zeros_like(inferred))


def _find_note_end(
    frame_values: torch.Tensor,
    offset_values: torch.Tensor,
    start_idx: int,
    min_frames: int,
    frame_threshold: float,
    offset_threshold: float,
) -> tuple[int, int | None]:
    low_frame_streak = 0
    for i in range(start_idx + min_frames, frame_values.numel()):
        if float(offset_values[i]) >= offset_threshold:
            return i + 1, i
        if float(frame_values[i]) < frame_threshold:
            low_frame_streak += 1
            if low_frame_streak >= 2:
                return max(start_idx + min_frames, i - 1), None
        else:
            low_frame_streak = 0
    return frame_values.numel(), None


def _unit_to_duration(value: float, max_duration_seconds: float) -> float:
    max_duration_seconds = max(0.1, float(max_duration_seconds))
    value = max(0.0, min(1.0, float(value)))
    return math.expm1(value * math.log1p(max_duration_seconds))


def _consume_duplicate_energy(
    note_items: list[tuple[float, NoteEvent]],
    neighbor_pitches: int,
    overlap_ratio: float,
) -> list[NoteEvent]:
    """Keep strong notes first and suppress same/near-pitch overlapping echoes."""
    if not note_items:
        return []
    neighbor_pitches = max(0, int(neighbor_pitches))
    overlap_ratio = max(0.0, min(1.0, float(overlap_ratio)))
    kept: list[tuple[float, NoteEvent]] = []
    for score, note in sorted(note_items, key=lambda item: (item[0], item[1].velocity), reverse=True):
        duplicate = False
        for _, prev in kept:
            if abs(note.pitch - prev.pitch) > neighbor_pitches:
                continue
            if _note_overlap_ratio(note, prev) >= overlap_ratio:
                duplicate = True
                break
        if not duplicate:
            kept.append((score, note))
    return [note for _, note in kept]


def _note_overlap_ratio(a: NoteEvent, b: NoteEvent) -> float:
    overlap = min(a.end, b.end) - max(a.start, b.start)
    if overlap <= 0.0:
        return 0.0
    shorter = max(1e-6, min(a.end - a.start, b.end - b.start))
    return overlap / shorter


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
