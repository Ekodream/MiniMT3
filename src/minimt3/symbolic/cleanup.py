from __future__ import annotations

from minimt3.symbolic.events import NoteEvent, PedalEvent


def pedal_aware_cleanup(
    notes: list[NoteEvent],
    pedals: list[PedalEvent],
    min_duration: float = 0.04,
    duplicate_gap: float = 0.03,
    max_extension: float = 1.5,
    same_pitch_margin: float = 0.02,
) -> list[NoteEvent]:
    """Extend notes inside sustain pedal regions and suppress tiny duplicate fragments."""
    cleaned = [NoteEvent(n.pitch, n.start, max(n.end, n.start + min_duration), n.velocity) for n in notes]
    by_pitch: dict[int, list[NoteEvent]] = {}
    for note in cleaned:
        by_pitch.setdefault(note.pitch, []).append(note)
    next_same_pitch: dict[int, float | None] = {}
    for pitch_notes in by_pitch.values():
        pitch_notes.sort(key=lambda n: (n.start, n.end))
        for idx, note in enumerate(pitch_notes):
            next_same_pitch[id(note)] = pitch_notes[idx + 1].start if idx + 1 < len(pitch_notes) else None

    for pedal in pedals:
        for note in cleaned:
            if pedal.start <= note.end <= pedal.end and note.start < pedal.end:
                cap = min(pedal.end, note.end + max_extension)
                next_start = next_same_pitch.get(id(note))
                if next_start is not None:
                    cap = min(cap, max(note.end, next_start - same_pitch_margin))
                note.end = max(note.end, cap)

    cleaned.sort(key=lambda n: (n.pitch, n.start, n.end))
    merged: list[NoteEvent] = []
    for note in cleaned:
        if merged and merged[-1].pitch == note.pitch and note.start - merged[-1].end <= duplicate_gap:
            merged[-1].end = max(merged[-1].end, note.end)
            merged[-1].velocity = max(merged[-1].velocity, note.velocity)
        elif note.end - note.start >= min_duration:
            merged.append(note)
    merged.sort(key=lambda n: (n.start, n.pitch, n.end))
    return merged


def infer_sustain_pedals(
    notes: list[NoteEvent],
    total_duration: float | None = None,
    window_seconds: float = 0.5,
    min_notes_per_window: int = 3,
    min_region_seconds: float = 0.75,
    merge_gap_seconds: float = 0.35,
    tail_seconds: float = 0.25,
) -> list[PedalEvent]:
    """Estimate simple sustain regions from dense harmonic activity.

    This is a display/performance fallback for checkpoints without an explicit
    pedal head. It favors longer harmonic passages and avoids adding pedal to
    sparse single-note runs.
    """
    if not notes:
        return []
    if total_duration is None:
        total_duration = max(n.end for n in notes)
    total_duration = max(0.0, float(total_duration))
    window_seconds = max(0.1, float(window_seconds))
    bins = max(1, int(total_duration / window_seconds) + 1)
    counts = [0 for _ in range(bins)]
    polyphony = [0 for _ in range(bins)]
    for note in notes:
        onset_bin = min(bins - 1, max(0, int(note.start / window_seconds)))
        counts[onset_bin] += 1
        start_bin = min(bins - 1, max(0, int(note.start / window_seconds)))
        end_bin = min(bins - 1, max(start_bin, int(note.end / window_seconds)))
        for idx in range(start_bin, end_bin + 1):
            polyphony[idx] += 1

    regions: list[PedalEvent] = []
    active_start: float | None = None
    for idx, (count, active) in enumerate(zip(counts, polyphony)):
        is_active = count >= min_notes_per_window or active >= min_notes_per_window + 1
        t = idx * window_seconds
        if is_active and active_start is None:
            active_start = t
        elif not is_active and active_start is not None:
            end = min(total_duration, t + tail_seconds)
            if end - active_start >= min_region_seconds:
                regions.append(PedalEvent(active_start, end))
            active_start = None
    if active_start is not None:
        end = min(total_duration, total_duration + tail_seconds)
        if end - active_start >= min_region_seconds:
            regions.append(PedalEvent(active_start, end))
    return merge_pedals(regions, merge_gap_seconds=merge_gap_seconds)


def merge_pedals(pedals: list[PedalEvent], merge_gap_seconds: float = 0.1) -> list[PedalEvent]:
    pedals = sorted((p for p in pedals if p.end > p.start), key=lambda p: (p.start, p.end))
    merged: list[PedalEvent] = []
    for pedal in pedals:
        if merged and pedal.start - merged[-1].end <= merge_gap_seconds:
            merged[-1].end = max(merged[-1].end, pedal.end)
        else:
            merged.append(PedalEvent(pedal.start, pedal.end))
    return merged


def suppress_noisy_notes(
    notes: list[NoteEvent],
    min_duration: float = 0.06,
    min_velocity: int = 4,
    same_pitch_gap: float = 0.04,
    chord_tolerance: float = 0.05,
    max_chord_notes: int = 12,
) -> list[NoteEvent]:
    """Remove very short/weak fragments while preserving plausible chords."""
    filtered = [
        NoteEvent(n.pitch, n.start, max(n.end, n.start + min_duration), n.velocity)
        for n in notes
        if n.end > n.start and n.velocity >= min_velocity and n.end - n.start >= min_duration
    ]
    filtered.sort(key=lambda n: (n.pitch, n.start, -n.velocity, n.end))
    deduped: list[NoteEvent] = []
    for note in filtered:
        if deduped and note.pitch == deduped[-1].pitch and note.start - deduped[-1].end <= same_pitch_gap:
            deduped[-1].end = max(deduped[-1].end, note.end)
            deduped[-1].velocity = max(deduped[-1].velocity, note.velocity)
        else:
            deduped.append(note)

    if max_chord_notes <= 0:
        deduped.sort(key=lambda n: (n.start, n.pitch, n.end))
        return deduped

    groups: list[list[NoteEvent]] = []
    for note in sorted(deduped, key=lambda n: (n.start, -n.velocity, n.pitch)):
        if groups and abs(note.start - groups[-1][0].start) <= chord_tolerance:
            groups[-1].append(note)
        else:
            groups.append([note])
    limited: list[NoteEvent] = []
    for group in groups:
        keep = sorted(group, key=lambda n: (n.velocity, n.end - n.start), reverse=True)[:max_chord_notes]
        limited.extend(keep)
    limited.sort(key=lambda n: (n.start, n.pitch, n.end))
    return limited


def quantize_notes(notes: list[NoteEvent], step: float = 0.125, min_duration: float = 0.125) -> list[NoteEvent]:
    quantized: list[NoteEvent] = []
    for note in notes:
        start = round(note.start / step) * step
        end = round(note.end / step) * step
        if end <= start:
            end = start + min_duration
        quantized.append(NoteEvent(note.pitch, start, end, note.velocity))
    quantized.sort(key=lambda n: (n.start, n.pitch, n.end))
    return quantized


def prepare_score_notes(
    notes: list[NoteEvent],
    quantize_step: float = 0.125,
    min_duration: float = 0.125,
    min_velocity: int = 6,
    max_chord_notes: int = 10,
) -> list[NoteEvent]:
    """Build a cleaner note list for readable MusicXML export."""
    clean = suppress_noisy_notes(
        notes,
        min_duration=min_duration,
        min_velocity=min_velocity,
        same_pitch_gap=quantize_step * 0.75,
        chord_tolerance=quantize_step * 0.75,
        max_chord_notes=max_chord_notes,
    )
    quantized = quantize_notes(clean, step=quantize_step, min_duration=min_duration)
    merged: dict[tuple[int, float], NoteEvent] = {}
    for note in quantized:
        key = (note.pitch, note.start)
        if key not in merged or note.end - note.start > merged[key].end - merged[key].start:
            merged[key] = note
    out = list(merged.values())
    out.sort(key=lambda n: (n.start, n.pitch, n.end))
    return out


def split_hands(notes: list[NoteEvent], split_pitch: int = 60) -> tuple[list[NoteEvent], list[NoteEvent]]:
    right = [n for n in notes if n.pitch >= split_pitch]
    left = [n for n in notes if n.pitch < split_pitch]
    return right, left
