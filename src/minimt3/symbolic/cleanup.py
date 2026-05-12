from __future__ import annotations

from minimt3.symbolic.events import NoteEvent, PedalEvent


def pedal_aware_cleanup(
    notes: list[NoteEvent],
    pedals: list[PedalEvent],
    min_duration: float = 0.04,
    duplicate_gap: float = 0.03,
) -> list[NoteEvent]:
    """Extend notes inside sustain pedal regions and suppress tiny duplicate fragments."""
    cleaned = [NoteEvent(n.pitch, n.start, max(n.end, n.start + min_duration), n.velocity) for n in notes]
    for pedal in pedals:
        for note in cleaned:
            if pedal.start <= note.end <= pedal.end and note.start < pedal.end:
                note.end = max(note.end, pedal.end)

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


def split_hands(notes: list[NoteEvent], split_pitch: int = 60) -> tuple[list[NoteEvent], list[NoteEvent]]:
    right = [n for n in notes if n.pitch >= split_pitch]
    left = [n for n in notes if n.pitch < split_pitch]
    return right, left
