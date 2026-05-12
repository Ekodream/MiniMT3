from __future__ import annotations

from minimt3.symbolic.events import NoteEvent, PedalEvent


def offset_events(
    notes: list[NoteEvent],
    pedals: list[PedalEvent],
    offset: float,
) -> tuple[list[NoteEvent], list[PedalEvent]]:
    return (
        [NoteEvent(n.pitch, n.start + offset, n.end + offset, n.velocity) for n in notes],
        [PedalEvent(p.start + offset, p.end + offset) for p in pedals],
    )


def merge_overlapping_notes(notes: list[NoteEvent], tolerance: float = 0.05) -> list[NoteEvent]:
    notes = sorted(notes, key=lambda n: (n.pitch, n.start, n.end))
    merged: list[NoteEvent] = []
    for note in notes:
        if (
            merged
            and merged[-1].pitch == note.pitch
            and abs(merged[-1].start - note.start) <= tolerance
            and abs(merged[-1].end - note.end) <= tolerance
        ):
            merged[-1].velocity = max(merged[-1].velocity, note.velocity)
            merged[-1].end = max(merged[-1].end, note.end)
        else:
            merged.append(note)
    merged.sort(key=lambda n: (n.start, n.pitch, n.end))
    return merged
