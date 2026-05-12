from __future__ import annotations

from pathlib import Path

import pretty_midi

from minimt3.symbolic.events import NoteEvent, PedalEvent


def write_midi(
    path: str | Path,
    notes: list[NoteEvent],
    pedals: list[PedalEvent] | None = None,
    program: int = 0,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program, name="Piano")
    for note in notes:
        if note.end <= note.start:
            continue
        instrument.notes.append(
            pretty_midi.Note(
                velocity=max(1, min(127, int(note.velocity))),
                pitch=int(note.pitch),
                start=max(0.0, float(note.start)),
                end=max(float(note.start) + 0.01, float(note.end)),
            )
        )
    for pedal in pedals or []:
        instrument.control_changes.append(pretty_midi.ControlChange(64, 127, pedal.start))
        instrument.control_changes.append(pretty_midi.ControlChange(64, 0, pedal.end))
    midi.instruments.append(instrument)
    midi.write(str(path))
    return path


def read_midi(path: str | Path) -> tuple[list[NoteEvent], list[PedalEvent]]:
    from minimt3.symbolic.events import load_midi_events

    return load_midi_events(path)
