from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pretty_midi

PITCH_MIN = 21
PITCH_MAX = 108


@dataclass
class NoteEvent:
    pitch: int
    start: float
    end: float
    velocity: int = 80


@dataclass
class PedalEvent:
    start: float
    end: float


@dataclass
class DecodeResult:
    notes: list[NoteEvent]
    pedals: list[PedalEvent]
    invalid_events: int
    total_events: int

    def to_json(self) -> dict:
        return {
            "notes": [asdict(n) for n in self.notes],
            "pedals": [asdict(p) for p in self.pedals],
            "invalid_events": self.invalid_events,
            "total_events": self.total_events,
            "invalid_event_rate": self.invalid_events / max(1, self.total_events),
        }


class EventCodec:
    """Piano-only MIDI-like event codec with fixed 10 ms time-shift steps by default."""

    def __init__(
        self,
        time_shift_ms: int = 10,
        max_time_shift_steps: int = 100,
        velocity_bins: int = 32,
    ):
        self.time_shift_ms = time_shift_ms
        self.step_seconds = time_shift_ms / 1000.0
        self.max_time_shift_steps = max_time_shift_steps
        self.velocity_bins = velocity_bins
        self.tokens = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "PEDAL_ON", "PEDAL_OFF"]
        self.tokens += [f"VELOCITY_{i}" for i in range(velocity_bins)]
        self.tokens += [f"TIME_SHIFT_{i}" for i in range(1, max_time_shift_steps + 1)]
        self.tokens += [f"NOTE_ON_{p}" for p in range(PITCH_MIN, PITCH_MAX + 1)]
        self.tokens += [f"NOTE_OFF_{p}" for p in range(PITCH_MIN, PITCH_MAX + 1)]
        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<EOS>"]

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def token_id(self, token: str) -> int:
        return self.token_to_id.get(token, self.token_to_id["<UNK>"])

    def token(self, token_id: int) -> str:
        return self.id_to_token.get(int(token_id), "<UNK>")

    def velocity_to_bin(self, velocity: int) -> int:
        return min(self.velocity_bins - 1, max(0, int(round(velocity / 127 * (self.velocity_bins - 1)))))

    def bin_to_velocity(self, bin_id: int) -> int:
        return int(round(max(0, min(self.velocity_bins - 1, bin_id)) / (self.velocity_bins - 1) * 127))

    def encode_midi_file(
        self,
        midi_path: str | Path,
        start: float = 0.0,
        end: float | None = None,
        add_special: bool = True,
    ) -> list[int]:
        notes, pedals = load_midi_events(midi_path, start=start, end=end)
        return self.encode_events(notes, pedals, start=start, add_special=add_special)

    def encode_events(
        self,
        notes: Iterable[NoteEvent],
        pedals: Iterable[PedalEvent] = (),
        start: float = 0.0,
        add_special: bool = True,
    ) -> list[int]:
        timeline: list[tuple[float, int, str]] = []
        for note in notes:
            if PITCH_MIN <= note.pitch <= PITCH_MAX and note.end > note.start:
                vel = self.velocity_to_bin(note.velocity)
                timeline.append((note.start, 1, f"VELOCITY_{vel}"))
                timeline.append((note.start, 2, f"NOTE_ON_{note.pitch}"))
                timeline.append((note.end, 0, f"NOTE_OFF_{note.pitch}"))
        for pedal in pedals:
            if pedal.end > pedal.start:
                timeline.append((pedal.start, 1, "PEDAL_ON"))
                timeline.append((pedal.end, 0, "PEDAL_OFF"))
        timeline.sort(key=lambda x: (x[0], x[1], x[2]))

        ids = [self.bos_id] if add_special else []
        cursor = start
        for time, _, token in timeline:
            if time < start:
                continue
            ids.extend(self._encode_time_shift(max(0.0, time - cursor)))
            ids.append(self.token_id(token))
            cursor = max(cursor, time)
        if add_special:
            ids.append(self.eos_id)
        return ids

    def decode(self, token_ids: Iterable[int], close_active: bool = True) -> DecodeResult:
        time = 0.0
        velocity = 80
        active: dict[int, tuple[float, int]] = {}
        pedal_start: float | None = None
        notes: list[NoteEvent] = []
        pedals: list[PedalEvent] = []
        invalid = 0
        total = 0

        for token_id in token_ids:
            token = self.token(int(token_id))
            if token in {"<PAD>", "<BOS>"}:
                continue
            total += 1
            if token == "<EOS>":
                break
            if token.startswith("TIME_SHIFT_"):
                time += int(token.rsplit("_", 1)[1]) * self.step_seconds
            elif token.startswith("VELOCITY_"):
                velocity = self.bin_to_velocity(int(token.rsplit("_", 1)[1]))
            elif token.startswith("NOTE_ON_"):
                pitch = int(token.rsplit("_", 1)[1])
                if pitch in active:
                    invalid += 1
                    start, old_velocity = active[pitch]
                    if time > start:
                        notes.append(NoteEvent(pitch, start, time, old_velocity))
                active[pitch] = (time, velocity)
            elif token.startswith("NOTE_OFF_"):
                pitch = int(token.rsplit("_", 1)[1])
                if pitch not in active:
                    invalid += 1
                    continue
                start, note_velocity = active.pop(pitch)
                if time <= start:
                    invalid += 1
                    continue
                notes.append(NoteEvent(pitch, start, time, note_velocity))
            elif token == "PEDAL_ON":
                if pedal_start is not None:
                    invalid += 1
                pedal_start = time
            elif token == "PEDAL_OFF":
                if pedal_start is None:
                    invalid += 1
                    continue
                if time > pedal_start:
                    pedals.append(PedalEvent(pedal_start, time))
                pedal_start = None
            else:
                invalid += 1

        if close_active:
            final_time = max(time, max((s for s, _ in active.values()), default=time) + 0.05)
            for pitch, (start, note_velocity) in active.items():
                if final_time > start:
                    notes.append(NoteEvent(pitch, start, final_time, note_velocity))
            if pedal_start is not None and time > pedal_start:
                pedals.append(PedalEvent(pedal_start, time))

        notes.sort(key=lambda n: (n.start, n.pitch, n.end))
        pedals.sort(key=lambda p: (p.start, p.end))
        return DecodeResult(notes, pedals, invalid, total)

    def _encode_time_shift(self, seconds: float) -> list[int]:
        steps = int(round(seconds / self.step_seconds))
        ids = []
        while steps > 0:
            chunk = min(steps, self.max_time_shift_steps)
            ids.append(self.token_id(f"TIME_SHIFT_{chunk}"))
            steps -= chunk
        return ids


def load_midi_events(
    midi_path: str | Path,
    start: float = 0.0,
    end: float | None = None,
) -> tuple[list[NoteEvent], list[PedalEvent]]:
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes: list[NoteEvent] = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            note_start, note_end = note.start, note.end
            if end is not None and note_start >= end:
                continue
            if note_end <= start:
                continue
            notes.append(
                NoteEvent(
                    pitch=note.pitch,
                    start=max(note_start, start) - start,
                    end=(min(note_end, end) if end is not None else note_end) - start,
                    velocity=note.velocity,
                )
            )

    pedals: list[PedalEvent] = []
    for instrument in midi.instruments:
        controls = [cc for cc in instrument.control_changes if cc.number == 64]
        if not controls:
            continue
        controls.sort(key=lambda cc: cc.time)
        active_start: float | None = None
        for cc in controls:
            if end is not None and cc.time > end:
                break
            if cc.value >= 64 and active_start is None:
                active_start = cc.time
            elif cc.value < 64 and active_start is not None:
                if cc.time > start and active_start < (end if end is not None else cc.time):
                    pedals.append(PedalEvent(max(active_start, start) - start, cc.time - start))
                active_start = None
        if active_start is not None and end is not None and active_start < end:
            pedals.append(PedalEvent(max(active_start, start) - start, end - start))
        break
    return notes, pedals
