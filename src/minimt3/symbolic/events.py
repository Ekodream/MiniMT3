from __future__ import annotations

from collections import Counter
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
    eos_hit: bool = False
    stop_reason: str = "decoded"

    def to_json(self) -> dict:
        return {
            "notes": [asdict(n) for n in self.notes],
            "pedals": [asdict(p) for p in self.pedals],
            "invalid_events": self.invalid_events,
            "total_events": self.total_events,
            "invalid_event_rate": self.invalid_events / max(1, self.total_events),
            "eos_hit": self.eos_hit,
            "stop_reason": self.stop_reason,
        }


class EventCodec:
    """MT3-style piano event codec.

    The default representation uses event families:
    SHIFT, VELOCITY, PITCH, and PEDAL. Note-off is encoded as
    VELOCITY_0 followed by PITCH_x; note-on is VELOCITY_v followed by PITCH_x.
    This reduces the local NOTE_ON/NOTE_OFF looping space and matches the
    event-state style used by MT3/Seq2Seq Piano.
    """

    def __init__(
        self,
        time_shift_ms: int = 10,
        max_time_shift_steps: int = 1000,
        velocity_bins: int = 32,
        time_mode: str = "absolute",
    ):
        if time_mode not in {"absolute", "relative"}:
            raise ValueError("time_mode must be 'absolute' or 'relative'")
        self.time_shift_ms = time_shift_ms
        self.step_seconds = time_shift_ms / 1000.0
        self.max_time_shift_steps = max_time_shift_steps
        self.velocity_bins = velocity_bins
        self.time_mode = time_mode
        self.tokens = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
        shift_start = 0 if time_mode == "absolute" else 1
        self.tokens += [f"SHIFT_{i}" for i in range(shift_start, max_time_shift_steps + 1)]
        self.tokens += [f"VELOCITY_{i}" for i in range(velocity_bins + 1)]
        self.tokens += [f"PITCH_{p}" for p in range(PITCH_MIN, PITCH_MAX + 1)]
        self.tokens += ["PEDAL_ON", "PEDAL_OFF", "TIE"]
        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.family_ranges = self._build_family_ranges()
        self.shift_token_ids = [self.token_to_id[t] for t in self.tokens if t.startswith("SHIFT_")]
        self.shift_steps = [int(self.id_to_token[i].rsplit("_", 1)[1]) for i in self.shift_token_ids]
        self.velocity_token_ids = [self.token_to_id[t] for t in self.tokens if t.startswith("VELOCITY_")]
        self.velocity_values = [int(self.id_to_token[i].rsplit("_", 1)[1]) for i in self.velocity_token_ids]
        self.pitch_token_ids = [self.token_to_id[t] for t in self.tokens if t.startswith("PITCH_")]
        self.pitch_values = [int(self.id_to_token[i].rsplit("_", 1)[1]) for i in self.pitch_token_ids]
        self._tensor_cache: dict[str, dict[str, object]] = {}

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

    def token_family(self, token_id: int | str) -> str:
        token = token_id if isinstance(token_id, str) else self.token(token_id)
        if token.startswith("SHIFT_"):
            return "SHIFT"
        if token.startswith("VELOCITY_"):
            return "VELOCITY"
        if token.startswith("PITCH_"):
            return "PITCH"
        if token.startswith("PEDAL_"):
            return "PEDAL"
        if token in {"<PAD>", "<BOS>", "<EOS>", "<UNK>"}:
            return token.strip("<>")
        return token

    def family_mask(self, family_weights: dict[str, float]) -> list[float]:
        return [family_weights.get(self.token_family(i), 1.0) for i in range(self.vocab_size)]

    def token_family_counts(self, token_ids: Iterable[int]) -> dict[str, int]:
        return dict(Counter(self.token_family(i) for i in token_ids))

    def constraint_tensors(self, device) -> dict[str, object]:
        import torch

        key = str(device)
        if key not in self._tensor_cache:
            self._tensor_cache[key] = {
                "shift_ids": torch.tensor(self.shift_token_ids, dtype=torch.long, device=device),
                "shift_steps": torch.tensor(self.shift_steps, dtype=torch.float32, device=device),
                "velocity_ids": torch.tensor(self.velocity_token_ids, dtype=torch.long, device=device),
                "velocity_values": torch.tensor(self.velocity_values, dtype=torch.long, device=device),
                "pitch_ids": torch.tensor(self.pitch_token_ids, dtype=torch.long, device=device),
                "pitch_values": torch.tensor(self.pitch_values, dtype=torch.long, device=device),
            }
        return self._tensor_cache[key]

    def velocity_to_bin(self, velocity: int) -> int:
        if velocity <= 0:
            return 0
        return min(self.velocity_bins, max(1, int(round(velocity / 127 * self.velocity_bins))))

    def bin_to_velocity(self, bin_id: int) -> int:
        if bin_id <= 0:
            return 0
        return int(round(max(1, min(self.velocity_bins, bin_id)) / self.velocity_bins * 127))

    def encode_midi_file(
        self,
        midi_path: str | Path,
        start: float = 0.0,
        end: float | None = None,
        add_special: bool = True,
        include_ties: bool = False,
    ) -> list[int]:
        notes, pedals = load_midi_events(midi_path, start=start, end=end)
        return self.encode_events(notes, pedals, add_special=add_special, include_ties=include_ties)

    def encode_events(
        self,
        notes: Iterable[NoteEvent],
        pedals: Iterable[PedalEvent] = (),
        add_special: bool = True,
        include_ties: bool = False,
    ) -> list[int]:
        del include_ties  # Reserved for a future state-prefix compatible manifest.
        cleaned_notes = _trim_overlapping_notes(_clean_notes(notes))
        cleaned_pedals = _clean_pedals(pedals)
        timeline: list[tuple[float, int, str]] = []
        for note in cleaned_notes:
            vel = self.velocity_to_bin(note.velocity)
            timeline.append((note.end, 0, "VELOCITY_0"))
            timeline.append((note.end, 1, f"PITCH_{note.pitch}"))
            timeline.append((note.start, 2, f"VELOCITY_{vel}"))
            timeline.append((note.start, 3, f"PITCH_{note.pitch}"))
        for pedal in cleaned_pedals:
            timeline.append((pedal.start, 2, "PEDAL_ON"))
            timeline.append((pedal.end, 0, "PEDAL_OFF"))
        timeline.sort(key=lambda x: (x[0], x[1], x[2]))

        ids = [self.bos_id] if add_special else []
        cursor = 0.0
        wrote_any_event = False
        current_velocity: int | None = None
        pedal_active = False
        for time, _, token in timeline:
            time = max(0.0, time)
            if self.time_mode == "absolute":
                if (not wrote_any_event and time > 0) or (
                    wrote_any_event and abs(time - cursor) > self.step_seconds / 2
                ):
                    ids.extend(self._encode_time_position(time))
            else:
                ids.extend(self._encode_time_shift(max(0.0, time - cursor)))
            cursor = max(cursor, time)
            if token.startswith("VELOCITY_"):
                velocity = int(token.rsplit("_", 1)[1])
                if current_velocity == velocity:
                    continue
                current_velocity = velocity
            elif token == "PEDAL_ON":
                if pedal_active:
                    continue
                pedal_active = True
            elif token == "PEDAL_OFF":
                if not pedal_active:
                    continue
                pedal_active = False
            ids.append(self.token_id(token))
            wrote_any_event = True
        if add_special:
            ids.append(self.eos_id)
        return ids

    def decode(
        self,
        token_ids: Iterable[int],
        close_active: bool = True,
        stop_reason: str = "decoded",
    ) -> DecodeResult:
        time = 0.0
        velocity = 80
        active: dict[int, tuple[float, int]] = {}
        pedal_start: float | None = None
        notes: list[NoteEvent] = []
        pedals: list[PedalEvent] = []
        invalid = 0
        total = 0
        eos_hit = False

        for token_id in token_ids:
            token = self.token(int(token_id))
            if token in {"<PAD>", "<BOS>"}:
                continue
            total += 1
            if token == "<EOS>":
                eos_hit = True
                break
            if token.startswith("SHIFT_"):
                shift = int(token.rsplit("_", 1)[1]) * self.step_seconds
                time = shift if self.time_mode == "absolute" else time + shift
            elif token.startswith("VELOCITY_"):
                velocity = self.bin_to_velocity(int(token.rsplit("_", 1)[1]))
            elif token.startswith("PITCH_"):
                pitch = int(token.rsplit("_", 1)[1])
                if velocity == 0:
                    if pitch not in active:
                        invalid += 1
                        continue
                    start, note_velocity = active.pop(pitch)
                    if time <= start:
                        invalid += 1
                        continue
                    notes.append(NoteEvent(pitch, start, time, note_velocity))
                else:
                    if pitch in active:
                        invalid += 1
                        start, old_velocity = active.pop(pitch)
                        if time > start:
                            notes.append(NoteEvent(pitch, start, time, old_velocity))
                    active[pitch] = (time, velocity)
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
            elif token == "TIE":
                continue
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
        return DecodeResult(notes, pedals, invalid, total, eos_hit=eos_hit, stop_reason=stop_reason)

    def _encode_time_position(self, seconds: float) -> list[int]:
        steps = min(self.max_time_shift_steps, max(0, int(round(seconds / self.step_seconds))))
        return [self.token_id(f"SHIFT_{steps}")]

    def _encode_time_shift(self, seconds: float) -> list[int]:
        steps = int(round(seconds / self.step_seconds))
        ids = []
        while steps > 0:
            chunk = min(steps, self.max_time_shift_steps)
            ids.append(self.token_id(f"SHIFT_{chunk}"))
            steps -= chunk
        return ids

    def _build_family_ranges(self) -> dict[str, tuple[int, int]]:
        ranges: dict[str, list[int]] = {}
        for token_id in range(len(self.tokens)):
            ranges.setdefault(self.token_family(token_id), []).append(token_id)
        return {k: (min(v), max(v)) for k, v in ranges.items()}


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
            clipped_start = max(note_start, start) - start
            clipped_end = (min(note_end, end) if end is not None else note_end) - start
            if clipped_end > clipped_start:
                notes.append(
                    NoteEvent(
                        pitch=note.pitch,
                        start=clipped_start,
                        end=clipped_end,
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
    return _trim_overlapping_notes(_clean_notes(notes)), _clean_pedals(pedals)


def _clean_notes(notes: Iterable[NoteEvent], min_duration: float = 0.01) -> list[NoteEvent]:
    clean: list[NoteEvent] = []
    for note in notes:
        if not (PITCH_MIN <= int(note.pitch) <= PITCH_MAX):
            continue
        start = max(0.0, float(note.start))
        end = max(0.0, float(note.end))
        if end - start < min_duration:
            continue
        clean.append(NoteEvent(int(note.pitch), start, end, max(1, min(127, int(note.velocity)))))
    return sorted(clean, key=lambda n: (n.pitch, n.start, n.end))


def _trim_overlapping_notes(notes: Iterable[NoteEvent]) -> list[NoteEvent]:
    by_pitch: dict[int, list[NoteEvent]] = {}
    for note in notes:
        by_pitch.setdefault(note.pitch, []).append(note)
    trimmed: list[NoteEvent] = []
    for pitch, pitch_notes in by_pitch.items():
        pitch_notes.sort(key=lambda n: (n.start, n.end))
        for i, note in enumerate(pitch_notes):
            end = note.end
            if i + 1 < len(pitch_notes):
                end = min(end, pitch_notes[i + 1].start)
            if end > note.start:
                trimmed.append(NoteEvent(pitch, note.start, end, note.velocity))
    return sorted(trimmed, key=lambda n: (n.start, n.pitch, n.end))


def _clean_pedals(pedals: Iterable[PedalEvent], min_duration: float = 0.01) -> list[PedalEvent]:
    clean = [
        PedalEvent(max(0.0, float(p.start)), max(0.0, float(p.end)))
        for p in pedals
        if float(p.end) - float(p.start) >= min_duration
    ]
    clean.sort(key=lambda p: (p.start, p.end))
    merged: list[PedalEvent] = []
    for pedal in clean:
        if merged and pedal.start <= merged[-1].end:
            merged[-1].end = max(merged[-1].end, pedal.end)
        else:
            merged.append(pedal)
    return merged
