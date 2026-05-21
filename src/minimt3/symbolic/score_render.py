from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from music21 import (
    chord,
    converter,
    duration,
    dynamics,
    expressions,
    instrument,
    key as m21key,
    layout,
    metadata,
    meter,
    note,
    pitch,
    stream,
    tempo,
)

from minimt3.symbolic.cleanup import split_hands
from minimt3.symbolic.events import NoteEvent, PedalEvent


def notes_to_score(
    notes: list[NoteEvent],
    title: str = "MiniMT3-Piano Transcription",
    seconds_per_quarter: float = 0.5,
    key_signature: str | None = None,
    time_signature: str = "4/4",
    tempo_bpm: float | None = None,
    right_notes: list[NoteEvent] | None = None,
    left_notes: list[NoteEvent] | None = None,
    beat_divisions: tuple[int, ...] = (4,),
    pedals: list[PedalEvent] | None = None,
) -> stream.Score:
    right, left = (right_notes, left_notes) if right_notes is not None and left_notes is not None else split_hands(notes)
    score = stream.Score()
    score.metadata = metadata.Metadata(title=title)
    right_part = stream.Part(id="right")
    right_part.insert(0, instrument.Piano())
    left_part = stream.Part(id="left")
    left_part.insert(0, instrument.Piano())
    _insert_score_context(
        right_part,
        notes=right,
        key_signature=key_signature,
        time_signature=time_signature,
        tempo_bpm=tempo_bpm,
        pedals=pedals or [],
    )
    _insert_score_context(
        left_part,
        notes=left,
        key_signature=key_signature,
        time_signature=time_signature,
        tempo_bpm=None,
        pedals=[],
    )
    _fill_part(right_part, right, seconds_per_quarter, beat_divisions=beat_divisions)
    _fill_part(left_part, left, seconds_per_quarter, beat_divisions=beat_divisions)
    score.insert(0, right_part)
    score.insert(0, left_part)
    try:
        score.insert(0, layout.StaffGroup([right_part, left_part], symbol="brace", barTogether=True))
    except Exception:
        pass
    return score


def write_musicxml(
    path: str | Path,
    notes: list[NoteEvent],
    title: str = "MiniMT3-Piano",
    seconds_per_quarter: float = 0.5,
    key_signature: str | None = None,
    time_signature: str = "4/4",
    tempo_bpm: float | None = None,
    right_notes: list[NoteEvent] | None = None,
    left_notes: list[NoteEvent] | None = None,
    beat_divisions: tuple[int, ...] = (4,),
    pedals: list[PedalEvent] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    score = notes_to_score(
        notes,
        title=title,
        seconds_per_quarter=seconds_per_quarter,
        key_signature=key_signature,
        time_signature=time_signature,
        tempo_bpm=tempo_bpm,
        right_notes=right_notes,
        left_notes=left_notes,
        beat_divisions=beat_divisions,
        pedals=pedals,
    )
    score.write("musicxml", fp=str(path))
    return path


def render_score(
    musicxml_path: str | Path,
    png_path: str | Path | None = None,
    pdf_path: str | Path | None = None,
    svg_path: str | Path | None = None,
) -> dict[str, str]:
    musicxml_path = Path(musicxml_path)
    outputs: dict[str, str] = {}
    musescore = _find_musescore()
    if musescore:
        for out_path in [png_path, pdf_path]:
            if out_path:
                out = Path(out_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run([musescore, "-o", str(out), str(musicxml_path)], check=True)
                outputs[out.suffix.lstrip(".")] = str(out)
    elif png_path or pdf_path:
        outputs["musescore_missing"] = "MuseScore CLI not found; skipped PNG/PDF rendering."

    verovio = shutil.which("verovio")
    if svg_path and verovio:
        out = Path(svg_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([verovio, str(musicxml_path), "-o", str(out)], check=True)
        outputs["svg"] = str(out)
    elif svg_path:
        outputs["verovio_missing"] = "Verovio CLI not found; skipped SVG rendering."
    return outputs


def _fill_part(
    part: stream.Part,
    notes: list[NoteEvent],
    seconds_per_quarter: float,
    beat_divisions: tuple[int, ...] = (4,),
) -> None:
    by_start: dict[float, list[NoteEvent]] = {}
    for item in notes:
        q_start = _quantize_quarter(item.start / seconds_per_quarter, beat_divisions, min_value=0.0)
        by_start.setdefault(q_start, []).append(item)
    last_offset = 0.0
    for q_offset in sorted(by_start):
        if q_offset > last_offset:
            rest = note.Rest()
            rest.duration = duration.Duration(
                _quantize_quarter(q_offset - last_offset, beat_divisions, min_value=_min_quarter(beat_divisions))
            )
            part.insert(last_offset, rest)
        group = by_start[q_offset]
        q_len = _quantize_quarter(
            max(n.end - n.start for n in group) / seconds_per_quarter,
            beat_divisions,
            min_value=_min_quarter(beat_divisions),
        )
        if len(group) == 1:
            element = note.Note(pitch.Pitch(midi=group[0].pitch))
        else:
            element = chord.Chord([pitch.Pitch(midi=n.pitch) for n in group])
        element.duration = duration.Duration(q_len)
        part.insert(q_offset, element)
        last_offset = max(last_offset, q_offset + q_len)
    part.makeMeasures(inPlace=True)


def _insert_score_context(
    part: stream.Part,
    notes: list[NoteEvent],
    key_signature: str | None,
    time_signature: str,
    tempo_bpm: float | None,
    pedals: list[PedalEvent],
) -> None:
    if key_signature:
        try:
            part.insert(0, _make_key(key_signature))
        except Exception:
            pass
    try:
        part.insert(0, meter.TimeSignature(time_signature))
    except Exception:
        pass
    if tempo_bpm:
        part.insert(0, tempo.MetronomeMark(number=float(tempo_bpm)))
    if notes:
        part.insert(0, dynamics.Dynamic(_velocity_to_dynamic(notes)))
    if pedals:
        text = expressions.TextExpression("con pedale")
        part.insert(0, text)


def _velocity_to_dynamic(notes: list[NoteEvent]) -> str:
    avg = sum(n.velocity for n in notes) / max(1, len(notes))
    if avg < 38:
        return "pp"
    if avg < 55:
        return "p"
    if avg < 72:
        return "mp"
    if avg < 90:
        return "mf"
    return "f"


def _make_key(key_signature: str) -> m21key.Key:
    value = str(key_signature).strip()
    parts = value.split()
    if len(parts) >= 2 and parts[-1].lower() in {"major", "minor"}:
        return m21key.Key(" ".join(parts[:-1]), parts[-1].lower())
    return m21key.Key(value)


def _quantize_quarter(
    value: float,
    beat_divisions: tuple[int, ...] = (4,),
    min_value: float = 0.0,
) -> float:
    divisions = tuple(sorted({max(1, int(d)) for d in beat_divisions})) or (4,)
    best = 0.0
    best_error = float("inf")
    for div in divisions:
        grid = 1.0 / div
        candidate = round(float(value) / grid) * grid
        error = abs(candidate - float(value)) + div * 1e-6
        if error < best_error:
            best_error = error
            best = candidate
    return max(min_value, best)


def _min_quarter(beat_divisions: tuple[int, ...]) -> float:
    return 1.0 / max(1, max(beat_divisions or (4,)))


def _find_musescore() -> str | None:
    for name in ["musescore", "mscore", "MuseScore4", "musescore4"]:
        path = shutil.which(name)
        if path:
            return path
    return None


def validate_musicxml(path: str | Path) -> None:
    converter.parse(str(path))
