from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from music21 import chord, converter, duration, instrument, metadata, note, pitch, stream

from minimt3.symbolic.cleanup import split_hands
from minimt3.symbolic.events import NoteEvent


def notes_to_score(
    notes: list[NoteEvent],
    title: str = "MiniMT3-Piano Transcription",
    seconds_per_quarter: float = 0.5,
) -> stream.Score:
    right, left = split_hands(notes)
    score = stream.Score()
    score.metadata = metadata.Metadata(title=title)
    right_part = stream.Part(id="right")
    right_part.insert(0, instrument.Piano())
    left_part = stream.Part(id="left")
    left_part.insert(0, instrument.Piano())
    _fill_part(right_part, right, seconds_per_quarter)
    _fill_part(left_part, left, seconds_per_quarter)
    score.insert(0, right_part)
    score.insert(0, left_part)
    return score


def write_musicxml(path: str | Path, notes: list[NoteEvent], title: str = "MiniMT3-Piano") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    score = notes_to_score(notes, title=title)
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


def _fill_part(part: stream.Part, notes: list[NoteEvent], seconds_per_quarter: float) -> None:
    by_start: dict[float, list[NoteEvent]] = {}
    for item in notes:
        by_start.setdefault(item.start, []).append(item)
    last_offset = 0.0
    for start in sorted(by_start):
        q_offset = start / seconds_per_quarter
        if q_offset > last_offset:
            rest = note.Rest()
            rest.duration = duration.Duration(q_offset - last_offset)
            part.insert(last_offset, rest)
        group = by_start[start]
        q_len = max(0.25, max(n.end - n.start for n in group) / seconds_per_quarter)
        if len(group) == 1:
            element = note.Note(pitch.Pitch(midi=group[0].pitch))
        else:
            element = chord.Chord([pitch.Pitch(midi=n.pitch) for n in group])
        element.duration = duration.Duration(q_len)
        part.insert(q_offset, element)
        last_offset = max(last_offset, q_offset + q_len)
    part.makeMeasures(inPlace=True)


def _find_musescore() -> str | None:
    for name in ["musescore", "mscore", "MuseScore4", "musescore4"]:
        path = shutil.which(name)
        if path:
            return path
    return None


def validate_musicxml(path: str | Path) -> None:
    converter.parse(str(path))
