from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
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
    tie,
)

from minimt3.symbolic.cleanup import split_hands
from minimt3.symbolic.events import NoteEvent, PedalEvent


@dataclass(frozen=True)
class _RenderNote:
    pitch: int
    start: float
    end: float
    velocity: int

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class _RenderSegment:
    pitch: int
    start: float
    end: float
    velocity: int
    tie_type: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


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
    voice_mode: str = "dual_staff_2voice",
    split_ties: bool = True,
    hide_filler_rests: bool = True,
) -> stream.Score:
    right, left = (right_notes, left_notes) if right_notes is not None and left_notes is not None else split_hands(notes)
    bar_quarters = _time_signature_quarters(time_signature)
    right_render = _prepare_render_notes(right, seconds_per_quarter, beat_divisions)
    left_render = _prepare_render_notes(left, seconds_per_quarter, beat_divisions)
    all_render_notes = right_render + left_render
    total_end = max([n.end for n in all_render_notes] + [bar_quarters])
    total_measures = max(1, int(math.ceil((total_end - 1e-7) / bar_quarters)))
    active_measures = _active_measures(all_render_notes, bar_quarters)
    score = stream.Score()
    score.metadata = metadata.Metadata(title=title)
    right_part = stream.Part(id="right")
    right_part.insert(0, instrument.Piano())
    left_part = stream.Part(id="left")
    left_part.insert(0, instrument.Piano())
    _fill_part(
        right_part,
        right,
        seconds_per_quarter,
        render_notes=right_render,
        total_measures=total_measures,
        active_measures=active_measures,
        beat_divisions=beat_divisions,
        key_signature=key_signature,
        time_signature=time_signature,
        tempo_bpm=tempo_bpm,
        pedals=pedals or [],
        staff="right",
        voice_mode=voice_mode,
        split_ties=split_ties,
        hide_filler_rests=hide_filler_rests,
    )
    _fill_part(
        left_part,
        left,
        seconds_per_quarter,
        render_notes=left_render,
        total_measures=total_measures,
        active_measures=active_measures,
        beat_divisions=beat_divisions,
        key_signature=key_signature,
        time_signature=time_signature,
        tempo_bpm=None,
        pedals=[],
        staff="left",
        voice_mode=voice_mode,
        split_ties=split_ties,
        hide_filler_rests=hide_filler_rests,
    )
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
    voice_mode: str = "dual_staff_2voice",
    split_ties: bool = True,
    hide_filler_rests: bool = True,
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
        voice_mode=voice_mode,
        split_ties=split_ties,
        hide_filler_rests=hide_filler_rests,
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
    render_notes: list[_RenderNote] | None = None,
    total_measures: int | None = None,
    active_measures: set[int] | None = None,
    beat_divisions: tuple[int, ...] = (4,),
    key_signature: str | None = None,
    time_signature: str = "4/4",
    tempo_bpm: float | None = None,
    pedals: list[PedalEvent] | None = None,
    staff: str = "right",
    voice_mode: str = "dual_staff_2voice",
    split_ties: bool = True,
    hide_filler_rests: bool = True,
) -> None:
    bar_quarters = _time_signature_quarters(time_signature)
    render_notes = list(render_notes) if render_notes is not None else _prepare_render_notes(notes, seconds_per_quarter, beat_divisions)
    voices = _assign_render_voices(render_notes, staff=staff, voice_mode=voice_mode)
    total_end = max([n.end for n in render_notes] + [bar_quarters])
    measure_count = int(total_measures) if total_measures is not None else max(1, int(math.ceil((total_end - 1e-7) / bar_quarters)))
    active_measures = active_measures or _active_measures(render_notes, bar_quarters)
    segments_by_voice = {
        voice_id: _segments_by_measure(voice_notes, bar_quarters, split_ties=split_ties)
        for voice_id, voice_notes in voices.items()
    }
    multi_voice = len(voices) > 1
    for measure_idx in range(measure_count):
        measure = stream.Measure(number=measure_idx + 1)
        if measure_idx == 0:
            _insert_score_context(
                measure,
                notes=notes,
                key_signature=key_signature,
                time_signature=time_signature,
                tempo_bpm=tempo_bpm,
                pedals=pedals or [],
            )
        if multi_voice:
            measure_has_any_voice = any(
                segments_by_voice.get(voice_id, {}).get(measure_idx) for voice_id in voices
            )
            for voice_id in sorted(voices):
                voice_segments = segments_by_voice.get(voice_id, {}).get(measure_idx, [])
                voice = stream.Voice(id=str(voice_id))
                has_content = _fill_voice_measure(
                    voice,
                    voice_segments,
                    bar_quarters=bar_quarters,
                    beat_divisions=beat_divisions,
                    key_signature=key_signature,
                    primary_voice=voice_id == "1" and (bool(voice_segments) or not measure_has_any_voice),
                    hide_filler_rests=hide_filler_rests,
                    hide_empty_measure=hide_filler_rests and measure_idx in active_measures and not voice_segments,
                )
                if has_content:
                    measure.insert(0, voice)
        else:
            _fill_voice_measure(
                measure,
                segments_by_voice.get("1", {}).get(measure_idx, []),
                bar_quarters=bar_quarters,
                beat_divisions=beat_divisions,
                key_signature=key_signature,
                primary_voice=True,
                hide_filler_rests=hide_filler_rests,
                hide_empty_measure=hide_filler_rests and measure_idx in active_measures,
            )
        part.append(measure)


def _insert_score_context(
    part: stream.Stream,
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


def score_notation_metrics(
    notes: list[NoteEvent],
    seconds_per_quarter: float = 0.5,
    key_signature: str | None = None,
    time_signature: str = "4/4",
    right_notes: list[NoteEvent] | None = None,
    left_notes: list[NoteEvent] | None = None,
    beat_divisions: tuple[int, ...] = (4,),
    voice_mode: str = "dual_staff_2voice",
    split_ties: bool = True,
    hide_filler_rests: bool = True,
    performance_note_count: int | None = None,
    key_signature_source: str = "auto",
    time_signature_source: str = "auto",
) -> dict[str, float | str]:
    del key_signature
    right, left = (right_notes, left_notes) if right_notes is not None and left_notes is not None else split_hands(notes)
    bar_quarters = _time_signature_quarters(time_signature)
    parts = {
        "right": _prepare_render_notes(right, seconds_per_quarter, beat_divisions),
        "left": _prepare_render_notes(left, seconds_per_quarter, beat_divisions),
    }
    all_render_notes = parts["right"] + parts["left"]
    total_end = max([n.end for n in all_render_notes] + [bar_quarters])
    measure_count = max(1, int(math.ceil((total_end - 1e-7) / bar_quarters)))
    active_measures = _active_measures(all_render_notes, bar_quarters)
    visible_rests = 0
    hidden_rests = 0
    collisions = 0
    secondary_voice_notes = 0
    for staff, render_notes in parts.items():
        voices = _assign_render_voices(render_notes, staff=staff, voice_mode=voice_mode)
        voice_segments_by_id = {
            voice_id: _segments_by_measure(voice_notes, bar_quarters, split_ties=split_ties)
            for voice_id, voice_notes in voices.items()
        }
        for voice_id, voice_notes in voices.items():
            if voice_id != "1":
                secondary_voice_notes += len(voice_notes)
            collisions += _voice_collision_count(voice_notes)
            segments = voice_segments_by_id[voice_id]
            for measure_idx in range(measure_count):
                measure_has_any_voice = any(item.get(measure_idx) for item in voice_segments_by_id.values())
                measure_segments = segments.get(measure_idx, [])
                stats = _voice_rest_stats(
                    measure_segments,
                    bar_quarters=bar_quarters,
                    beat_divisions=beat_divisions,
                    primary_voice=voice_id == "1" and (bool(measure_segments) or not measure_has_any_voice),
                    hide_filler_rests=hide_filler_rests,
                    hide_empty_measure=hide_filler_rests and measure_idx in active_measures and not measure_segments,
                )
                visible_rests += stats["visible"]
                hidden_rests += stats["hidden"]
    crossing_notes = [n for n in all_render_notes if _crosses_barline(n.start, n.end, bar_quarters)]
    tied_crossings = crossing_notes if split_ties else []
    long_note_tie_rate = 1.0 if not crossing_notes else len(tied_crossings) / float(len(crossing_notes))
    chord_groups = [group for group in _group_render_notes_by_start(all_render_notes) if len(group) >= 2]
    chord_notes = sum(len(group) for group in chord_groups)
    return {
        "rest_density": visible_rests / max(1.0, float(measure_count * 2)),
        "visible_rest_count": float(visible_rests),
        "hidden_rest_count": float(hidden_rests),
        "long_note_tie_rate": long_note_tie_rate,
        "chord_verticality": chord_notes / max(1.0, float(len(all_render_notes))),
        "voice_collision_count": float(collisions),
        "secondary_voice_notes": float(secondary_voice_notes),
        "score_notes_per_measure": len(all_render_notes) / max(1.0, float(measure_count)),
        "score_notes_per_performance_note": len(all_render_notes) / max(1.0, float(performance_note_count or len(notes))),
        "measure_count": float(measure_count),
        "bar_quarters": float(bar_quarters),
        "time_signature_source": time_signature_source,
        "key_signature_source": key_signature_source,
    }


def _prepare_render_notes(
    notes: list[NoteEvent],
    seconds_per_quarter: float,
    beat_divisions: tuple[int, ...],
) -> list[_RenderNote]:
    min_q = _min_quarter(beat_divisions)
    prepared: list[_RenderNote] = []
    for item in notes:
        if item.end <= item.start:
            continue
        q_start = _quantize_quarter(item.start / max(1e-6, seconds_per_quarter), beat_divisions, min_value=0.0)
        q_end = _quantize_quarter(item.end / max(1e-6, seconds_per_quarter), beat_divisions, min_value=0.0)
        q_end = max(q_start + min_q, q_end)
        prepared.append(_RenderNote(int(item.pitch), q_start, q_end, int(item.velocity)))
    prepared.sort(key=lambda n: (n.start, n.pitch, n.end))
    return prepared


def _assign_render_voices(
    notes: list[_RenderNote],
    staff: str,
    voice_mode: str,
) -> dict[str, list[_RenderNote]]:
    if not notes or voice_mode == "single":
        return {"1": list(notes)}
    voice_count = 2 if voice_mode == "dual_staff_2voice" else 1
    if voice_count <= 1:
        return {"1": list(notes)}

    median_pitch = sorted(n.pitch for n in notes)[len(notes) // 2]
    long_threshold = 1.5 if staff == "right" else 1.0
    voices: dict[str, list[_RenderNote]] = {"1": [], "2": []}
    voice_end = {"1": 0.0, "2": 0.0}
    for group in _group_render_notes_by_start(notes):
        lower_sustains = [
            n
            for n in group
            if n.duration >= max(0.75, long_threshold * 0.75)
            and n.pitch <= median_pitch + (4 if staff == "right" else 9)
        ]
        if len(group) >= 3 and lower_sustains:
            sustain_ids = {id(n) for n in lower_sustains}
            moving = [n for n in group if id(n) not in sustain_ids]
            _add_voice_group(voices, voice_end, "2", lower_sustains)
            _add_voice_group(voices, voice_end, "1", moving)
            continue
        low_long = [
            n
            for n in group
            if n.duration >= long_threshold and n.pitch <= median_pitch + (3 if staff == "right" else 8)
        ]
        if 0 < len(low_long) < len(group):
            low_ids = {id(n) for n in low_long}
            high_or_short = [n for n in group if id(n) not in low_ids]
            _add_voice_group(voices, voice_end, "2", low_long)
            _add_voice_group(voices, voice_end, "1", high_or_short)
            continue
        choice = _choose_voice(group, voice_end, staff=staff, median_pitch=median_pitch, long_threshold=long_threshold)
        _add_voice_group(voices, voice_end, choice, group)
    for voice_notes in voices.values():
        voice_notes.sort(key=lambda n: (n.start, n.pitch, n.end))
    return voices


def _choose_voice(
    group: list[_RenderNote],
    voice_end: dict[str, float],
    staff: str,
    median_pitch: int,
    long_threshold: float,
) -> str:
    start = group[0].start
    mean_pitch = sum(n.pitch for n in group) / max(1, len(group))
    mean_duration = sum(n.duration for n in group) / max(1, len(group))
    prefer_second = mean_duration >= long_threshold and mean_pitch <= median_pitch + (2 if staff == "right" else 6)
    preferred = "2" if prefer_second else "1"
    other = "1" if preferred == "2" else "2"
    if voice_end[preferred] <= start + 1e-6:
        return preferred
    if voice_end[other] <= start + 1e-6:
        return other
    return min(("1", "2"), key=lambda voice_id: voice_end[voice_id])


def _add_voice_group(
    voices: dict[str, list[_RenderNote]],
    voice_end: dict[str, float],
    voice_id: str,
    group: list[_RenderNote],
) -> None:
    if not group:
        return
    voices[voice_id].extend(group)
    voice_end[voice_id] = max(voice_end[voice_id], max(n.end for n in group))


def _segments_by_measure(
    notes: list[_RenderNote],
    bar_quarters: float,
    split_ties: bool,
) -> dict[int, list[_RenderSegment]]:
    by_measure: dict[int, list[_RenderSegment]] = {}
    for item in notes:
        cur = item.start
        while cur < item.end - 1e-6:
            measure_idx = int(cur // bar_quarters)
            boundary = (measure_idx + 1) * bar_quarters
            seg_end = min(item.end, boundary)
            has_before = cur > item.start + 1e-6
            has_after = seg_end < item.end - 1e-6
            tie_type = None
            if split_ties:
                if has_before and has_after:
                    tie_type = "continue"
                elif has_before:
                    tie_type = "stop"
                elif has_after:
                    tie_type = "start"
            by_measure.setdefault(measure_idx, []).append(
                _RenderSegment(item.pitch, cur, seg_end, item.velocity, tie_type=tie_type)
            )
            cur = seg_end
    for segments in by_measure.values():
        segments.sort(key=lambda n: (n.start, n.pitch, n.end))
    return by_measure


def _fill_voice_measure(
    container: stream.Stream,
    segments: list[_RenderSegment],
    bar_quarters: float,
    beat_divisions: tuple[int, ...],
    key_signature: str | None,
    primary_voice: bool,
    hide_filler_rests: bool,
    hide_empty_measure: bool = False,
) -> bool:
    min_q = _min_quarter(beat_divisions)
    if not segments:
        if primary_voice:
            _insert_rest(container, 0.0, bar_quarters, hidden=hide_empty_measure)
            return True
        return False

    local_segments = sorted(segments, key=lambda n: (n.start, n.pitch, n.end))
    measure_start = math.floor(local_segments[0].start / bar_quarters) * bar_quarters
    groups = _group_segments_for_render(local_segments)
    cursor = 0.0
    has_content = False
    for group in groups:
        rel_start = max(0.0, group[0].start - measure_start)
        rel_end = min(bar_quarters, max(item.end for item in group) - measure_start)
        if rel_start > cursor + 1e-6:
            rest_len = _quantize_quarter(rel_start - cursor, beat_divisions, min_value=min_q)
            hidden = _hide_rest(rest_len, primary_voice, hide_filler_rests, leading=False, trailing=False)
            _insert_rest(container, cursor, rest_len, hidden=hidden)
            has_content = True
        element = _make_note_element(group, key_signature)
        element.duration = duration.Duration(max(min_q, rel_end - rel_start))
        if group[0].tie_type:
            element.tie = tie.Tie(group[0].tie_type)
        container.insert(rel_start, element)
        has_content = True
        cursor = max(cursor, rel_end)
    if cursor < bar_quarters - 1e-6:
        rest_len = _quantize_quarter(bar_quarters - cursor, beat_divisions, min_value=min_q)
        hidden = _hide_rest(rest_len, primary_voice, hide_filler_rests, leading=False, trailing=True)
        _insert_rest(container, cursor, rest_len, hidden=hidden)
        has_content = True
    return has_content


def _voice_rest_stats(
    segments: list[_RenderSegment],
    bar_quarters: float,
    beat_divisions: tuple[int, ...],
    primary_voice: bool,
    hide_filler_rests: bool,
    hide_empty_measure: bool = False,
) -> dict[str, int]:
    min_q = _min_quarter(beat_divisions)
    if not segments:
        if primary_voice:
            return {"visible": 0, "hidden": 1} if hide_empty_measure else {"visible": 1, "hidden": 0}
        return {"visible": 0, "hidden": 0}
    measure_start = math.floor(segments[0].start / bar_quarters) * bar_quarters
    cursor = 0.0
    visible = 0
    hidden = 0
    for group in _group_segments_for_render(segments):
        rel_start = max(0.0, group[0].start - measure_start)
        rel_end = min(bar_quarters, max(item.end for item in group) - measure_start)
        if rel_start > cursor + 1e-6:
            rest_len = _quantize_quarter(rel_start - cursor, beat_divisions, min_value=min_q)
            if _hide_rest(rest_len, primary_voice, hide_filler_rests, leading=False, trailing=False):
                hidden += 1
            else:
                visible += 1
        cursor = max(cursor, rel_end)
    if cursor < bar_quarters - 1e-6:
        rest_len = _quantize_quarter(bar_quarters - cursor, beat_divisions, min_value=min_q)
        if _hide_rest(rest_len, primary_voice, hide_filler_rests, leading=False, trailing=True):
            hidden += 1
        else:
            visible += 1
    return {"visible": visible, "hidden": hidden}


def _hide_rest(
    rest_len: float,
    primary_voice: bool,
    hide_filler_rests: bool,
    leading: bool,
    trailing: bool,
) -> bool:
    del rest_len, primary_voice, leading, trailing
    if not hide_filler_rests:
        return False
    return True


def _insert_rest(container: stream.Stream, offset: float, q_len: float, hidden: bool) -> None:
    rest = note.Rest()
    rest.duration = duration.Duration(max(0.0625, q_len))
    if hidden:
        _hide_music21_object(rest)
    container.insert(offset, rest)


def _hide_music21_object(element: object) -> None:
    try:
        element.style.hideObjectOnPrint = True
    except Exception:
        pass
    try:
        element.style.hidden = True
    except Exception:
        pass


def _make_note_element(
    group: list[_RenderSegment],
    key_signature: str | None,
) -> note.Note | chord.Chord:
    if len(group) == 1:
        return note.Note(_make_pitch(group[0].pitch, key_signature))
    return chord.Chord([_make_pitch(item.pitch, key_signature) for item in sorted(group, key=lambda n: n.pitch)])


def _group_segments_for_render(segments: list[_RenderSegment]) -> list[list[_RenderSegment]]:
    groups: dict[tuple[float, float, str | None], list[_RenderSegment]] = {}
    for item in segments:
        key = (round(item.start, 6), round(item.end, 6), item.tie_type)
        groups.setdefault(key, []).append(item)
    return [groups[key] for key in sorted(groups, key=lambda item: (item[0], item[1], item[2] or ""))]


def _group_render_notes_by_start(notes: list[_RenderNote]) -> list[list[_RenderNote]]:
    groups: list[list[_RenderNote]] = []
    for item in sorted(notes, key=lambda n: (n.start, n.pitch, n.end)):
        if groups and abs(item.start - groups[-1][0].start) <= 1e-6:
            groups[-1].append(item)
        else:
            groups.append([item])
    return groups


def _active_measures(notes: list[_RenderNote], bar_quarters: float) -> set[int]:
    active: set[int] = set()
    for item in notes:
        if item.end <= item.start:
            continue
        start_idx = int(max(0.0, item.start) // bar_quarters)
        end_idx = int(max(item.start, item.end - 1e-7) // bar_quarters)
        active.update(range(start_idx, end_idx + 1))
    return active


def _voice_collision_count(notes: list[_RenderNote]) -> int:
    collisions = 0
    last_start = -1.0
    last_end = 0.0
    for item in sorted(notes, key=lambda n: (n.start, n.pitch, n.end)):
        same_onset = abs(item.start - last_start) <= 1e-6
        if not same_onset and item.start < last_end - 1e-6:
            collisions += 1
        last_start = item.start
        last_end = max(last_end, item.end)
    return collisions


def _crosses_barline(start: float, end: float, bar_quarters: float) -> bool:
    if end <= start:
        return False
    return int(start // bar_quarters) != int((end - 1e-6) // bar_quarters)


def _time_signature_quarters(time_signature: str) -> float:
    try:
        return float(meter.TimeSignature(str(time_signature)).barDuration.quarterLength)
    except Exception:
        try:
            num, den = str(time_signature).split("/", 1)
            return max(1.0, float(num) * 4.0 / float(den))
        except Exception:
            return 4.0


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


def _make_pitch(midi: int, key_signature: str | None = None) -> pitch.Pitch:
    if not key_signature:
        return pitch.Pitch(midi=int(midi))
    prefer_flats = _prefer_flats(key_signature)
    names = (
        ["C", "D-", "D", "E-", "E", "F", "G-", "G", "A-", "A", "B-", "B"]
        if prefer_flats
        else ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    )
    pc = int(midi) % 12
    octave = int(midi) // 12 - 1
    return pitch.Pitch(f"{names[pc]}{octave}")


def _prefer_flats(key_signature: str) -> bool:
    value = str(key_signature).replace("♭", "b").strip()
    tonic = value.split()[0] if value else ""
    flat_keys = {"F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb"}
    sharp_keys = {"G", "D", "A", "E", "B", "F#", "C#"}
    if tonic in flat_keys:
        return True
    if tonic in sharp_keys:
        return False
    return "b" in tonic and "#" not in tonic


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
