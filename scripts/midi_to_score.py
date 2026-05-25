#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from minimt3.amt.presets import apply_score_preset
from minimt3.symbolic.events import load_midi_events
from minimt3.symbolic.midi_io import write_midi
from minimt3.symbolic.score_polish import ScorePolishConfig, polish_score_notes
from minimt3.symbolic.score_render import score_notation_metrics, write_musicxml
from minimt3.utils import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a performance MIDI to polished score MIDI/MusicXML.")
    parser.add_argument("--midi", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--score_preset", default="score_demo_4_4")
    parser.add_argument("--title", default="MiniMT3 Score")
    parser.add_argument("--score_time_signature")
    parser.add_argument("--score_key_signature")
    parser.add_argument("--score_tempo_bpm", type=float)
    parser.add_argument("--score_voice_mode", choices=["single", "dual_staff_2voice"])
    parser.add_argument("--score_split_ties", action="store_true")
    parser.add_argument("--score_hide_filler_rests", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out)
    score_cfg = apply_score_preset({}, args.score_preset)
    notes, pedals = load_midi_events(args.midi)
    tempo_bpm = float(args.score_tempo_bpm or score_cfg.get("tempo_bpm") or 100.0)
    seconds_per_quarter = 60.0 / max(20.0, tempo_bpm)
    time_signature = args.score_time_signature or score_cfg.get("time_signature", "4/4")
    key_signature = args.score_key_signature or score_cfg.get("key_signature")
    beat_divisions = _parse_divisions(str(score_cfg.get("score_beat_divisions", "2,4")))
    polished = polish_score_notes(
        notes,
        pedals=pedals,
        config=ScorePolishConfig(
            key_signature=key_signature,
            time_signature=time_signature,
            tempo_bpm=tempo_bpm,
            beat_divisions=beat_divisions,
            allow_tuplets=bool(score_cfg.get("score_allow_tuplets", False)),
            chord_tolerance_seconds=float(score_cfg.get("score_chord_tolerance_seconds", 0.08)),
            min_note_beats=float(score_cfg.get("score_min_note_beats", 0.25)),
            max_note_beats=float(score_cfg.get("score_max_note_beats", 6.0)),
            min_velocity=int(score_cfg.get("score_min_velocity", 8)),
            max_chord_notes=int(score_cfg.get("score_max_chord_notes", 10)),
            max_notes_per_beat=int(score_cfg.get("score_max_notes_per_beat", 5)),
            max_overlap_beats=float(score_cfg.get("score_max_overlap_beats", 0.0)),
            max_short_rest_beats=float(score_cfg.get("score_max_short_rest_beats", 0.5)),
            chord_snap_seconds=float(score_cfg.get("score_chord_snap_seconds", 0.10)),
            chord_snap_max_spread_beats=float(score_cfg.get("score_chord_snap_max_spread_beats", 0.22)),
            lock_chord_durations=bool(score_cfg.get("score_lock_chord_durations", True)),
            chord_lock_max_duration_spread_beats=float(
                score_cfg.get("score_chord_lock_max_duration_spread_beats", 0.75)
            ),
        ),
    )
    stem = Path(args.midi).stem
    score_midi = out_dir / f"{stem}_score.mid"
    musicxml = out_dir / f"{stem}.musicxml"
    debug_json = out_dir / f"{stem}_debug.json"
    write_midi(score_midi, polished.notes, pedals)
    write_musicxml(
        musicxml,
        polished.notes,
        title=args.title,
        seconds_per_quarter=polished.seconds_per_quarter,
        key_signature=polished.key_signature,
        time_signature=polished.time_signature,
        tempo_bpm=polished.tempo_bpm,
        right_notes=polished.right_notes,
        left_notes=polished.left_notes,
        beat_divisions=polished.beat_divisions,
        pedals=pedals,
        voice_mode=str(args.score_voice_mode or score_cfg.get("score_voice_mode", "dual_staff_2voice")),
        split_ties=bool(args.score_split_ties or score_cfg.get("score_split_ties", True)),
        hide_filler_rests=bool(args.score_hide_filler_rests or score_cfg.get("score_hide_filler_rests", True)),
    )
    notation = score_notation_metrics(
        polished.notes,
        seconds_per_quarter=polished.seconds_per_quarter,
        key_signature=polished.key_signature,
        time_signature=polished.time_signature,
        right_notes=polished.right_notes,
        left_notes=polished.left_notes,
        beat_divisions=polished.beat_divisions,
        performance_note_count=len(notes),
    )
    write_json(
        debug_json,
        {
            "midi": str(args.midi),
            "score_midi": str(score_midi),
            "musicxml": str(musicxml),
            "notes": len(notes),
            "score_notes": len(polished.notes),
            "score_polish": polished.to_json(),
            "score_notation": notation,
        },
    )
    print(f"midi_to_score score_midi={score_midi} musicxml={musicxml} debug={debug_json}", flush=True)


def _parse_divisions(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return tuple(sorted({max(1, int(part)) for part in parts})) or (2, 4)


if __name__ == "__main__":
    main()
