#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from minimt3.amt.decode import decode_dense_notes, decode_dense_pedals
from minimt3.amt.model import DenseAMT, DenseAMTConfig
from minimt3.audio.features import LogMelConfig, LogMelExtractor
from minimt3.audio.preprocess import load_audio
from minimt3.symbolic.cleanup import (
    infer_sustain_pedals,
    merge_pedals,
    pedal_aware_cleanup,
    prepare_score_notes,
    suppress_noisy_notes,
)
from minimt3.symbolic.midi_io import write_midi
from minimt3.symbolic.score_polish import ScorePolishConfig, polish_score_notes
from minimt3.utils import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense-AMT inference.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="outputs/amt_demo")
    parser.add_argument("--window_seconds", type=float, default=2.0)
    parser.add_argument("--overlap_seconds", type=float, default=0.25)
    parser.add_argument("--onset_threshold", type=float)
    parser.add_argument("--frame_threshold", type=float)
    parser.add_argument("--offset_threshold", type=float)
    parser.add_argument("--min_note_seconds", type=float)
    parser.add_argument("--max_notes_per_second", type=float)
    parser.add_argument("--max_polyphony", type=int)
    parser.add_argument("--min_onset_gap_seconds", type=float)
    parser.add_argument("--min_frame_at_onset", type=float)
    parser.add_argument("--merge_tolerance_seconds", type=float)
    parser.add_argument("--onset_frame_fusion_weight", type=float)
    parser.add_argument("--chord_onset_threshold", type=float)
    parser.add_argument("--chord_frame_threshold", type=float)
    parser.add_argument("--chord_window_frames", type=int)
    parser.add_argument("--disable_chord_recovery", action="store_true")
    parser.add_argument("--chord_score_ratio", type=float)
    parser.add_argument("--onset_peak_prominence", type=float)
    parser.add_argument("--max_notes_per_start_window", type=int)
    parser.add_argument("--start_window_seconds", type=float)
    parser.add_argument("--disable_duration_head", action="store_true")
    parser.add_argument("--max_duration_seconds", type=float)
    parser.add_argument("--duration_extension_weight", type=float)
    parser.add_argument("--pedal_threshold", type=float)
    parser.add_argument("--disable_sustain_heuristic", action="store_true")
    parser.add_argument("--performance_min_note_seconds", type=float)
    parser.add_argument("--performance_min_velocity", type=int)
    parser.add_argument("--sustain_max_extension", type=float)
    parser.add_argument("--score_quantize_seconds", type=float)
    parser.add_argument("--score_min_note_seconds", type=float)
    parser.add_argument("--score_min_velocity", type=int)
    parser.add_argument("--score_max_chord_notes", type=int)
    parser.add_argument("--disable_score_polish", action="store_true")
    parser.add_argument("--key_signature")
    parser.add_argument("--time_signature")
    parser.add_argument("--tempo_bpm", type=float)
    parser.add_argument("--score_beat_divisions")
    parser.add_argument("--score_chord_tolerance_seconds", type=float)
    parser.add_argument("--score_max_note_beats", type=float)
    parser.add_argument("--score_min_note_beats", type=float)
    parser.add_argument("--score_max_notes_per_beat", type=int)
    parser.add_argument("--score_merge_extension_seconds", type=float)
    parser.add_argument("--disable_score_overlap_trim", action="store_true")
    parser.add_argument("--score_max_overlap_beats", type=float)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["config"]
    decode_cfg = cfg.get("decode", {})
    onset_threshold = float(args.onset_threshold or decode_cfg.get("onset_threshold", 0.55))
    frame_threshold = float(args.frame_threshold or decode_cfg.get("frame_threshold", 0.25))
    offset_threshold = float(args.offset_threshold or decode_cfg.get("offset_threshold", 0.25))
    min_note_seconds = float(args.min_note_seconds or decode_cfg.get("min_note_seconds", 0.04))
    max_notes_per_second = float(args.max_notes_per_second or decode_cfg.get("max_notes_per_second", 24.0))
    max_polyphony = int(args.max_polyphony or decode_cfg.get("max_polyphony", 12))
    min_onset_gap_seconds = float(
        args.min_onset_gap_seconds
        if args.min_onset_gap_seconds is not None
        else decode_cfg.get("min_onset_gap_seconds", 0.06)
    )
    min_frame_at_onset = float(
        args.min_frame_at_onset
        if args.min_frame_at_onset is not None
        else decode_cfg.get("min_frame_at_onset", 0.0)
    )
    merge_tolerance_seconds = float(args.merge_tolerance_seconds or decode_cfg.get("merge_tolerance_seconds", 0.06))
    onset_frame_fusion_weight = float(
        args.onset_frame_fusion_weight
        if args.onset_frame_fusion_weight is not None
        else decode_cfg.get("onset_frame_fusion_weight", 0.0)
    )
    if args.disable_chord_recovery:
        chord_onset_threshold = None
    else:
        chord_onset_threshold = (
            float(args.chord_onset_threshold)
            if args.chord_onset_threshold is not None
            else decode_cfg.get("chord_onset_threshold")
        )
    chord_frame_threshold = float(
        args.chord_frame_threshold
        if args.chord_frame_threshold is not None
        else decode_cfg.get("chord_frame_threshold", 0.35)
    )
    chord_window_frames = int(args.chord_window_frames or decode_cfg.get("chord_window_frames", 1))
    chord_score_ratio = float(
        args.chord_score_ratio
        if args.chord_score_ratio is not None
        else decode_cfg.get("chord_score_ratio", 0.75)
    )
    onset_peak_prominence = float(
        args.onset_peak_prominence
        if args.onset_peak_prominence is not None
        else decode_cfg.get("onset_peak_prominence", 0.0)
    )
    max_notes_per_start_window = (
        int(args.max_notes_per_start_window)
        if args.max_notes_per_start_window is not None
        else decode_cfg.get("max_notes_per_start_window")
    )
    start_window_seconds = float(
        args.start_window_seconds
        if args.start_window_seconds is not None
        else decode_cfg.get("start_window_seconds", 0.08)
    )
    max_duration_seconds = float(args.max_duration_seconds or decode_cfg.get("max_duration_seconds", 8.0))
    duration_extension_weight = float(
        args.duration_extension_weight
        if args.duration_extension_weight is not None
        else decode_cfg.get("duration_extension_weight", 1.0)
    )
    pedal_threshold = float(args.pedal_threshold or decode_cfg.get("pedal_threshold", 0.50))
    performance_min_note_seconds = float(
        args.performance_min_note_seconds or decode_cfg.get("performance_min_note_seconds", 0.06)
    )
    performance_min_velocity = int(args.performance_min_velocity or decode_cfg.get("performance_min_velocity", 4))
    sustain_max_extension = float(args.sustain_max_extension or decode_cfg.get("sustain_max_extension", 1.5))
    score_quantize_seconds = float(args.score_quantize_seconds or decode_cfg.get("score_quantize_seconds", 0.125))
    score_min_note_seconds = float(args.score_min_note_seconds or decode_cfg.get("score_min_note_seconds", 0.125))
    score_min_velocity = int(args.score_min_velocity or decode_cfg.get("score_min_velocity", 6))
    score_max_chord_notes = int(args.score_max_chord_notes or decode_cfg.get("score_max_chord_notes", 10))
    key_signature = args.key_signature or decode_cfg.get("key_signature")
    time_signature = args.time_signature or decode_cfg.get("time_signature", "4/4")
    tempo_bpm = args.tempo_bpm if args.tempo_bpm is not None else decode_cfg.get("tempo_bpm")
    score_beat_divisions = _parse_int_tuple(
        args.score_beat_divisions or str(decode_cfg.get("score_beat_divisions", "2,3,4"))
    )
    score_chord_tolerance_seconds = float(
        args.score_chord_tolerance_seconds or decode_cfg.get("score_chord_tolerance_seconds", 0.055)
    )
    score_max_note_beats = float(args.score_max_note_beats or decode_cfg.get("score_max_note_beats", 4.0))
    score_min_note_beats = float(args.score_min_note_beats or decode_cfg.get("score_min_note_beats", 0.25))
    score_max_notes_per_beat = int(args.score_max_notes_per_beat or decode_cfg.get("score_max_notes_per_beat", 8))
    score_merge_extension_seconds = float(
        args.score_merge_extension_seconds or decode_cfg.get("score_merge_extension_seconds", 0.25)
    )
    score_max_overlap_beats = float(args.score_max_overlap_beats or decode_cfg.get("score_max_overlap_beats", 0.0))
    audio_cfg = LogMelConfig(**cfg.get("audio", {}))
    model = DenseAMT(DenseAMTConfig(**cfg.get("model", {}))).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    waveform = load_audio(args.audio, audio_cfg.sample_rate)
    extractor = LogMelExtractor(audio_cfg).to(device)
    sr = audio_cfg.sample_rate
    total_seconds = waveform.shape[-1] / sr
    step = max(0.1, args.window_seconds - args.overlap_seconds)
    starts = []
    t = 0.0
    while t < total_seconds:
        starts.append(t)
        t += step
        if t + 0.05 >= total_seconds:
            break
    notes = []
    pedals = []
    debug = []
    with torch.no_grad():
        for start in starts:
            end = min(total_seconds, start + args.window_seconds)
            segment = waveform[:, int(start * sr) : int(end * sr)]
            features = extractor(segment.to(device))
            out = model(features)
            window_notes = decode_dense_notes(
                out,
                duration=end - start,
                onset_threshold=onset_threshold,
                frame_threshold=frame_threshold,
                offset_threshold=offset_threshold,
                min_note_seconds=min_note_seconds,
                max_notes_per_second=max_notes_per_second,
                max_polyphony=max_polyphony,
                min_onset_gap_seconds=min_onset_gap_seconds,
                min_frame_at_onset=min_frame_at_onset,
                onset_frame_fusion_weight=onset_frame_fusion_weight,
                chord_onset_threshold=chord_onset_threshold,
                chord_frame_threshold=chord_frame_threshold,
                chord_window_frames=chord_window_frames,
                chord_score_ratio=chord_score_ratio,
                onset_peak_prominence=onset_peak_prominence,
                max_notes_per_start_window=max_notes_per_start_window,
                start_window_seconds=start_window_seconds,
                use_duration_head=not args.disable_duration_head,
                max_duration_seconds=max_duration_seconds,
                duration_extension_weight=duration_extension_weight,
            )
            for note in window_notes:
                note.start += start
                note.end += start
            notes.extend(window_notes)
            window_pedals = decode_dense_pedals(out, duration=end - start, threshold=pedal_threshold)
            for pedal in window_pedals:
                pedal.start += start
                pedal.end += start
            pedals.extend(window_pedals)
            debug.append({"start": start, "end": end, "notes": len(window_notes), "pedals": len(window_pedals)})
    raw_note_count = len(notes)
    notes = merge_overlapping_window_notes(notes, tolerance=merge_tolerance_seconds)
    merged_note_count = len(notes)
    notes = suppress_noisy_notes(
        notes,
        min_duration=performance_min_note_seconds,
        min_velocity=performance_min_velocity,
        same_pitch_gap=merge_tolerance_seconds,
        chord_tolerance=merge_tolerance_seconds,
        max_chord_notes=max_polyphony,
    )
    clean_note_count = len(notes)
    pedals = merge_pedals(pedals, merge_gap_seconds=0.12)
    predicted_pedal_count = len(pedals)
    if not pedals and not args.disable_sustain_heuristic:
        pedals = infer_sustain_pedals(notes, total_duration=total_seconds)
    score_base_notes = pedal_aware_cleanup(
        notes,
        pedals,
        min_duration=max(performance_min_note_seconds, score_min_note_seconds * 0.5),
        duplicate_gap=merge_tolerance_seconds,
        max_extension=score_merge_extension_seconds,
    )
    performance_notes = pedal_aware_cleanup(
        notes,
        pedals,
        min_duration=performance_min_note_seconds,
        duplicate_gap=merge_tolerance_seconds,
        max_extension=sustain_max_extension,
    )
    score_polish_debug = None
    polished = None
    if args.disable_score_polish:
        score_notes = prepare_score_notes(
            score_base_notes,
            quantize_step=score_quantize_seconds,
            min_duration=score_min_note_seconds,
            min_velocity=score_min_velocity,
            max_chord_notes=score_max_chord_notes,
        )
    else:
        polished = polish_score_notes(
            score_base_notes,
            pedals=pedals,
            config=ScorePolishConfig(
                key_signature=key_signature,
                time_signature=time_signature,
                tempo_bpm=float(tempo_bpm) if tempo_bpm is not None else None,
                beat_divisions=score_beat_divisions,
                chord_tolerance_seconds=score_chord_tolerance_seconds,
                min_note_beats=score_min_note_beats,
                max_note_beats=score_max_note_beats,
                min_velocity=score_min_velocity,
                max_chord_notes=score_max_chord_notes,
                max_notes_per_beat=score_max_notes_per_beat,
                trim_score_overlaps=not args.disable_score_overlap_trim,
                max_overlap_beats=score_max_overlap_beats,
            ),
        )
        score_notes = polished.notes
        score_polish_debug = polished.to_json()
    out_dir = ensure_dir(args.out)
    stem = Path(args.audio).stem
    midi_path = write_midi(out_dir / f"{stem}.mid", performance_notes, pedals)
    score_midi_path = write_midi(out_dir / f"{stem}_score.mid", score_notes, [])
    musicxml_path = None
    musicxml_error = None
    try:
        from minimt3.symbolic.score_render import write_musicxml

        musicxml_path = write_musicxml(
            out_dir / f"{stem}.musicxml",
            score_notes,
            title=stem,
            seconds_per_quarter=polished.seconds_per_quarter if polished else 0.5,
            key_signature=polished.key_signature if polished else key_signature,
            time_signature=polished.time_signature if polished else time_signature,
            tempo_bpm=polished.tempo_bpm if polished else tempo_bpm,
            right_notes=polished.right_notes if polished else None,
            left_notes=polished.left_notes if polished else None,
            beat_divisions=polished.beat_divisions if polished else (4,),
            pedals=pedals,
        )
    except Exception as exc:
        musicxml_error = str(exc)
    write_json(
        out_dir / f"{stem}_debug.json",
        {
            "windows": debug,
            "notes": len(performance_notes),
            "score_notes": len(score_notes),
            "raw_notes": raw_note_count,
            "merged_notes": merged_note_count,
            "clean_notes": clean_note_count,
            "pedals": len(pedals),
            "predicted_pedals": predicted_pedal_count,
            "midi": str(midi_path),
            "score_midi": str(score_midi_path),
            "musicxml": str(musicxml_path) if musicxml_path else None,
            "musicxml_error": musicxml_error,
            "score_polish": score_polish_debug,
            "decode": {
                "onset_threshold": onset_threshold,
                "frame_threshold": frame_threshold,
                "offset_threshold": offset_threshold,
                "min_note_seconds": min_note_seconds,
                "max_notes_per_second": max_notes_per_second,
                "max_polyphony": max_polyphony,
                "min_onset_gap_seconds": min_onset_gap_seconds,
                "min_frame_at_onset": min_frame_at_onset,
                "merge_tolerance_seconds": merge_tolerance_seconds,
                "onset_frame_fusion_weight": onset_frame_fusion_weight,
                "chord_onset_threshold": chord_onset_threshold,
                "chord_frame_threshold": chord_frame_threshold,
                "chord_window_frames": chord_window_frames,
                "disable_chord_recovery": args.disable_chord_recovery,
                "chord_score_ratio": chord_score_ratio,
                "onset_peak_prominence": onset_peak_prominence,
                "max_notes_per_start_window": max_notes_per_start_window,
                "start_window_seconds": start_window_seconds,
                "disable_duration_head": args.disable_duration_head,
                "max_duration_seconds": max_duration_seconds,
                "duration_extension_weight": duration_extension_weight,
                "pedal_threshold": pedal_threshold,
                "disable_sustain_heuristic": args.disable_sustain_heuristic,
                "performance_min_note_seconds": performance_min_note_seconds,
                "performance_min_velocity": performance_min_velocity,
                "sustain_max_extension": sustain_max_extension,
                "score_quantize_seconds": score_quantize_seconds,
                "score_min_note_seconds": score_min_note_seconds,
                "score_min_velocity": score_min_velocity,
                "score_max_chord_notes": score_max_chord_notes,
                "disable_score_polish": args.disable_score_polish,
                "key_signature": key_signature,
                "time_signature": time_signature,
                "tempo_bpm": tempo_bpm,
                "score_beat_divisions": list(score_beat_divisions),
                "score_chord_tolerance_seconds": score_chord_tolerance_seconds,
                "score_max_note_beats": score_max_note_beats,
                "score_min_note_beats": score_min_note_beats,
                "score_max_notes_per_beat": score_max_notes_per_beat,
                "score_merge_extension_seconds": score_merge_extension_seconds,
                "disable_score_overlap_trim": args.disable_score_overlap_trim,
                "score_max_overlap_beats": score_max_overlap_beats,
            },
        },
    )
    print(
        f"notes={len(performance_notes)} score_notes={len(score_notes)} "
        f"pedals={len(pedals)} midi={midi_path} musicxml={musicxml_path}"
    )


def merge_overlapping_window_notes(notes, tolerance: float = 0.06):
    """Remove duplicate notes created by overlapping inference windows."""
    if not notes:
        return []
    notes = sorted(notes, key=lambda n: (n.pitch, n.start, -n.velocity, -(n.end - n.start)))
    merged = []
    for note in notes:
        if not merged:
            merged.append(note)
            continue
        prev = merged[-1]
        if note.pitch == prev.pitch and abs(note.start - prev.start) <= tolerance:
            prev_score = (prev.velocity, prev.end - prev.start)
            note_score = (note.velocity, note.end - note.start)
            if note_score > prev_score:
                merged[-1] = note
            else:
                prev.end = max(prev.end, note.end)
            continue
        merged.append(note)
    merged.sort(key=lambda n: (n.start, n.pitch, n.end))
    return merged


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    if not value:
        return (2, 3, 4)
    cleaned = str(value).strip().strip("[]()")
    out = []
    for part in cleaned.split(","):
        part = part.strip()
        if part:
            out.append(max(1, int(part)))
    return tuple(out or [2, 3, 4])


if __name__ == "__main__":
    main()
