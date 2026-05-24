#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from minimt3.amt.decode import decode_dense_notes, decode_dense_pedals
from minimt3.amt.hybrid import HybridRescueConfig, hybrid_rescue_notes
from minimt3.amt.model import DenseAMT, DenseAMTConfig
from minimt3.amt.presets import apply_decode_preset, apply_score_preset
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
    parser.add_argument("--decode_preset", default="practice_score")
    parser.add_argument("--score_preset", default="score_auto_safe")
    parser.add_argument("--assistant_ckpt")
    parser.add_argument("--assistant_decode_preset", default="v15_rescue")
    parser.add_argument("--hybrid_rescue", action="store_true")
    parser.add_argument("--hybrid_mode", default="chord_long")
    parser.add_argument("--hybrid_chord_window_seconds", type=float)
    parser.add_argument("--hybrid_long_window_seconds", type=float)
    parser.add_argument("--hybrid_duplicate_window_seconds", type=float)
    parser.add_argument("--hybrid_duplicate_overlap_ratio", type=float)
    parser.add_argument("--hybrid_min_velocity", type=int)
    parser.add_argument("--hybrid_min_duration_seconds", type=float)
    parser.add_argument("--hybrid_long_min_duration_seconds", type=float)
    parser.add_argument("--hybrid_bass_pitch_max", type=int)
    parser.add_argument("--hybrid_max_added_ratio", type=float)
    parser.add_argument("--hybrid_max_added_per_second", type=float)
    parser.add_argument("--hybrid_max_added_per_base_onset", type=int)
    parser.add_argument("--hybrid_max_total_notes_per_second", type=float)
    parser.add_argument("--window_seconds", type=float)
    parser.add_argument("--overlap_seconds", type=float)
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
    parser.add_argument("--consume_note_energy", action="store_true")
    parser.add_argument("--disable_consume_note_energy", action="store_true")
    parser.add_argument("--energy_neighbor_pitches", type=int)
    parser.add_argument("--energy_overlap_ratio", type=float)
    parser.add_argument("--infer_onsets_from_frame_diff", action="store_true")
    parser.add_argument("--disable_infer_onsets_from_frame_diff", action="store_true")
    parser.add_argument("--frame_diff_n", type=int)
    parser.add_argument("--frame_diff_scale", type=float)
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
    parser.add_argument("--auto_time_signature", action="store_true")
    parser.add_argument("--tempo_bpm", type=float)
    parser.add_argument("--score_key_signature")
    parser.add_argument("--score_time_signature")
    parser.add_argument("--score_tempo_bpm", type=float)
    parser.add_argument("--score_voice_mode", choices=["single", "dual_staff_2voice"])
    parser.add_argument("--score_split_ties", action="store_true")
    parser.add_argument("--disable_score_split_ties", action="store_true")
    parser.add_argument("--score_hide_filler_rests", action="store_true")
    parser.add_argument("--disable_score_hide_filler_rests", action="store_true")
    parser.add_argument("--score_beat_divisions")
    parser.add_argument("--score_allow_tuplets", action="store_true")
    parser.add_argument("--score_chord_tolerance_seconds", type=float)
    parser.add_argument("--score_max_note_beats", type=float)
    parser.add_argument("--score_min_note_beats", type=float)
    parser.add_argument("--score_max_notes_per_beat", type=int)
    parser.add_argument("--score_merge_extension_seconds", type=float)
    parser.add_argument("--disable_score_overlap_trim", action="store_true")
    parser.add_argument("--score_max_overlap_beats", type=float)
    parser.add_argument("--disable_score_key_filter", action="store_true")
    parser.add_argument("--disable_score_isolation_filter", action="store_true")
    parser.add_argument("--disable_score_fill_rests", action="store_true")
    parser.add_argument("--score_max_short_rest_beats", type=float)
    parser.add_argument("--disable_score_start_align", action="store_true")
    parser.add_argument("--score_leading_rest_threshold_beats", type=float)
    parser.add_argument("--score_start_offset_beats", type=float)
    parser.add_argument("--score_start_offset_seconds", type=float)
    parser.add_argument("--score_chord_snap_seconds", type=float)
    parser.add_argument("--score_chord_snap_max_spread_beats", type=float)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["config"]
    target_cfg = cfg.get("targets", {})
    decode_cfg = apply_decode_preset(cfg.get("decode", {}), args.decode_preset)
    score_cfg = apply_score_preset(decode_cfg, args.score_preset)
    inference_cfg = cfg.get("inference", {})
    window_seconds = float(
        args.window_seconds
        if args.window_seconds is not None
        else inference_cfg.get("window_seconds", decode_cfg.get("window_seconds", _default_window_seconds(cfg)))
    )
    overlap_seconds = float(
        args.overlap_seconds
        if args.overlap_seconds is not None
        else inference_cfg.get("overlap_seconds", decode_cfg.get("overlap_seconds", _default_overlap_seconds(window_seconds)))
    )
    onset_threshold = float(
        args.onset_threshold if args.onset_threshold is not None else _default_onset_threshold(cfg, decode_cfg)
    )
    frame_threshold = float(
        args.frame_threshold if args.frame_threshold is not None else _default_frame_threshold(cfg, decode_cfg)
    )
    offset_threshold = float(
        args.offset_threshold if args.offset_threshold is not None else _default_offset_threshold(cfg, decode_cfg)
    )
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
    disable_chord_recovery = bool(args.disable_chord_recovery or decode_cfg.get("disable_chord_recovery", False))
    if disable_chord_recovery:
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
    consume_note_energy = bool(
        (args.consume_note_energy or _default_consume_note_energy(cfg, decode_cfg))
        and not args.disable_consume_note_energy
    )
    energy_neighbor_pitches = int(
        args.energy_neighbor_pitches
        if args.energy_neighbor_pitches is not None
        else decode_cfg.get("energy_neighbor_pitches", 1)
    )
    energy_overlap_ratio = float(
        args.energy_overlap_ratio
        if args.energy_overlap_ratio is not None
        else decode_cfg.get("energy_overlap_ratio", 0.5)
    )
    infer_onsets_from_frame_diff = bool(
        (args.infer_onsets_from_frame_diff or decode_cfg.get("infer_onsets_from_frame_diff", False))
        and not args.disable_infer_onsets_from_frame_diff
    )
    frame_diff_n = int(args.frame_diff_n if args.frame_diff_n is not None else decode_cfg.get("frame_diff_n", 2))
    frame_diff_scale = float(
        args.frame_diff_scale if args.frame_diff_scale is not None else decode_cfg.get("frame_diff_scale", 1.0)
    )
    decode_center_only = bool(decode_cfg.get("decode_center_only", False))
    reliable_margin_seconds = float(
        decode_cfg.get("reliable_margin_seconds", target_cfg.get("supervision_margin_seconds", 0.0) or 0.0)
    )
    pedal_threshold = float(args.pedal_threshold or decode_cfg.get("pedal_threshold", 0.50))
    performance_min_note_seconds = float(
        args.performance_min_note_seconds or decode_cfg.get("performance_min_note_seconds", 0.06)
    )
    performance_min_velocity = int(args.performance_min_velocity or decode_cfg.get("performance_min_velocity", 4))
    sustain_max_extension = float(args.sustain_max_extension or decode_cfg.get("sustain_max_extension", 1.5))
    score_quantize_seconds = float(args.score_quantize_seconds or score_cfg.get("score_quantize_seconds", 0.125))
    score_min_note_seconds = float(args.score_min_note_seconds or score_cfg.get("score_min_note_seconds", 0.125))
    score_min_velocity = int(args.score_min_velocity or score_cfg.get("score_min_velocity", 6))
    score_max_chord_notes = int(args.score_max_chord_notes or score_cfg.get("score_max_chord_notes", 10))
    key_signature_source = "auto"
    if args.score_key_signature or args.key_signature:
        key_signature_source = "cli"
    elif score_cfg.get("key_signature"):
        key_signature_source = "preset"
    key_signature = args.score_key_signature or args.key_signature or score_cfg.get("key_signature")
    time_signature_source = "preset"
    time_signature = args.score_time_signature or args.time_signature or score_cfg.get("time_signature", "4/4")
    if args.score_time_signature or args.time_signature:
        time_signature_source = "cli"
    if args.auto_time_signature:
        time_signature = "auto"
        time_signature_source = "auto"
    tempo_bpm = (
        args.score_tempo_bpm
        if args.score_tempo_bpm is not None
        else (args.tempo_bpm if args.tempo_bpm is not None else score_cfg.get("tempo_bpm"))
    )
    score_voice_mode = str(args.score_voice_mode or score_cfg.get("score_voice_mode", "dual_staff_2voice"))
    score_split_ties = bool((args.score_split_ties or score_cfg.get("score_split_ties", True)) and not args.disable_score_split_ties)
    score_hide_filler_rests = bool(
        (args.score_hide_filler_rests or score_cfg.get("score_hide_filler_rests", True))
        and not args.disable_score_hide_filler_rests
    )
    score_beat_divisions = _parse_int_tuple(
        args.score_beat_divisions or str(score_cfg.get("score_beat_divisions", "2,4"))
    )
    score_chord_tolerance_seconds = float(
        args.score_chord_tolerance_seconds or score_cfg.get("score_chord_tolerance_seconds", 0.055)
    )
    score_max_note_beats = float(args.score_max_note_beats or score_cfg.get("score_max_note_beats", 4.0))
    score_min_note_beats = float(args.score_min_note_beats or score_cfg.get("score_min_note_beats", 0.25))
    score_max_notes_per_beat = int(args.score_max_notes_per_beat or score_cfg.get("score_max_notes_per_beat", 8))
    score_merge_extension_seconds = float(
        args.score_merge_extension_seconds or score_cfg.get("score_merge_extension_seconds", 0.25)
    )
    score_max_overlap_beats = float(args.score_max_overlap_beats or score_cfg.get("score_max_overlap_beats", 0.0))
    score_allow_tuplets = bool(args.score_allow_tuplets or score_cfg.get("score_allow_tuplets", False))
    score_max_short_rest_beats = float(
        args.score_max_short_rest_beats
        if args.score_max_short_rest_beats is not None
        else score_cfg.get("score_max_short_rest_beats", 0.5)
    )
    score_leading_rest_threshold_beats = float(
        args.score_leading_rest_threshold_beats
        if args.score_leading_rest_threshold_beats is not None
        else score_cfg.get("score_leading_rest_threshold_beats", 0.5)
    )
    score_chord_snap_seconds = float(
        args.score_chord_snap_seconds
        if args.score_chord_snap_seconds is not None
        else score_cfg.get("score_chord_snap_seconds", 0.075)
    )
    score_chord_snap_max_spread_beats = float(
        args.score_chord_snap_max_spread_beats
        if args.score_chord_snap_max_spread_beats is not None
        else score_cfg.get("score_chord_snap_max_spread_beats", 0.25)
    )
    audio_cfg = LogMelConfig(**cfg.get("audio", {}))
    model = DenseAMT(DenseAMTConfig(**cfg.get("model", {}))).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    assistant_model = None
    assistant_decode_cfg = None
    assistant_target_cfg = None
    assistant_extractor = None
    assistant_decode_center_only = False
    assistant_reliable_margin_seconds = 0.0
    if args.assistant_ckpt:
        assistant_ckpt = torch.load(args.assistant_ckpt, map_location=device)
        assistant_cfg = assistant_ckpt["config"]
        assistant_audio_cfg = LogMelConfig(**assistant_cfg.get("audio", {}))
        if int(assistant_audio_cfg.sample_rate) != int(audio_cfg.sample_rate):
            raise ValueError(
                "assistant_ckpt sample_rate differs from primary ckpt; run hybrid only with matching audio sample rates"
            )
        assistant_decode_cfg = apply_decode_preset(assistant_cfg.get("decode", {}), args.assistant_decode_preset)
        assistant_target_cfg = assistant_cfg.get("targets", {})
        assistant_model = DenseAMT(DenseAMTConfig(**assistant_cfg.get("model", {}))).to(device)
        assistant_model.load_state_dict(assistant_ckpt["model"], strict=False)
        assistant_model.eval()
        assistant_extractor = LogMelExtractor(assistant_audio_cfg).to(device)
        assistant_decode_center_only = bool(assistant_decode_cfg.get("decode_center_only", False))
        assistant_reliable_margin_seconds = float(
            assistant_decode_cfg.get(
                "reliable_margin_seconds",
                assistant_target_cfg.get("supervision_margin_seconds", 0.0) or 0.0,
            )
        )
    hybrid_cfg = _hybrid_config_from_args(args)
    waveform = load_audio(args.audio, audio_cfg.sample_rate)
    extractor = LogMelExtractor(audio_cfg).to(device)
    sr = audio_cfg.sample_rate
    total_seconds = waveform.shape[-1] / sr
    if overlap_seconds >= window_seconds:
        overlap_seconds = max(0.0, window_seconds * 0.25)
    step = max(0.1, window_seconds - overlap_seconds)
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
    hybrid_totals: dict[str, float] = {}
    with torch.no_grad():
        for start in starts:
            end = min(total_seconds, start + window_seconds)
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
                time_shift_clip_frames=float(target_cfg.get("time_shift_clip_frames", 1.0)),
                consume_note_energy=consume_note_energy,
                energy_neighbor_pitches=energy_neighbor_pitches,
                energy_overlap_ratio=energy_overlap_ratio,
                infer_onsets_from_frame_diff=infer_onsets_from_frame_diff,
                frame_diff_n=frame_diff_n,
                frame_diff_scale=frame_diff_scale,
            )
            before_center_notes = len(window_notes)
            if decode_center_only and reliable_margin_seconds > 0.0 and end - start > reliable_margin_seconds * 2.0:
                keep_lo = 0.0 if start <= 1e-6 else reliable_margin_seconds
                keep_hi = (end - start) if end >= total_seconds - 1e-6 else (end - start - reliable_margin_seconds)
                window_notes = [note for note in window_notes if keep_lo <= note.start < keep_hi]
            assistant_window_notes = []
            assistant_before_center_notes = 0
            hybrid_stats = None
            if assistant_model is not None and assistant_decode_cfg is not None and assistant_target_cfg is not None:
                assistant_features = assistant_extractor(segment.to(device)) if assistant_extractor is not None else features
                assistant_out = assistant_model(assistant_features)
                assistant_window_notes = _decode_notes_from_config(
                    assistant_out,
                    duration=end - start,
                    decode_cfg=assistant_decode_cfg,
                    target_cfg=assistant_target_cfg,
                )
                assistant_before_center_notes = len(assistant_window_notes)
                if (
                    assistant_decode_center_only
                    and assistant_reliable_margin_seconds > 0.0
                    and end - start > assistant_reliable_margin_seconds * 2.0
                ):
                    keep_lo = 0.0 if start <= 1e-6 else assistant_reliable_margin_seconds
                    keep_hi = (
                        (end - start)
                        if end >= total_seconds - 1e-6
                        else (end - start - assistant_reliable_margin_seconds)
                    )
                    assistant_window_notes = [note for note in assistant_window_notes if keep_lo <= note.start < keep_hi]
                if hybrid_cfg.enabled:
                    window_notes, hybrid_stats = hybrid_rescue_notes(
                        window_notes,
                        assistant_window_notes,
                        duration=end - start,
                        config=hybrid_cfg,
                    )
                    _add_stat_totals(hybrid_totals, hybrid_stats)
            for note in window_notes:
                note.start += start
                note.end += start
            notes.extend(window_notes)
            window_pedals = decode_dense_pedals(out, duration=end - start, threshold=pedal_threshold)
            for pedal in window_pedals:
                pedal.start += start
                pedal.end += start
            pedals.extend(window_pedals)
            debug.append(
                {
                    "start": start,
                    "end": end,
                    "notes": len(window_notes),
                    "notes_before_center_crop": before_center_notes,
                    "assistant_notes": len(assistant_window_notes),
                    "assistant_notes_before_center_crop": assistant_before_center_notes,
                    "hybrid": hybrid_stats,
                    "pedals": len(window_pedals),
                }
            )
    _finalize_stat_totals(hybrid_totals)
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
                allow_tuplets=score_allow_tuplets,
                chord_tolerance_seconds=score_chord_tolerance_seconds,
                min_note_beats=score_min_note_beats,
                max_note_beats=score_max_note_beats,
                min_velocity=score_min_velocity,
                max_chord_notes=score_max_chord_notes,
                max_notes_per_beat=score_max_notes_per_beat,
                trim_score_overlaps=not args.disable_score_overlap_trim,
                max_overlap_beats=score_max_overlap_beats,
                filter_key_outliers=not args.disable_score_key_filter,
                filter_isolated_notes=not args.disable_score_isolation_filter,
                fill_short_rests=not args.disable_score_fill_rests,
                max_short_rest_beats=score_max_short_rest_beats,
                align_score_start=not args.disable_score_start_align,
                leading_rest_threshold_beats=score_leading_rest_threshold_beats,
                start_offset_beats=args.score_start_offset_beats,
                start_offset_seconds=args.score_start_offset_seconds,
                chord_snap_seconds=score_chord_snap_seconds,
                chord_snap_max_spread_beats=score_chord_snap_max_spread_beats,
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
    score_notation_debug = None
    try:
        from minimt3.symbolic.score_render import score_notation_metrics, write_musicxml

        score_notation_debug = score_notation_metrics(
            score_notes,
            seconds_per_quarter=polished.seconds_per_quarter if polished else 0.5,
            key_signature=polished.key_signature if polished else key_signature,
            time_signature=polished.time_signature if polished else time_signature,
            right_notes=polished.right_notes if polished else None,
            left_notes=polished.left_notes if polished else None,
            beat_divisions=polished.beat_divisions if polished else score_beat_divisions,
            voice_mode=score_voice_mode,
            split_ties=score_split_ties,
            hide_filler_rests=score_hide_filler_rests,
            performance_note_count=len(performance_notes),
            key_signature_source=key_signature_source,
            time_signature_source=time_signature_source,
        )

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
            beat_divisions=polished.beat_divisions if polished else score_beat_divisions,
            pedals=pedals,
            voice_mode=score_voice_mode,
            split_ties=score_split_ties,
            hide_filler_rests=score_hide_filler_rests,
        )
    except Exception as exc:
        musicxml_error = str(exc)
    write_json(
        out_dir / f"{stem}_debug.json",
        {
            "windows": debug,
            "window_seconds": window_seconds,
            "overlap_seconds": overlap_seconds,
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
            "score_notation": score_notation_debug,
            "hybrid_rescue": hybrid_cfg.to_json() if hybrid_cfg.enabled else {},
            "hybrid_stats": hybrid_totals,
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
                "disable_chord_recovery": disable_chord_recovery,
                "decode_preset": args.decode_preset,
                "score_preset": args.score_preset,
                "score_profile": score_cfg.get("score_profile", decode_cfg.get("score_profile")),
                "chord_score_ratio": chord_score_ratio,
                "onset_peak_prominence": onset_peak_prominence,
                "max_notes_per_start_window": max_notes_per_start_window,
                "start_window_seconds": start_window_seconds,
                "disable_duration_head": args.disable_duration_head,
                "max_duration_seconds": max_duration_seconds,
                "duration_extension_weight": duration_extension_weight,
                "consume_note_energy": consume_note_energy,
                "energy_neighbor_pitches": energy_neighbor_pitches,
                "energy_overlap_ratio": energy_overlap_ratio,
                "infer_onsets_from_frame_diff": infer_onsets_from_frame_diff,
                "frame_diff_n": frame_diff_n,
                "frame_diff_scale": frame_diff_scale,
                "decode_center_only": decode_center_only,
                "reliable_margin_seconds": reliable_margin_seconds,
                "pedal_threshold": pedal_threshold,
                "disable_sustain_heuristic": args.disable_sustain_heuristic,
                "performance_min_note_seconds": performance_min_note_seconds,
                "performance_min_velocity": performance_min_velocity,
                "sustain_max_extension": sustain_max_extension,
                "score_quantize_seconds": score_quantize_seconds,
                "score_min_note_seconds": score_min_note_seconds,
                "score_min_velocity": score_min_velocity,
                "score_max_chord_notes": score_max_chord_notes,
                "score_voice_mode": score_voice_mode,
                "score_split_ties": score_split_ties,
                "score_hide_filler_rests": score_hide_filler_rests,
                "disable_score_polish": args.disable_score_polish,
                "key_signature": key_signature,
                "key_signature_source": key_signature_source,
                "time_signature": time_signature,
                "time_signature_source": time_signature_source,
                "tempo_bpm": tempo_bpm,
                "score_beat_divisions": list(score_beat_divisions),
                "score_allow_tuplets": score_allow_tuplets,
                "score_chord_tolerance_seconds": score_chord_tolerance_seconds,
                "score_max_note_beats": score_max_note_beats,
                "score_min_note_beats": score_min_note_beats,
                "score_max_notes_per_beat": score_max_notes_per_beat,
                "score_merge_extension_seconds": score_merge_extension_seconds,
                "disable_score_overlap_trim": args.disable_score_overlap_trim,
                "score_max_overlap_beats": score_max_overlap_beats,
                "disable_score_key_filter": args.disable_score_key_filter,
                "disable_score_isolation_filter": args.disable_score_isolation_filter,
                "disable_score_fill_rests": args.disable_score_fill_rests,
                "score_max_short_rest_beats": score_max_short_rest_beats,
                "disable_score_start_align": args.disable_score_start_align,
                "score_leading_rest_threshold_beats": score_leading_rest_threshold_beats,
                "score_start_offset_beats": args.score_start_offset_beats,
                "score_start_offset_seconds": args.score_start_offset_seconds,
                "score_chord_snap_seconds": score_chord_snap_seconds,
                "score_chord_snap_max_spread_beats": score_chord_snap_max_spread_beats,
                "assistant_ckpt": args.assistant_ckpt,
                "assistant_decode_preset": args.assistant_decode_preset if args.assistant_ckpt else None,
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


def _decode_notes_from_config(
    out: dict[str, torch.Tensor],
    duration: float,
    decode_cfg: dict,
    target_cfg: dict,
) -> list:
    disable_chord_recovery = bool(decode_cfg.get("disable_chord_recovery", False))
    chord_onset_threshold = None if disable_chord_recovery else decode_cfg.get("chord_onset_threshold")
    return decode_dense_notes(
        out,
        duration=duration,
        onset_threshold=float(decode_cfg.get("onset_threshold", 0.55)),
        frame_threshold=float(decode_cfg.get("frame_threshold", 0.25)),
        offset_threshold=float(decode_cfg.get("offset_threshold", 0.25)),
        min_note_seconds=float(decode_cfg.get("min_note_seconds", 0.04)),
        max_notes_per_second=float(decode_cfg.get("max_notes_per_second", 24.0)),
        max_polyphony=int(decode_cfg.get("max_polyphony", 12)),
        min_onset_gap_seconds=float(decode_cfg.get("min_onset_gap_seconds", 0.06)),
        min_frame_at_onset=float(decode_cfg.get("min_frame_at_onset", 0.0)),
        onset_frame_fusion_weight=float(decode_cfg.get("onset_frame_fusion_weight", 0.0)),
        chord_onset_threshold=chord_onset_threshold,
        chord_frame_threshold=float(decode_cfg.get("chord_frame_threshold", 0.35)),
        chord_window_frames=int(decode_cfg.get("chord_window_frames", 1)),
        chord_score_ratio=float(decode_cfg.get("chord_score_ratio", 0.75)),
        onset_peak_prominence=float(decode_cfg.get("onset_peak_prominence", 0.0)),
        max_notes_per_start_window=decode_cfg.get("max_notes_per_start_window"),
        start_window_seconds=float(decode_cfg.get("start_window_seconds", 0.08)),
        use_duration_head=not bool(decode_cfg.get("disable_duration_head", False)),
        max_duration_seconds=float(decode_cfg.get("max_duration_seconds", 8.0)),
        duration_extension_weight=float(decode_cfg.get("duration_extension_weight", 1.0)),
        time_shift_clip_frames=float(target_cfg.get("time_shift_clip_frames", 1.0)),
        consume_note_energy=bool(decode_cfg.get("consume_note_energy", False)),
        energy_neighbor_pitches=int(decode_cfg.get("energy_neighbor_pitches", 1)),
        energy_overlap_ratio=float(decode_cfg.get("energy_overlap_ratio", 0.5)),
        infer_onsets_from_frame_diff=bool(decode_cfg.get("infer_onsets_from_frame_diff", False)),
        frame_diff_n=int(decode_cfg.get("frame_diff_n", 2)),
        frame_diff_scale=float(decode_cfg.get("frame_diff_scale", 1.0)),
    )


def _hybrid_config_from_args(args: argparse.Namespace) -> HybridRescueConfig:
    defaults = HybridRescueConfig()
    return HybridRescueConfig(
        enabled=bool(args.hybrid_rescue and args.assistant_ckpt),
        mode=str(args.hybrid_mode or defaults.mode),
        chord_window_seconds=_arg_or_default(args.hybrid_chord_window_seconds, defaults.chord_window_seconds),
        long_window_seconds=_arg_or_default(args.hybrid_long_window_seconds, defaults.long_window_seconds),
        duplicate_window_seconds=_arg_or_default(
            args.hybrid_duplicate_window_seconds,
            defaults.duplicate_window_seconds,
        ),
        duplicate_overlap_ratio=_arg_or_default(
            args.hybrid_duplicate_overlap_ratio,
            defaults.duplicate_overlap_ratio,
        ),
        min_velocity=int(_arg_or_default(args.hybrid_min_velocity, defaults.min_velocity)),
        min_duration_seconds=_arg_or_default(args.hybrid_min_duration_seconds, defaults.min_duration_seconds),
        long_min_duration_seconds=_arg_or_default(
            args.hybrid_long_min_duration_seconds,
            defaults.long_min_duration_seconds,
        ),
        bass_pitch_max=int(_arg_or_default(args.hybrid_bass_pitch_max, defaults.bass_pitch_max)),
        max_added_ratio=_arg_or_default(args.hybrid_max_added_ratio, defaults.max_added_ratio),
        max_added_per_second=_arg_or_default(args.hybrid_max_added_per_second, defaults.max_added_per_second),
        max_added_per_base_onset=int(
            _arg_or_default(args.hybrid_max_added_per_base_onset, defaults.max_added_per_base_onset)
        ),
        max_total_notes_per_second=_arg_or_default(
            args.hybrid_max_total_notes_per_second,
            defaults.max_total_notes_per_second,
        ),
    )


def _arg_or_default(value, default):
    return default if value is None else value


def _add_stat_totals(totals: dict[str, float], values: dict[str, float] | None) -> None:
    if not values:
        return
    for key, value in values.items():
        if key == "hybrid_added_ratio":
            continue
        if isinstance(value, (int, float)):
            totals[key] = totals.get(key, 0.0) + float(value)


def _finalize_stat_totals(totals: dict[str, float]) -> None:
    if not totals:
        return
    added = totals.get("hybrid_added_notes", 0.0)
    base = totals.get("hybrid_base_notes", 0.0)
    totals["hybrid_added_ratio"] = added / max(1.0, base)


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


def _default_window_seconds(cfg: dict) -> float:
    manifest = str(cfg.get("train_manifest", ""))
    target_cfg = cfg.get("targets", {})
    if "4s" in manifest or float(target_cfg.get("max_duration_seconds", 8.0)) >= 12.0:
        return 4.0
    return 2.0


def _default_onset_threshold(cfg: dict, decode_cfg: dict) -> float:
    configured = float(decode_cfg.get("onset_threshold", 0.55))
    manifest = str(cfg.get("train_manifest", ""))
    output_dir = str(cfg.get("output_dir", ""))
    if "v12_crnn_bytedance" in output_dir and "v12_2" not in output_dir:
        return 0.42
    if "v8_1" in output_dir or "v9" in output_dir:
        return 0.32 if configured >= 0.40 else configured
    if configured < 0.40 and ("v8" in manifest or "v8" in output_dir):
        return 0.42
    return configured


def _default_frame_threshold(cfg: dict, decode_cfg: dict) -> float:
    return float(decode_cfg.get("frame_threshold", 0.25))


def _default_offset_threshold(cfg: dict, decode_cfg: dict) -> float:
    configured = float(decode_cfg.get("offset_threshold", 0.25))
    output_dir = str(cfg.get("output_dir", ""))
    if "v12_crnn_bytedance" in output_dir and "v12_2" not in output_dir:
        return 0.24
    return configured


def _default_consume_note_energy(cfg: dict, decode_cfg: dict) -> bool:
    output_dir = str(cfg.get("output_dir", ""))
    if "v12_crnn_bytedance" in output_dir and "v12_2" not in output_dir:
        return True
    return bool(decode_cfg.get("consume_note_energy", False))


def _default_overlap_seconds(window_seconds: float) -> float:
    if window_seconds >= 4.0:
        return 0.75
    return 0.25


if __name__ == "__main__":
    main()
