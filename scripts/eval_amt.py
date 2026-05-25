#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from typing import Any

import torch

from minimt3.amt.analysis import (
    add_metric_total,
    chord_metrics,
    detailed_note_metrics,
    duration_bucket_metrics,
    error_records,
    load_teacher_notes,
    manifest_size,
    model_parameter_count,
    new_metric_total,
    parse_duration_buckets,
    score_quality_metrics,
    summarize_metric_total,
)
from minimt3.amt.data import DenseAMTDataset
from minimt3.amt.decode import decode_dense_notes
from minimt3.amt.hybrid import HybridRescueConfig, hybrid_rescue_notes
from minimt3.amt.model import DenseAMT, DenseAMTConfig
from minimt3.amt.presets import apply_decode_preset
from minimt3.amt.targets import DenseTargetConfig
from minimt3.audio.features import LogMelConfig
from minimt3.symbolic.events import NoteEvent, load_midi_events
from minimt3.symbolic.midi_io import write_midi
from minimt3.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold-sweep eval for dense AMT checkpoints.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--items", type=int, default=16)
    parser.add_argument("--cache_dir")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--decode_preset")
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
    parser.add_argument("--onset_thresholds")
    parser.add_argument("--frame_thresholds")
    parser.add_argument("--offset_thresholds")
    parser.add_argument("--max_polyphony", type=int)
    parser.add_argument("--max_notes_per_second", type=float)
    parser.add_argument("--min_onset_gap_seconds", type=float)
    parser.add_argument("--min_frame_at_onset", type=float)
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
    parser.add_argument("--duration_extension_weights")
    parser.add_argument("--consume_note_energy", action="store_true")
    parser.add_argument("--energy_neighbor_pitches", type=int)
    parser.add_argument("--energy_overlap_ratio", type=float)
    parser.add_argument("--infer_onsets_from_frame_diff", action="store_true")
    parser.add_argument("--frame_diff_modes")
    parser.add_argument("--frame_diff_n", type=int)
    parser.add_argument("--frame_diff_scale", type=float)
    parser.add_argument("--frame_diff_scales")
    parser.add_argument("--frame_diff_min_onset", type=float)
    parser.add_argument("--frame_diff_context_threshold", type=float)
    parser.add_argument("--frame_diff_context_window_frames", type=int)
    parser.add_argument("--frame_diff_context_min_pitches", type=int)
    parser.add_argument("--eval_center_only", action="store_true")
    parser.add_argument("--balanced_min_pred_ref", type=float, default=0.90)
    parser.add_argument("--balanced_max_pred_ref", type=float, default=1.15)
    parser.add_argument("--f1_min_pred_ref", type=float, default=0.78)
    parser.add_argument("--f1_max_pred_ref", type=float, default=1.20)
    parser.add_argument("--analysis_json_out")
    parser.add_argument("--json_out")
    parser.add_argument("--error_midi_out")
    parser.add_argument("--score_quality_eval", action="store_true")
    parser.add_argument("--score_quality_items", type=int, default=0)
    parser.add_argument("--teacher_midi_dir")
    parser.add_argument("--duration_buckets", default="0,0.125,0.5,2.0,inf")
    parser.add_argument("--chord_tolerance_seconds", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["config"]
    decode_cfg = apply_decode_preset(cfg.get("decode", {}), args.decode_preset)
    model = DenseAMT(DenseAMTConfig(**cfg.get("model", {}))).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    target_config = DenseTargetConfig(**cfg.get("targets", {}))
    assistant_model = None
    assistant_decode_cfg = None
    assistant_target_config = None
    assistant_param_count = None
    if args.assistant_ckpt:
        assistant_ckpt = torch.load(args.assistant_ckpt, map_location=device)
        assistant_cfg = assistant_ckpt["config"]
        if dict(assistant_cfg.get("audio", {})) != dict(cfg.get("audio", {})):
            raise ValueError("assistant_ckpt audio config must match primary ckpt for cached eval features")
        assistant_decode_cfg = apply_decode_preset(assistant_cfg.get("decode", {}), args.assistant_decode_preset)
        assistant_model = DenseAMT(DenseAMTConfig(**assistant_cfg.get("model", {}))).to(device)
        assistant_model.load_state_dict(assistant_ckpt["model"], strict=False)
        assistant_model.eval()
        assistant_target_config = DenseTargetConfig(**assistant_cfg.get("targets", {}))
        assistant_param_count = model_parameter_count(assistant_model)
    hybrid_cfg = _hybrid_config_from_args(args)
    dataset = DenseAMTDataset(
        args.manifest,
        feature_config=LogMelConfig(**cfg.get("audio", {})),
        split=args.split,
        max_items=args.items,
        cache_dir=args.cache_dir,
        target_config=target_config,
    )
    onset_values = _threshold_values(args.onset_thresholds, decode_cfg, "onset_threshold", [0.45, 0.55, 0.65, 0.75])
    frame_values = _threshold_values(args.frame_thresholds, decode_cfg, "frame_threshold", [0.30, 0.40, 0.50])
    offset_values = _threshold_values(args.offset_thresholds, decode_cfg, "offset_threshold", [0.30, 0.40, 0.50])
    duration_buckets = parse_duration_buckets(args.duration_buckets)
    max_polyphony = int(args.max_polyphony or decode_cfg.get("max_polyphony", 12))
    max_notes_per_second = float(args.max_notes_per_second or decode_cfg.get("max_notes_per_second", 45.0))
    min_note_seconds = float(decode_cfg.get("min_note_seconds", 0.04))
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
    onset_frame_fusion_weight = float(
        args.onset_frame_fusion_weight
        if args.onset_frame_fusion_weight is not None
        else decode_cfg.get("onset_frame_fusion_weight", 0.0)
    )
    disable_chord_recovery = bool(args.disable_chord_recovery or decode_cfg.get("disable_chord_recovery", False))
    chord_onset_threshold = None
    if not disable_chord_recovery:
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
    consume_note_energy = bool(args.consume_note_energy or decode_cfg.get("consume_note_energy", False))
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
        args.infer_onsets_from_frame_diff or decode_cfg.get("infer_onsets_from_frame_diff", False)
    )
    frame_diff_n = int(args.frame_diff_n if args.frame_diff_n is not None else decode_cfg.get("frame_diff_n", 2))
    frame_diff_scale = float(
        args.frame_diff_scale if args.frame_diff_scale is not None else decode_cfg.get("frame_diff_scale", 1.0)
    )
    frame_diff_min_onset = float(
        args.frame_diff_min_onset
        if args.frame_diff_min_onset is not None
        else decode_cfg.get("frame_diff_min_onset", 0.0)
    )
    frame_diff_context_threshold = float(
        args.frame_diff_context_threshold
        if args.frame_diff_context_threshold is not None
        else decode_cfg.get("frame_diff_context_threshold", 0.0)
    )
    frame_diff_context_window_frames = int(
        args.frame_diff_context_window_frames
        if args.frame_diff_context_window_frames is not None
        else decode_cfg.get("frame_diff_context_window_frames", 0)
    )
    frame_diff_context_min_pitches = int(
        args.frame_diff_context_min_pitches
        if args.frame_diff_context_min_pitches is not None
        else decode_cfg.get("frame_diff_context_min_pitches", 0)
    )
    eval_center_only = bool(args.eval_center_only or decode_cfg.get("eval_center_only", False))
    combos = _decode_combos(
        onset_values,
        frame_values,
        offset_values,
        _bool_values(args.frame_diff_modes, decode_cfg, "frame_diff_mode", infer_onsets_from_frame_diff),
        _sweep_values(args.frame_diff_scales, decode_cfg, "frame_diff_scale", frame_diff_scale),
        _sweep_values(args.duration_extension_weights, decode_cfg, "duration_extension_weight", duration_extension_weight),
    )

    totals = {combo: new_metric_total() for combo in combos}
    combo_item_records: dict[tuple[float, float, float], list[dict[str, Any]]] = {combo: [] for combo in combos}
    teacher_total = new_metric_total()
    teacher_items = 0
    item_best_records = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            row = sample["meta"]
            duration = float(row["end_sec"]) - float(row["start_sec"])
            features = sample["features"].unsqueeze(0).to(device)
            out = model(features)
            assistant_notes = []
            if assistant_model is not None and assistant_decode_cfg is not None and assistant_target_config is not None:
                assistant_notes = _decode_notes_from_config(
                    assistant_model(features),
                    duration=duration,
                    decode_cfg=assistant_decode_cfg,
                    target_config=assistant_target_config,
                )
            ref_notes, _ = load_midi_events(row["midi"], start=float(row["start_sec"]), end=float(row["end_sec"]))
            clip_id = str(row.get("clip_id", idx))
            best_item = None
            for combo in combos:
                notes = decode_dense_notes(
                    out,
                    duration=duration,
                    onset_threshold=combo[0],
                    frame_threshold=combo[1],
                    offset_threshold=combo[2],
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
                    duration_extension_weight=combo[5],
                    time_shift_clip_frames=float(target_config.time_shift_clip_frames),
                    consume_note_energy=consume_note_energy,
                    energy_neighbor_pitches=energy_neighbor_pitches,
                    energy_overlap_ratio=energy_overlap_ratio,
                    infer_onsets_from_frame_diff=combo[3],
                    frame_diff_n=frame_diff_n,
                    frame_diff_scale=combo[4],
                    frame_diff_min_onset=frame_diff_min_onset,
                    frame_diff_context_threshold=frame_diff_context_threshold,
                    frame_diff_context_window_frames=frame_diff_context_window_frames,
                    frame_diff_context_min_pitches=frame_diff_context_min_pitches,
                )
                hybrid_stats = {}
                if hybrid_cfg.enabled:
                    notes, hybrid_stats = hybrid_rescue_notes(notes, assistant_notes, duration, hybrid_cfg)
                eval_notes, eval_ref_notes = _maybe_center_crop_notes(
                    notes,
                    ref_notes,
                    duration,
                    float(target_config.supervision_margin_seconds),
                    eval_center_only,
                )
                metric = detailed_note_metrics(eval_notes, eval_ref_notes)
                add_metric_total(totals[combo], metric)
                fps, fns = error_records(eval_notes, eval_ref_notes, metric["matches"], clip_id)
                record = {
                    "index": idx,
                    "clip_id": clip_id,
                    "thresholds": _combo_dict(combo),
                    "duration": duration,
                    "audio": row.get("audio"),
                    "midi": row.get("midi"),
                    "metrics": _metric_without_matches(metric),
                    "duration_buckets": duration_bucket_metrics(
                        eval_notes,
                        eval_ref_notes,
                        metric["matches"],
                        duration_buckets,
                    ),
                    "chord_metrics": chord_metrics(
                        eval_notes,
                        eval_ref_notes,
                        metric["matches"],
                        tolerance=float(args.chord_tolerance_seconds),
                    ),
                    "hybrid": hybrid_stats,
                    "false_positives": fps,
                    "false_negatives": fns,
                }
                combo_item_records[combo].append(record)
                cand = (metric["note_f1"], metric["offset_f1"], -metric["note_fp"], combo, record)
                if best_item is None or cand[:4] > best_item[:4]:
                    best_item = cand
            assert best_item is not None
            item_best_records.append(best_item[4])
            teacher_notes = load_teacher_notes(args.teacher_midi_dir, row)
            if teacher_notes is not None:
                teacher_eval, teacher_ref = _maybe_center_crop_notes(
                    teacher_notes,
                    ref_notes,
                    duration,
                    float(target_config.supervision_margin_seconds),
                    eval_center_only,
                )
                teacher_metric = detailed_note_metrics(teacher_eval, teacher_ref)
                add_metric_total(teacher_total, teacher_metric)
                teacher_items += 1
            best_metric = best_item[4]["metrics"]
            print(
                "amt_item "
                f"index={idx} clip_id={clip_id} "
                f"best_note_f1={best_metric['note_f1']:.4f} "
                f"best_offset_f1={best_metric['offset_f1']:.4f} "
                f"note_p={best_metric['note_precision']:.4f} note_r={best_metric['note_recall']:.4f} "
                f"pred_notes={best_metric['pred_notes']} thresholds={best_item[3]} "
                f"ref_notes={best_metric['ref_notes']}",
                flush=True,
            )

    best_combo = None
    best_score = -1e9
    grid_summaries = []
    for combo, values in sorted(totals.items()):
        summary = summarize_metric_total(values)
        score = selection_score(
            summary["note_f1"],
            summary["offset_f1"],
            summary["pred_ref_ratio"],
            min_pred_ref=float(args.balanced_min_pred_ref),
            max_pred_ref=float(args.balanced_max_pred_ref),
        )
        row = {"combo": combo, "thresholds": _combo_dict(combo), "score": score, **summary}
        grid_summaries.append(row)
        print(
            "amt_summary "
            f"onset_t={combo[0]:.2f} frame_t={combo[1]:.2f} offset_t={combo[2]:.2f} "
            f"frame_diff={int(combo[3])} frame_diff_scale={combo[4]:.2f} duration_ext={combo[5]:.2f} "
            f"note_p={summary['note_precision']:.4f} note_r={summary['note_recall']:.4f} "
            f"note_f1={summary['note_f1']:.4f} offset_f1={summary['offset_f1']:.4f} "
            f"pred_ref_ratio={summary['pred_ref_ratio']:.3f} "
            f"pred_notes={summary['pred_notes']:.0f} ref_notes={summary['ref_notes']:.0f} "
            f"score={score:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_combo = combo
    assert best_combo is not None
    print(f"amt_best thresholds={best_combo} score={best_score:.4f}", flush=True)

    best_summary = next(row for row in grid_summaries if row["combo"] == best_combo)
    best_note_f1_summary = _best_note_f1_summary(
        grid_summaries,
        min_pred_ref=float(args.f1_min_pred_ref),
        max_pred_ref=float(args.f1_max_pred_ref),
    )
    best_records = combo_item_records[best_combo]
    if args.score_quality_eval:
        _attach_score_quality_for_combo(
            model=model,
            dataset=dataset,
            device=device,
            combo=best_combo,
            records=best_records,
            decode_cfg=decode_cfg,
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
            time_shift_clip_frames=float(target_config.time_shift_clip_frames),
            consume_note_energy=consume_note_energy,
            energy_neighbor_pitches=energy_neighbor_pitches,
            energy_overlap_ratio=energy_overlap_ratio,
            infer_onsets_from_frame_diff=infer_onsets_from_frame_diff,
            frame_diff_n=frame_diff_n,
            frame_diff_scale=frame_diff_scale,
            frame_diff_min_onset=frame_diff_min_onset,
            frame_diff_context_threshold=frame_diff_context_threshold,
            frame_diff_context_window_frames=frame_diff_context_window_frames,
            frame_diff_context_min_pitches=frame_diff_context_min_pitches,
            eval_center_only=eval_center_only,
            supervision_margin_seconds=float(target_config.supervision_margin_seconds),
            chord_tolerance_seconds=float(args.chord_tolerance_seconds),
            max_items=max(0, int(args.score_quality_items)),
            assistant_model=assistant_model,
            assistant_decode_cfg=assistant_decode_cfg,
            assistant_target_config=assistant_target_config,
            hybrid_cfg=hybrid_cfg,
        )
    report = {
        "summary": {
            "ckpt": args.ckpt,
            "manifest": args.manifest,
            "split": args.split,
            "items": len(dataset),
            "decode_preset": args.decode_preset,
            "best_thresholds": _combo_dict(best_combo),
            "balanced_score_threshold": _combo_dict(best_combo),
            "best_note_f1_threshold": best_note_f1_summary["thresholds"],
            "best_score": best_score,
            "best_note_f1_score": best_note_f1_summary["note_f1_score"],
            "param_count": model_parameter_count(model),
            "train_manifest_size": manifest_size(cfg.get("train_manifest")),
            "eval_manifest_size": manifest_size(args.manifest),
            "assistant_ckpt": args.assistant_ckpt,
            "assistant_decode_preset": args.assistant_decode_preset if args.assistant_ckpt else None,
            "assistant_param_count": assistant_param_count,
            **best_summary,
        },
        "balanced_score_threshold": _combo_dict(best_combo),
        "balanced_summary": _json_ready_summary(best_summary),
        "best_note_f1_threshold": best_note_f1_summary["thresholds"],
        "best_note_f1_summary": _json_ready_summary(best_note_f1_summary),
        "param_count": model_parameter_count(model),
        "train_manifest_size": manifest_size(cfg.get("train_manifest")),
        "threshold_grid": grid_summaries,
        "precision_recall": {
            key: best_summary[key]
            for key in ("note_precision", "note_recall", "note_f1", "offset_precision", "offset_recall", "offset_f1")
        },
        "duration_buckets": _average_nested(best_records, "duration_buckets"),
        "long_note_metrics": _average_nested(best_records, "duration_buckets").get(">2s", {}),
        "chord_metrics": _average_nested(best_records, "chord_metrics"),
        "velocity_metrics": {
            "velocity_mae": best_summary["velocity_mae"],
            "velocity_bias": best_summary["velocity_bias"],
        },
        "score_quality": _average_nested(best_records, "score_quality") if args.score_quality_eval else {},
        "score_notation": _average_nested(best_records, "score_notation") if args.score_quality_eval else {},
        "hybrid_rescue": hybrid_cfg.to_json() if hybrid_cfg.enabled else {},
        "hybrid": _average_nested(best_records, "hybrid"),
        "teacher_baseline": summarize_metric_total(teacher_total) if teacher_items else {},
        "pitch_calibration": _pitch_calibration(best_records),
        "worst_items": sorted(best_records, key=lambda item: item["metrics"]["note_f1"])[:20],
        "false_positives": _flatten_limited(best_records, "false_positives", 200),
        "false_negatives": _flatten_limited(best_records, "false_negatives", 200),
        "items": best_records,
        "item_best_records": item_best_records,
        "decode": {
            "consume_note_energy": consume_note_energy,
            "infer_onsets_from_frame_diff": infer_onsets_from_frame_diff,
            "frame_diff_min_onset": frame_diff_min_onset,
            "frame_diff_context_threshold": frame_diff_context_threshold,
            "frame_diff_context_window_frames": frame_diff_context_window_frames,
            "frame_diff_context_min_pitches": frame_diff_context_min_pitches,
            "eval_center_only": eval_center_only,
            "disable_chord_recovery": disable_chord_recovery,
            "chord_tolerance_seconds": args.chord_tolerance_seconds,
            "duration_buckets": [label for label, _, _ in duration_buckets],
            "assistant_decode_preset": args.assistant_decode_preset if args.assistant_ckpt else None,
            "score_quality_items": int(args.score_quality_items) if args.score_quality_eval else 0,
        },
    }
    if args.analysis_json_out:
        write_json(args.analysis_json_out, report)
    if args.json_out:
        write_json(args.json_out, report)
    if args.error_midi_out:
        write_midi(args.error_midi_out, _error_notes_for_midi(best_records), [])


def _threshold_values(arg_value: str | None, decode_cfg: dict[str, Any], key: str, defaults: list[float]) -> list[float]:
    if arg_value:
        return _floats(arg_value)
    grid_key = f"{key}s"
    if grid_key in decode_cfg:
        return _floats(decode_cfg[grid_key])
    if key in decode_cfg:
        return [float(decode_cfg[key])]
    return defaults


def _sweep_values(arg_value: str | None, decode_cfg: dict[str, Any], key: str, default: float) -> list[float]:
    if arg_value:
        return _floats(arg_value)
    grid_key = f"{key}s"
    if grid_key in decode_cfg:
        return _floats(decode_cfg[grid_key])
    return [float(default)]


def _bool_values(arg_value: str | None, decode_cfg: dict[str, Any], key: str, default: bool) -> list[bool]:
    if arg_value:
        return [_parse_bool(part) for part in str(arg_value).split(",") if part.strip()]
    grid_key = f"{key}s"
    if grid_key in decode_cfg:
        value = decode_cfg[grid_key]
        if isinstance(value, (list, tuple)):
            return [_parse_bool(item) for item in value]
        return [_parse_bool(part) for part in str(value).split(",") if part.strip()]
    return [bool(default)]


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _decode_combos(
    onset_values: list[float],
    frame_values: list[float],
    offset_values: list[float],
    frame_diff_modes: list[bool],
    frame_diff_scales: list[float],
    duration_extension_weights: list[float],
) -> list[tuple[float, float, float, bool, float, float]]:
    combos = []
    for onset_t in onset_values:
        for frame_t in frame_values:
            for offset_t in offset_values:
                for infer_frame_diff in frame_diff_modes:
                    scales = frame_diff_scales if infer_frame_diff else [frame_diff_scales[0]]
                    for frame_diff_scale in scales:
                        for duration_weight in duration_extension_weights:
                            combos.append(
                                (
                                    float(onset_t),
                                    float(frame_t),
                                    float(offset_t),
                                    bool(infer_frame_diff),
                                    float(frame_diff_scale),
                                    float(duration_weight),
                                )
                            )
    return combos


def _combo_dict(combo: tuple[float, float, float, bool, float, float]) -> dict[str, Any]:
    return {
        "onset_threshold": combo[0],
        "frame_threshold": combo[1],
        "offset_threshold": combo[2],
        "infer_onsets_from_frame_diff": combo[3],
        "frame_diff_scale": combo[4],
        "duration_extension_weight": combo[5],
    }


def _floats(value: str | float | int | list[float] | tuple[float, ...]) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def _decode_notes_from_config(
    out: dict[str, torch.Tensor],
    duration: float,
    decode_cfg: dict[str, Any],
    target_config: DenseTargetConfig,
) -> list[NoteEvent]:
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
        time_shift_clip_frames=float(target_config.time_shift_clip_frames),
        consume_note_energy=bool(decode_cfg.get("consume_note_energy", False)),
        energy_neighbor_pitches=int(decode_cfg.get("energy_neighbor_pitches", 1)),
        energy_overlap_ratio=float(decode_cfg.get("energy_overlap_ratio", 0.5)),
        infer_onsets_from_frame_diff=bool(decode_cfg.get("infer_onsets_from_frame_diff", False)),
        frame_diff_n=int(decode_cfg.get("frame_diff_n", 2)),
        frame_diff_scale=float(decode_cfg.get("frame_diff_scale", 1.0)),
        frame_diff_min_onset=float(decode_cfg.get("frame_diff_min_onset", 0.0)),
        frame_diff_context_threshold=float(decode_cfg.get("frame_diff_context_threshold", 0.0)),
        frame_diff_context_window_frames=int(decode_cfg.get("frame_diff_context_window_frames", 0)),
        frame_diff_context_min_pitches=int(decode_cfg.get("frame_diff_context_min_pitches", 0)),
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


@torch.no_grad()
def _attach_score_quality_for_combo(
    model: DenseAMT,
    dataset: DenseAMTDataset,
    device: torch.device,
    combo: tuple[float, float, float, bool, float, float],
    records: list[dict[str, Any]],
    decode_cfg: dict[str, Any],
    min_note_seconds: float,
    max_notes_per_second: float,
    max_polyphony: int,
    min_onset_gap_seconds: float,
    min_frame_at_onset: float,
    onset_frame_fusion_weight: float,
    chord_onset_threshold: float | None,
    chord_frame_threshold: float,
    chord_window_frames: int,
    chord_score_ratio: float,
    onset_peak_prominence: float,
    max_notes_per_start_window: int | None,
    start_window_seconds: float,
    use_duration_head: bool,
    max_duration_seconds: float,
    duration_extension_weight: float,
    time_shift_clip_frames: float,
    consume_note_energy: bool,
    energy_neighbor_pitches: int,
    energy_overlap_ratio: float,
    infer_onsets_from_frame_diff: bool,
    frame_diff_n: int,
    frame_diff_scale: float,
    frame_diff_min_onset: float,
    frame_diff_context_threshold: float,
    frame_diff_context_window_frames: int,
    frame_diff_context_min_pitches: int,
    eval_center_only: bool,
    supervision_margin_seconds: float,
    chord_tolerance_seconds: float,
    max_items: int = 0,
    assistant_model: DenseAMT | None = None,
    assistant_decode_cfg: dict[str, Any] | None = None,
    assistant_target_config: DenseTargetConfig | None = None,
    hybrid_cfg: HybridRescueConfig | None = None,
) -> None:
    del decode_cfg
    records_by_index = {int(record["index"]): record for record in records}
    processed = 0
    for idx in range(len(dataset)):
        if max_items > 0 and processed >= max_items:
            break
        record = records_by_index.get(idx)
        if record is None:
            continue
        sample = dataset[idx]
        row = sample["meta"]
        duration = float(row["end_sec"]) - float(row["start_sec"])
        features = sample["features"].unsqueeze(0).to(device)
        out = model(features)
        notes = decode_dense_notes(
            out,
            duration=duration,
            onset_threshold=combo[0],
            frame_threshold=combo[1],
            offset_threshold=combo[2],
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
            use_duration_head=use_duration_head,
            max_duration_seconds=max_duration_seconds,
            duration_extension_weight=combo[5],
            time_shift_clip_frames=time_shift_clip_frames,
            consume_note_energy=consume_note_energy,
            energy_neighbor_pitches=energy_neighbor_pitches,
            energy_overlap_ratio=energy_overlap_ratio,
            infer_onsets_from_frame_diff=combo[3],
            frame_diff_n=frame_diff_n,
            frame_diff_scale=combo[4],
            frame_diff_min_onset=frame_diff_min_onset,
            frame_diff_context_threshold=frame_diff_context_threshold,
            frame_diff_context_window_frames=frame_diff_context_window_frames,
            frame_diff_context_min_pitches=frame_diff_context_min_pitches,
        )
        if hybrid_cfg is not None and hybrid_cfg.enabled and assistant_model is not None and assistant_decode_cfg is not None:
            assistant_notes = _decode_notes_from_config(
                assistant_model(features),
                duration=duration,
                decode_cfg=assistant_decode_cfg,
                target_config=assistant_target_config or DenseTargetConfig(),
            )
            notes, hybrid_stats = hybrid_rescue_notes(notes, assistant_notes, duration, hybrid_cfg)
            record["hybrid"] = hybrid_stats
        eval_notes, _ = _maybe_center_crop_notes(
            notes,
            [],
            duration,
            supervision_margin_seconds,
            eval_center_only,
        )
        record["score_quality"] = score_quality_metrics(
            eval_notes,
            chord_tolerance_seconds=chord_tolerance_seconds,
        )
        record["score_notation"] = record["score_quality"].get("score_notation", {})
        processed += 1
    print(f"score_quality_items={processed}", flush=True)


def selection_score(
    note_f1: float,
    offset_f1: float,
    pred_ref_ratio: float,
    min_pred_ref: float = 0.90,
    max_pred_ref: float = 1.15,
) -> float:
    if pred_ref_ratio <= 0:
        return -1e9
    ratio_error = abs(math.log(max(1e-6, pred_ref_ratio)))
    over_generation = max(0.0, pred_ref_ratio - max_pred_ref)
    under_generation = max(0.0, min_pred_ref - pred_ref_ratio)
    return 10.0 * note_f1 + offset_f1 - 1.15 * ratio_error - 0.65 * over_generation - 0.35 * under_generation


def _best_note_f1_summary(
    rows: list[dict[str, Any]],
    min_pred_ref: float,
    max_pred_ref: float,
) -> dict[str, Any]:
    def score(row: dict[str, Any]) -> tuple[float, float, float, float]:
        pred_ref = float(row.get("pred_ref_ratio", 0.0))
        outside = max(0.0, float(min_pred_ref) - pred_ref, pred_ref - float(max_pred_ref))
        return (
            float(row.get("note_f1", 0.0)) - 0.05 * outside,
            float(row.get("offset_f1", 0.0)),
            -outside,
            -abs(math.log(max(1e-6, pred_ref))) if pred_ref > 0 else -999.0,
        )

    best = max(rows, key=score)
    out = dict(best)
    out["note_f1_score"] = score(best)[0]
    return out


def _json_ready_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "combo"}


def _maybe_center_crop_notes(
    pred_notes: list[NoteEvent],
    ref_notes: list[NoteEvent],
    duration: float,
    margin: float,
    enabled: bool,
) -> tuple[list[NoteEvent], list[NoteEvent]]:
    if not enabled or margin <= 0.0 or duration <= margin * 2.0:
        return pred_notes, ref_notes
    lo = margin
    hi = duration - margin
    return (
        [note for note in pred_notes if lo <= note.start < hi],
        [note for note in ref_notes if lo <= note.start < hi],
    )


def _metric_without_matches(metric: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metric.items() if key != "matches"}


def _average_nested(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    sums: dict[str, Any] = {}
    counts: dict[str, Any] = {}
    for record in records:
        value = record.get(key)
        if not isinstance(value, dict):
            continue
        _add_nested(sums, counts, value)
    return _divide_nested(sums, counts)


def _add_nested(sums: dict[str, Any], counts: dict[str, Any], value: dict[str, Any]) -> None:
    for key, item in value.items():
        if isinstance(item, dict):
            sums.setdefault(key, {})
            counts.setdefault(key, {})
            _add_nested(sums[key], counts[key], item)
        elif isinstance(item, (int, float)):
            sums[key] = sums.get(key, 0.0) + float(item)
            counts[key] = counts.get(key, 0) + 1


def _divide_nested(sums: dict[str, Any], counts: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, item in sums.items():
        if isinstance(item, dict):
            out[key] = _divide_nested(item, counts.get(key, {}))
        else:
            out[key] = item / max(1, counts.get(key, 0))
    return out


def _flatten_limited(records: list[dict[str, Any]], key: str, limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record in records:
        out.extend(record.get(key, []))
        if len(out) >= limit:
            return out[:limit]
    return out


def _pitch_calibration(records: list[dict[str, Any]]) -> dict[str, Any]:
    stats: dict[int, dict[str, float]] = {}
    for record in records:
        for item in record.get("false_positives", []):
            pitch = int(item["pitch"])
            stats.setdefault(pitch, {"false_positives": 0.0, "false_negatives": 0.0})
            stats[pitch]["false_positives"] += 1.0
        for item in record.get("false_negatives", []):
            pitch = int(item["pitch"])
            stats.setdefault(pitch, {"false_positives": 0.0, "false_negatives": 0.0})
            stats[pitch]["false_negatives"] += 1.0
    rows = []
    for pitch, row in sorted(stats.items()):
        fp = float(row["false_positives"])
        fn = float(row["false_negatives"])
        denom = max(1.0, fp + fn)
        # Positive bias means raise this pitch threshold; negative means lower it.
        threshold_bias_hint = max(-0.08, min(0.08, 0.08 * (fp - fn) / denom))
        rows.append(
            {
                "pitch": pitch,
                "false_positives": fp,
                "false_negatives": fn,
                "net_fp_minus_fn": fp - fn,
                "threshold_bias_hint": threshold_bias_hint,
            }
        )
    return {
        "by_pitch": rows,
        "top_false_positive_pitches": sorted(rows, key=lambda item: item["false_positives"], reverse=True)[:12],
        "top_false_negative_pitches": sorted(rows, key=lambda item: item["false_negatives"], reverse=True)[:12],
    }


def _error_notes_for_midi(records: list[dict[str, Any]]) -> list[NoteEvent]:
    notes: list[NoteEvent] = []
    cursor = 0.0
    for record in records:
        duration = float(record.get("duration", 0.0))
        for item in record.get("false_positives", []):
            notes.append(
                NoteEvent(
                    int(item["pitch"]),
                    cursor + float(item["start"]),
                    cursor + float(item["end"]),
                    38,
                )
            )
        for item in record.get("false_negatives", []):
            notes.append(
                NoteEvent(
                    int(item["pitch"]),
                    cursor + float(item["start"]),
                    cursor + float(item["end"]),
                    112,
                )
            )
        cursor += max(2.0, duration) + 1.0
    notes.sort(key=lambda note: (note.start, note.pitch, note.end))
    return notes


if __name__ == "__main__":
    main()
