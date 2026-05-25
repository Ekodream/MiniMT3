#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize final MiniMT3 AMT and score demo results.")
    parser.add_argument("--eval_dir", default="outputs/eval_final_mainline")
    parser.add_argument("--demo_root", default="outputs/demo_compare")
    parser.add_argument("--out_json", default="outputs/eval_final_mainline/final_metrics.json")
    parser.add_argument("--out_csv", default="outputs/eval_final_mainline/final_metrics.csv")
    parser.add_argument("--demo_json", default="outputs/demo_compare/final_demo_report.json")
    args = parser.parse_args()

    eval_records = [_eval_record(path) for path in sorted(Path(args.eval_dir).glob("*.json"))]
    demo_records = [_demo_record(path) for path in sorted(Path(args.demo_root).glob("final_*/*_debug.json"))]
    analysis_winner = _pick_analysis(eval_records)
    score_winner = _pick_score(eval_records)
    payload = {
        "analysis_midi_best": analysis_winner,
        "score_demo_best": score_winner,
        "eval_records": eval_records,
        "demo_summary": _aggregate_demo(demo_records),
        "demo_records": demo_records,
    }
    _write_json(Path(args.out_json), payload)
    _write_csv(Path(args.out_csv), eval_records, analysis_winner, score_winner)
    _write_json(
        Path(args.demo_json),
        {"summary": payload["demo_summary"], "items": demo_records},
    )
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.demo_json}")
    if analysis_winner:
        print(
            "analysis_midi_best "
            f"name={analysis_winner['name']} note_f1={analysis_winner['best_note_f1']:.4f} "
            f"offset_f1={analysis_winner['best_note_offset_f1']:.4f}"
        )
    if score_winner:
        print(
            "score_demo_best "
            f"name={score_winner['name']} note_f1={score_winner['note_f1']:.4f} "
            f"pred_ref={score_winner['pred_ref_ratio']:.3f}"
        )


def _eval_record(path: Path) -> dict[str, Any]:
    data = _read_json(path)
    summary = data.get("summary") or {}
    balanced = data.get("balanced_summary") or summary
    best_note = data.get("best_note_f1_summary") or summary
    chord = data.get("chord_metrics") or {}
    buckets = data.get("duration_buckets") or {}
    bucket_mid = buckets.get("0.5-2s") or {}
    bucket_long = buckets.get(">2s") or {}
    return {
        "name": path.stem,
        "path": str(path),
        "ckpt": summary.get("ckpt"),
        "decode_preset": summary.get("decode_preset"),
        "param_count": summary.get("param_count") or data.get("param_count"),
        "train_manifest_size": summary.get("train_manifest_size") or data.get("train_manifest_size"),
        "note_f1": _f(balanced.get("note_f1")),
        "note_precision": _f(balanced.get("note_precision")),
        "note_recall": _f(balanced.get("note_recall")),
        "offset_f1": _f(balanced.get("offset_f1")),
        "pred_ref_ratio": _f(balanced.get("pred_ref_ratio")),
        "best_note_f1": _f(best_note.get("note_f1")),
        "best_note_precision": _f(best_note.get("note_precision")),
        "best_note_recall": _f(best_note.get("note_recall")),
        "best_note_offset_f1": _f(best_note.get("offset_f1")),
        "best_note_pred_ref_ratio": _f(best_note.get("pred_ref_ratio")),
        "chord_note_recall": _f(chord.get("chord_note_recall")),
        "complete_chord_rate": _f(chord.get("complete_chord_rate")),
        "chord_false_positive_rate": _f(chord.get("chord_false_positive_rate")),
        "duration_ratio_0_5_2s": _f(bucket_mid.get("duration_ratio")),
        "duration_ratio_gt_2s": _f(bucket_long.get("duration_ratio")),
        "truncated_rate_gt_2s": _f(bucket_long.get("truncated_rate")),
    }


def _demo_record(path: Path) -> dict[str, Any]:
    data = _read_json(path)
    notation = data.get("score_notation") or {}
    polish = data.get("score_polish") or {}
    metrics = polish.get("metrics") or {}
    musicxml = data.get("musicxml")
    musicxml_ok = False
    if musicxml:
        try:
            from minimt3.symbolic.score_render import validate_musicxml

            validate_musicxml(musicxml)
            musicxml_ok = True
        except Exception:
            musicxml_ok = False
    return {
        "name": path.parent.name,
        "debug_json": str(path),
        "musicxml": musicxml,
        "musicxml_ok": musicxml_ok,
        "notes": int(data.get("notes") or 0),
        "score_notes": int(data.get("score_notes") or 0),
        "raw_notes": int(data.get("raw_notes") or 0),
        "clean_notes": int(data.get("clean_notes") or 0),
        "score_notes_per_performance_note": _f(notation.get("score_notes_per_performance_note")),
        "visible_rest_count": _f(notation.get("visible_rest_count")),
        "rest_density": _f(notation.get("rest_density")),
        "chord_verticality": _f(notation.get("chord_verticality")),
        "voice_collision_count": _f(notation.get("voice_collision_count")),
        "long_note_tie_rate": _f(notation.get("long_note_tie_rate")),
        "quantization_error_seconds": _f(metrics.get("quantization_error_seconds")),
        "density_pruned_rate": _f(metrics.get("density_pruned_rate")),
        "overlap_trim_rate": _f(metrics.get("overlap_trim_rate")),
    }


def _pick_analysis(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    candidates = [
        row
        for row in records
        if 0.78 <= row.get("best_note_pred_ref_ratio", 0.0) <= 1.20
    ] or records
    return max(candidates, key=lambda row: (row.get("best_note_f1", 0.0), row.get("best_note_offset_f1", 0.0)))


def _pick_score(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    candidates = [
        row
        for row in records
        if 0.90 <= row.get("pred_ref_ratio", 0.0) <= 1.15
    ] or records
    return max(candidates, key=lambda row: (row.get("note_f1", 0.0), row.get("offset_f1", 0.0)))


def _aggregate_demo(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {"items": 0}
    keys = (
        "musicxml_ok",
        "notes",
        "score_notes",
        "score_notes_per_performance_note",
        "visible_rest_count",
        "rest_density",
        "chord_verticality",
        "voice_collision_count",
        "long_note_tie_rate",
    )
    out: dict[str, float] = {"items": float(len(records))}
    for key in keys:
        values = [float(row.get(key, 0.0) or 0.0) for row in records]
        out[f"avg_{key}"] = sum(values) / max(1, len(values))
    return out


def _write_csv(
    path: Path,
    records: list[dict[str, Any]],
    analysis_winner: dict[str, Any] | None,
    score_winner: dict[str, Any] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name",
        "decode_preset",
        "note_f1",
        "note_precision",
        "note_recall",
        "offset_f1",
        "pred_ref_ratio",
        "best_note_f1",
        "best_note_precision",
        "best_note_recall",
        "best_note_offset_f1",
        "best_note_pred_ref_ratio",
        "chord_note_recall",
        "complete_chord_rate",
        "duration_ratio_0_5_2s",
        "duration_ratio_gt_2s",
        "truncated_rate_gt_2s",
        "analysis_winner",
        "score_winner",
        "ckpt",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in records:
            item = {field: row.get(field) for field in fields}
            item["analysis_winner"] = bool(analysis_winner and row["name"] == analysis_winner["name"])
            item["score_winner"] = bool(score_winner and row["name"] == score_winner["name"])
            writer.writerow(item)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _f(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
