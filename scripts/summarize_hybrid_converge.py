#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize v13+v19 hybrid convergence results.")
    parser.add_argument("--eval_dir", default="outputs/eval_hybrid_converge")
    parser.add_argument("--demo_root", default="outputs/demo_compare")
    parser.add_argument("--out_json", default="outputs/eval_hybrid_converge/hybrid_metrics.json")
    parser.add_argument("--out_csv", default="outputs/eval_hybrid_converge/hybrid_metrics.csv")
    parser.add_argument("--demo_json", default="outputs/demo_compare/hybrid_demo_report.json")
    args = parser.parse_args()

    eval_records = [_eval_record(path) for path in sorted(Path(args.eval_dir).glob("*.json")) if _is_eval_json(path)]
    demo_records = [_demo_record(path) for path in sorted(Path(args.demo_root).glob("hybrid_converge_*/*_debug.json"))]
    payload = {
        "hybrid_f1_best": _pick_hybrid_f1(eval_records),
        "hybrid_score_best": _pick_hybrid_score(eval_records),
        "eval_records": eval_records,
        "demo_summary": _aggregate_demo(demo_records),
        "demo_records": demo_records,
    }
    _write_json(args.out_json, payload)
    _write_csv(args.out_csv, eval_records)
    _write_json(args.demo_json, {"summary": payload["demo_summary"], "items": demo_records})
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.demo_json}")


def _is_eval_json(path: Path) -> bool:
    return path.name not in {"hybrid_metrics.json", "hybrid_demo_report.json"} and "pitch_calibration" not in path.name


def _eval_record(path: Path) -> dict[str, Any]:
    data = _read_json(path)
    summary = data.get("summary") or {}
    balanced = data.get("balanced_summary") or summary
    chord = data.get("chord_metrics") or {}
    buckets = data.get("duration_buckets") or {}
    hybrid = data.get("hybrid") or {}
    mid = buckets.get("0.5-2s") or {}
    long = buckets.get(">2s") or {}
    return {
        "name": path.stem,
        "path": str(path),
        "ckpt": summary.get("ckpt"),
        "decode_preset": summary.get("decode_preset"),
        "assistant_ckpt": summary.get("assistant_ckpt"),
        "assistant_decode_preset": summary.get("assistant_decode_preset"),
        "note_f1": _f(balanced.get("note_f1")),
        "note_precision": _f(balanced.get("note_precision")),
        "note_recall": _f(balanced.get("note_recall")),
        "offset_f1": _f(balanced.get("offset_f1")),
        "pred_ref_ratio": _f(balanced.get("pred_ref_ratio")),
        "chord_note_recall": _f(chord.get("chord_note_recall")),
        "complete_chord_rate": _f(chord.get("complete_chord_rate")),
        "chord_false_positive_rate": _f(chord.get("chord_false_positive_rate")),
        "duration_ratio_0_5_2s": _f(mid.get("duration_ratio")),
        "duration_ratio_gt_2s": _f(long.get("duration_ratio")),
        "truncated_rate_gt_2s": _f(long.get("truncated_rate")),
        "hybrid_added_notes": _f(hybrid.get("hybrid_added_notes")),
        "hybrid_added_chord_notes": _f(hybrid.get("hybrid_added_chord_notes")),
        "hybrid_added_long_notes": _f(hybrid.get("hybrid_added_long_notes")),
        "hybrid_extended_long_notes": _f(hybrid.get("hybrid_extended_long_notes")),
        "hybrid_rejected_isolated_short": _f(hybrid.get("hybrid_rejected_isolated_short")),
        "assistant_pitch_calibration": data.get("assistant_pitch_calibration_summary") or {},
    }


def _demo_record(path: Path) -> dict[str, Any]:
    data = _read_json(path)
    notation = data.get("score_notation") or {}
    return {
        "name": path.parent.name,
        "debug_json": str(path),
        "musicxml": data.get("musicxml"),
        "musicxml_ok": bool(data.get("musicxml")) and not data.get("musicxml_error"),
        "notes": int(data.get("notes") or 0),
        "score_notes": int(data.get("score_notes") or 0),
        "score_notes_per_performance_note": _f(notation.get("score_notes_per_performance_note")),
        "visible_rest_count": _f(notation.get("visible_rest_count")),
        "rest_density": _f(notation.get("rest_density")),
        "chord_verticality": _f(notation.get("chord_verticality")),
        "voice_collision_count": _f(notation.get("voice_collision_count")),
        "long_note_tie_rate": _f(notation.get("long_note_tie_rate")),
    }


def _pick_hybrid_f1(records: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in records if "hybrid_f1" in row["name"]]
    if not candidates:
        return {}
    in_range = [row for row in candidates if 0.90 <= row["pred_ref_ratio"] <= 1.15]
    pool = in_range or candidates
    return max(pool, key=lambda row: (row["note_f1"], row["offset_f1"], row["chord_note_recall"]))


def _pick_hybrid_score(records: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in records if "hybrid_score" in row["name"]]
    if not candidates:
        return {}
    def score(row: dict[str, Any]) -> tuple[bool, bool, bool, bool, float, float]:
        pred_ok = 0.90 <= row["pred_ref_ratio"] <= 1.05
        note_ok = row["note_f1"] >= 0.5344
        chord_ok = row["chord_note_recall"] >= 0.5055
        long_ok = row["duration_ratio_gt_2s"] >= 0.08 and row["truncated_rate_gt_2s"] <= 0.10
        return (
            pred_ok,
            note_ok,
            chord_ok,
            long_ok,
            row["note_f1"] + row["chord_note_recall"],
            -abs(row["pred_ref_ratio"] - 0.98),
        )
    return max(candidates, key=score)


def _aggregate_demo(records: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "musicxml_ok",
        "notes",
        "score_notes",
        "score_notes_per_performance_note",
        "visible_rest_count",
        "chord_verticality",
        "voice_collision_count",
        "long_note_tie_rate",
    ]
    out = {"items": float(len(records))}
    for key in keys:
        values = [float(row.get(key) or 0.0) for row in records]
        out[f"avg_{key}"] = sum(values) / max(1, len(values))
    return out


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "decode_preset",
        "note_f1",
        "note_precision",
        "note_recall",
        "offset_f1",
        "pred_ref_ratio",
        "chord_note_recall",
        "complete_chord_rate",
        "duration_ratio_0_5_2s",
        "duration_ratio_gt_2s",
        "truncated_rate_gt_2s",
        "hybrid_added_notes",
        "hybrid_added_chord_notes",
        "hybrid_added_long_notes",
        "hybrid_extended_long_notes",
        "hybrid_rejected_isolated_short",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
