from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from minimt3.symbolic.events import NoteEvent, load_midi_events


DEFAULT_DURATION_BOUNDS = [0.0, 0.125, 0.5, 2.0, math.inf]


def model_parameter_count(model) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def manifest_size(path: str | Path | None) -> int | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        for key in ("items", "clips", "data"):
            value = data.get(key)
            if isinstance(value, list):
                return len(value)
    return None


def parse_duration_buckets(value: str | None) -> list[tuple[str, float, float]]:
    if not value:
        bounds = DEFAULT_DURATION_BOUNDS
    else:
        bounds = []
        for part in value.split(","):
            part = part.strip().lower()
            if not part:
                continue
            bounds.append(math.inf if part in {"inf", "infinity"} else float(part))
        if len(bounds) < 2:
            bounds = DEFAULT_DURATION_BOUNDS
    buckets = []
    for lo, hi in zip(bounds, bounds[1:]):
        if math.isinf(hi):
            label = f">{lo:g}s"
        elif lo <= 0.0:
            label = f"<{hi:g}s"
        else:
            label = f"{lo:g}-{hi:g}s"
        buckets.append((label, float(lo), float(hi)))
    return buckets


def detailed_note_metrics(
    pred_notes: list[NoteEvent],
    ref_notes: list[NoteEvent],
    onset_tolerance: float = 0.05,
    offset_ratio: float = 0.2,
) -> dict[str, Any]:
    matches = match_notes(pred_notes, ref_notes, onset_tolerance=onset_tolerance, offset_ratio=offset_ratio)
    tp = len(matches)
    fp = max(0, len(pred_notes) - tp)
    fn = max(0, len(ref_notes) - tp)
    offset_tp = sum(1 for item in matches if item["offset_ok"])
    onset_precision = _safe_div(tp, tp + fp)
    onset_recall = _safe_div(tp, tp + fn)
    offset_precision = _safe_div(offset_tp, len(pred_notes))
    offset_recall = _safe_div(offset_tp, len(ref_notes))
    onset_errors = [float(item["onset_error"]) for item in matches]
    offset_errors = [float(item["offset_error"]) for item in matches]
    velocity_errors = [abs(float(item["velocity_error"])) for item in matches]
    signed_velocity_errors = [float(item["velocity_error"]) for item in matches]
    return {
        "note_precision": onset_precision,
        "note_recall": onset_recall,
        "note_f1": _f1(onset_precision, onset_recall),
        "offset_precision": offset_precision,
        "offset_recall": offset_recall,
        "offset_f1": _f1(offset_precision, offset_recall),
        "note_tp": tp,
        "note_fp": fp,
        "note_fn": fn,
        "offset_tp": offset_tp,
        "pred_notes": len(pred_notes),
        "ref_notes": len(ref_notes),
        "onset_mae": _mean_abs(onset_errors),
        "onset_bias": _mean(onset_errors),
        "offset_mae": _mean_abs(offset_errors),
        "offset_bias": _mean(offset_errors),
        "velocity_mae": _mean_abs(velocity_errors),
        "velocity_bias": _mean(signed_velocity_errors),
        "matches": matches,
    }


def match_notes(
    pred_notes: list[NoteEvent],
    ref_notes: list[NoteEvent],
    onset_tolerance: float = 0.05,
    offset_ratio: float = 0.2,
) -> list[dict[str, Any]]:
    pred_by_pitch: dict[int, list[int]] = {}
    for idx, note in enumerate(pred_notes):
        pred_by_pitch.setdefault(int(note.pitch), []).append(idx)
    used_pred: set[int] = set()
    matches: list[dict[str, Any]] = []
    ref_order = sorted(range(len(ref_notes)), key=lambda idx: (ref_notes[idx].start, ref_notes[idx].pitch))
    for ref_idx in ref_order:
        ref = ref_notes[ref_idx]
        best_idx = None
        best_error = math.inf
        for pred_idx in pred_by_pitch.get(int(ref.pitch), []):
            if pred_idx in used_pred:
                continue
            pred = pred_notes[pred_idx]
            error = abs(float(pred.start) - float(ref.start))
            if error <= onset_tolerance and error < best_error:
                best_idx = pred_idx
                best_error = error
        if best_idx is None:
            continue
        used_pred.add(best_idx)
        pred = pred_notes[best_idx]
        ref_duration = max(1e-6, float(ref.end) - float(ref.start))
        offset_tolerance = max(0.05, float(offset_ratio) * ref_duration)
        offset_error = float(pred.end) - float(ref.end)
        pred_duration = max(0.0, float(pred.end) - float(pred.start))
        matches.append(
            {
                "ref_index": ref_idx,
                "pred_index": best_idx,
                "pitch": int(ref.pitch),
                "onset_error": float(pred.start) - float(ref.start),
                "offset_error": offset_error,
                "offset_ok": abs(offset_error) <= offset_tolerance,
                "ref_duration": ref_duration,
                "pred_duration": pred_duration,
                "duration_ratio": pred_duration / ref_duration,
                "velocity_error": int(pred.velocity) - int(ref.velocity),
            }
        )
    matches.sort(key=lambda item: (item["ref_index"], item["pred_index"]))
    return matches


def new_metric_total() -> dict[str, float]:
    return {
        "note_tp": 0.0,
        "note_fp": 0.0,
        "note_fn": 0.0,
        "offset_tp": 0.0,
        "pred_notes": 0.0,
        "ref_notes": 0.0,
        "onset_abs_sum": 0.0,
        "onset_sum": 0.0,
        "offset_abs_sum": 0.0,
        "offset_sum": 0.0,
        "velocity_abs_sum": 0.0,
        "velocity_sum": 0.0,
        "match_count": 0.0,
    }


def add_metric_total(total: dict[str, float], metric: dict[str, Any]) -> None:
    total["note_tp"] += float(metric["note_tp"])
    total["note_fp"] += float(metric["note_fp"])
    total["note_fn"] += float(metric["note_fn"])
    total["offset_tp"] += float(metric["offset_tp"])
    total["pred_notes"] += float(metric["pred_notes"])
    total["ref_notes"] += float(metric["ref_notes"])
    for item in metric.get("matches", []):
        total["onset_abs_sum"] += abs(float(item["onset_error"]))
        total["onset_sum"] += float(item["onset_error"])
        total["offset_abs_sum"] += abs(float(item["offset_error"]))
        total["offset_sum"] += float(item["offset_error"])
        total["velocity_abs_sum"] += abs(float(item["velocity_error"]))
        total["velocity_sum"] += float(item["velocity_error"])
        total["match_count"] += 1.0


def summarize_metric_total(total: dict[str, float]) -> dict[str, float]:
    note_precision = _safe_div(total["note_tp"], total["note_tp"] + total["note_fp"])
    note_recall = _safe_div(total["note_tp"], total["note_tp"] + total["note_fn"])
    offset_precision = _safe_div(total["offset_tp"], total["pred_notes"])
    offset_recall = _safe_div(total["offset_tp"], total["ref_notes"])
    match_count = max(1.0, total["match_count"])
    return {
        "note_precision": note_precision,
        "note_recall": note_recall,
        "note_f1": _f1(note_precision, note_recall),
        "offset_precision": offset_precision,
        "offset_recall": offset_recall,
        "offset_f1": _f1(offset_precision, offset_recall),
        "pred_ref_ratio": _safe_div(total["pred_notes"], total["ref_notes"]),
        "pred_notes": total["pred_notes"],
        "ref_notes": total["ref_notes"],
        "onset_mae": total["onset_abs_sum"] / match_count,
        "onset_bias": total["onset_sum"] / match_count,
        "offset_mae": total["offset_abs_sum"] / match_count,
        "offset_bias": total["offset_sum"] / match_count,
        "velocity_mae": total["velocity_abs_sum"] / match_count,
        "velocity_bias": total["velocity_sum"] / match_count,
    }


def duration_bucket_metrics(
    pred_notes: list[NoteEvent],
    ref_notes: list[NoteEvent],
    matches: list[dict[str, Any]],
    buckets: list[tuple[str, float, float]],
) -> dict[str, dict[str, float]]:
    by_ref = {int(item["ref_index"]): item for item in matches}
    by_pred = {int(item["pred_index"]): item for item in matches}
    out = {
        label: {
            "ref_notes": 0.0,
            "pred_false_positives": 0.0,
            "onset_recall": 0.0,
            "offset_recall": 0.0,
            "duration_ratio": 0.0,
            "truncated_rate": 0.0,
            "extended_rate": 0.0,
        }
        for label, _, _ in buckets
    }
    ratio_counts = {label: 0 for label, _, _ in buckets}
    for idx, note in enumerate(ref_notes):
        label = _duration_label(note.end - note.start, buckets)
        row = out[label]
        row["ref_notes"] += 1.0
        match = by_ref.get(idx)
        if not match:
            continue
        row["onset_recall"] += 1.0
        if match["offset_ok"]:
            row["offset_recall"] += 1.0
        ratio = float(match["duration_ratio"])
        row["duration_ratio"] += ratio
        row["truncated_rate"] += 1.0 if ratio < 0.75 else 0.0
        row["extended_rate"] += 1.0 if ratio > 1.25 else 0.0
        ratio_counts[label] += 1
    for idx, note in enumerate(pred_notes):
        if idx in by_pred:
            continue
        out[_duration_label(note.end - note.start, buckets)]["pred_false_positives"] += 1.0
    for label, row in out.items():
        refs = max(1.0, row["ref_notes"])
        matched = max(1, ratio_counts[label])
        row["onset_recall"] /= refs
        row["offset_recall"] /= refs
        row["duration_ratio"] = row["duration_ratio"] / matched if ratio_counts[label] else 0.0
        row["truncated_rate"] = row["truncated_rate"] / matched if ratio_counts[label] else 0.0
        row["extended_rate"] = row["extended_rate"] / matched if ratio_counts[label] else 0.0
    return out


def chord_metrics(
    pred_notes: list[NoteEvent],
    ref_notes: list[NoteEvent],
    matches: list[dict[str, Any]],
    tolerance: float = 0.05,
) -> dict[str, float]:
    ref_groups = [group for group in group_by_start(ref_notes, tolerance) if len(group) >= 2]
    pred_groups = [group for group in group_by_start(pred_notes, tolerance) if len(group) >= 2]
    by_ref = {int(item["ref_index"]): item for item in matches}
    matched_pred = {int(item["pred_index"]) for item in matches}
    expected = 0
    matched = 0
    complete = 0
    split = 0
    spread_sum = 0.0
    spread_count = 0
    ref_index_by_id = {id(note): idx for idx, note in enumerate(ref_notes)}
    for group in ref_groups:
        group_ref_indices = [ref_index_by_id[id(note)] for note in group]
        group_matches = [by_ref[idx] for idx in group_ref_indices if idx in by_ref]
        expected += len(group)
        matched += len(group_matches)
        if len(group_matches) == len(group):
            complete += 1
        starts = [pred_notes[int(item["pred_index"])].start for item in group_matches]
        if len(starts) >= 2:
            spread = max(starts) - min(starts)
            spread_sum += spread
            spread_count += 1
            if spread > tolerance:
                split += 1
    pred_chord_notes = sum(len(group) for group in pred_groups)
    matched_pred_chord_notes = 0
    pred_index_by_id = {id(note): idx for idx, note in enumerate(pred_notes)}
    for group in pred_groups:
        pred_ids = {pred_index_by_id[id(note)] for note in group}
        matched_pred_chord_notes += len(pred_ids & matched_pred)
    precision = _safe_div(matched_pred_chord_notes, pred_chord_notes)
    return {
        "ref_chords": float(len(ref_groups)),
        "pred_chords": float(len(pred_groups)),
        "chord_note_recall": _safe_div(matched, expected),
        "chord_note_precision": precision,
        "chord_false_positive_rate": 1.0 - precision if pred_chord_notes else 0.0,
        "complete_chord_rate": _safe_div(complete, len(ref_groups)),
        "chord_split_rate": _safe_div(split, len(ref_groups)),
        "chord_onset_spread": _safe_div(spread_sum, spread_count),
    }


def score_quality_metrics(notes: list[NoteEvent], chord_tolerance_seconds: float = 0.075) -> dict[str, Any]:
    if not notes:
        return {
            "raw_notes": 0.0,
            "score_notes": 0.0,
            "score_pruned_rate": 0.0,
            "score_notation": score_notation_report([], performance_note_count=0),
        }
    try:
        from minimt3.symbolic.score_polish import ScorePolishConfig, polish_score_notes
    except Exception:
        return {"raw_notes": float(len(notes)), "score_notes": 0.0, "score_pruned_rate": 1.0}
    result = polish_score_notes(
        notes,
        config=ScorePolishConfig(
            chord_tolerance_seconds=chord_tolerance_seconds,
            chord_snap_seconds=max(0.0, chord_tolerance_seconds),
            max_note_beats=6.0,
            max_notes_per_beat=5,
        ),
    )
    metrics = {key: float(value) for key, value in result.metrics.items() if isinstance(value, (int, float))}
    metrics["raw_notes"] = float(len(notes))
    metrics["score_notes"] = float(len(result.notes))
    metrics["score_pruned_rate"] = max(0.0, 1.0 - len(result.notes) / max(1, len(notes)))
    metrics["score_notation"] = score_notation_report(
        result.notes,
        seconds_per_quarter=result.seconds_per_quarter,
        key_signature=result.key_signature,
        time_signature=result.time_signature,
        right_notes=result.right_notes,
        left_notes=result.left_notes,
        beat_divisions=result.beat_divisions,
        performance_note_count=len(notes),
    )
    return metrics


def score_notation_report(
    notes: list[NoteEvent],
    seconds_per_quarter: float = 0.5,
    key_signature: str | None = None,
    time_signature: str = "4/4",
    right_notes: list[NoteEvent] | None = None,
    left_notes: list[NoteEvent] | None = None,
    beat_divisions: tuple[int, ...] = (4,),
    performance_note_count: int | None = None,
    key_signature_source: str = "auto",
    time_signature_source: str = "auto",
) -> dict[str, Any]:
    try:
        from minimt3.symbolic.score_render import score_notation_metrics
    except Exception:
        return {}
    return score_notation_metrics(
        notes,
        seconds_per_quarter=seconds_per_quarter,
        key_signature=key_signature,
        time_signature=time_signature,
        right_notes=right_notes,
        left_notes=left_notes,
        beat_divisions=beat_divisions,
        performance_note_count=performance_note_count,
        key_signature_source=key_signature_source,
        time_signature_source=time_signature_source,
    )


def error_records(
    pred_notes: list[NoteEvent],
    ref_notes: list[NoteEvent],
    matches: list[dict[str, Any]],
    clip_id: str,
    limit: int = 12,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matched_pred = {int(item["pred_index"]) for item in matches}
    matched_ref = {int(item["ref_index"]) for item in matches}
    fps = [
        {"clip_id": clip_id, "kind": "false_positive", **asdict(note), "duration": note.end - note.start}
        for idx, note in enumerate(pred_notes)
        if idx not in matched_pred
    ]
    fns = [
        {"clip_id": clip_id, "kind": "false_negative", **asdict(note), "duration": note.end - note.start}
        for idx, note in enumerate(ref_notes)
        if idx not in matched_ref
    ]
    fps.sort(key=lambda item: (item["start"], item["pitch"]))
    fns.sort(key=lambda item: (item["start"], item["pitch"]))
    return fps[:limit], fns[:limit]


def group_by_start(notes: list[NoteEvent], tolerance: float) -> list[list[NoteEvent]]:
    groups: list[list[NoteEvent]] = []
    for note in sorted(notes, key=lambda item: (item.start, item.pitch, item.end)):
        if groups and note.start - groups[-1][0].start <= tolerance:
            groups[-1].append(note)
        else:
            groups.append([note])
    return groups


def find_teacher_midi(teacher_midi_dir: str | Path | None, row: dict[str, Any]) -> Path | None:
    if not teacher_midi_dir:
        return None
    root = Path(teacher_midi_dir)
    candidates = [
        root / f"{row.get('clip_id')}.mid",
        root / f"{Path(str(row.get('audio', ''))).stem}.mid",
        root / f"{Path(str(row.get('midi', ''))).stem}.mid",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_teacher_notes(teacher_midi_dir: str | Path | None, row: dict[str, Any]) -> list[NoteEvent] | None:
    path = find_teacher_midi(teacher_midi_dir, row)
    if path is None:
        return None
    notes, _ = load_midi_events(path)
    return notes


def _duration_label(duration: float, buckets: list[tuple[str, float, float]]) -> str:
    for label, lo, hi in buckets:
        if lo <= duration < hi:
            return label
    return buckets[-1][0]


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_abs(values: list[float]) -> float:
    return sum(abs(value) for value in values) / len(values) if values else 0.0
