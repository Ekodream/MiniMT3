from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from minimt3.symbolic.midi_io import read_midi
from minimt3.utils import read_json


def note_arrays(midi_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    notes, _ = read_midi(midi_path)
    if not notes:
        return np.zeros((0, 2)), np.zeros((0,), dtype=int), np.zeros((0,))
    intervals = np.array([[n.start, n.end] for n in notes], dtype=float)
    pitches = np.array([n.pitch for n in notes], dtype=int)
    velocities = np.array([n.velocity for n in notes], dtype=float) / 127.0
    return intervals, pitches, velocities


def evaluate_pair(pred_midi: str | Path, ref_midi: str | Path) -> dict[str, float]:
    try:
        import mir_eval.transcription
    except ImportError as exc:
        raise ImportError("Install mir_eval to run note metrics: pip install mir_eval") from exc

    ref_intervals, ref_pitches, ref_vel = note_arrays(ref_midi)
    pred_intervals, pred_pitches, pred_vel = note_arrays(pred_midi)
    onset = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches, pred_intervals, pred_pitches, offset_ratio=None
    )
    offset = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches, pred_intervals, pred_pitches, offset_ratio=0.2
    )
    velocity = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals,
        ref_pitches,
        pred_intervals,
        pred_pitches,
        ref_velocities=ref_vel,
        est_velocities=pred_vel,
        offset_ratio=0.2,
    )
    return {
        "note_precision": float(onset[0]),
        "note_recall": float(onset[1]),
        "note_f1": float(onset[2]),
        "note_offset_precision": float(offset[0]),
        "note_offset_recall": float(offset[1]),
        "note_offset_f1": float(offset[2]),
        "velocity_precision": float(velocity[0]),
        "velocity_recall": float(velocity[1]),
        "velocity_f1": float(velocity[2]),
    }


def evaluate_directory(pred_dir: str | Path, ref_meta: str | Path) -> dict[str, Any]:
    pred_dir = Path(pred_dir)
    rows = read_json(ref_meta)
    ref_by_stem = {Path(row["midi"]).stem: row["midi"] for row in rows if row.get("midi_exists", True)}
    results: dict[str, Any] = {"items": []}
    for pred in sorted(pred_dir.glob("*.mid")):
        ref = ref_by_stem.get(pred.stem)
        if not ref:
            continue
        item = {"stem": pred.stem, **evaluate_pair(pred, ref)}
        debug = pred.with_suffix(".json")
        if debug.exists():
            with debug.open("r", encoding="utf-8") as f:
                data = json.load(f)
            rates = [w.get("invalid_event_rate", 0.0) for w in data.get("windows", [])]
            item["invalid_event_rate"] = float(np.mean(rates)) if rates else 0.0
        results["items"].append(item)
    results["summary"] = _summarize(results["items"])
    return results


def _summarize(items: list[dict[str, Any]]) -> dict[str, float]:
    if not items:
        return {}
    keys = [k for k, v in items[0].items() if isinstance(v, (int, float))]
    return {key: float(np.mean([item[key] for item in items if key in item])) for key in keys}
