from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_pitch_threshold_bias(path: str | Path | None) -> dict[int, float]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return parse_pitch_threshold_bias(data)


def parse_pitch_threshold_bias(data: Any) -> dict[int, float]:
    if not isinstance(data, dict):
        return {}
    for key in ("pitch_threshold_bias", "onset_bias_by_pitch", "pitch_bias", "bias_by_pitch"):
        value = data.get(key)
        if isinstance(value, dict):
            return _coerce_bias_dict(value)
    rows = data.get("by_pitch")
    if rows is None and isinstance(data.get("pitch_calibration"), dict):
        rows = data["pitch_calibration"].get("by_pitch")
    if isinstance(rows, list):
        out: dict[int, float] = {}
        for row in rows:
            if not isinstance(row, dict) or "pitch" not in row:
                continue
            bias = row.get("bias", row.get("threshold_bias_hint"))
            if bias is None:
                continue
            out[int(row["pitch"])] = float(bias)
        return out
    return {}


def summarize_pitch_threshold_bias(bias: dict[int, float]) -> dict[str, Any]:
    values = [float(v) for v in bias.values() if abs(float(v)) > 1e-9]
    if not values:
        return {"pitches": 0, "min_bias": 0.0, "max_bias": 0.0, "lowered": 0, "raised": 0}
    return {
        "pitches": len(values),
        "min_bias": min(values),
        "max_bias": max(values),
        "lowered": sum(1 for value in values if value < 0.0),
        "raised": sum(1 for value in values if value > 0.0),
    }


def _coerce_bias_dict(value: dict[Any, Any]) -> dict[int, float]:
    out: dict[int, float] = {}
    for pitch, bias in value.items():
        out[int(pitch)] = float(bias)
    return out
