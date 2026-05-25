#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-pitch decode threshold calibration from eval JSON.")
    parser.add_argument("--eval_json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--clamp_min", type=float, default=-0.06)
    parser.add_argument("--clamp_max", type=float, default=0.04)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--min_observations", type=int, default=3)
    args = parser.parse_args()

    data = _read_json(args.eval_json)
    rows = _pitch_rows(data)
    biases: dict[str, float] = {}
    out_rows = []
    for row in rows:
        pitch = int(row["pitch"])
        fp = float(row.get("false_positives", 0.0))
        fn = float(row.get("false_negatives", 0.0))
        observations = fp + fn
        if observations < int(args.min_observations):
            continue
        hint = row.get("threshold_bias_hint")
        if hint is None:
            hint = 0.08 * (fp - fn) / max(1.0, observations)
        bias = _clamp(float(hint) * float(args.scale), float(args.clamp_min), float(args.clamp_max))
        if abs(bias) < 1e-9:
            continue
        biases[str(pitch)] = bias
        out_rows.append(
            {
                "pitch": pitch,
                "false_positives": fp,
                "false_negatives": fn,
                "observations": observations,
                "bias": bias,
            }
        )
    payload = {
        "source": str(args.eval_json),
        "pitch_threshold_bias": biases,
        "by_pitch": sorted(out_rows, key=lambda item: item["pitch"]),
        "summary": {
            "pitches": len(biases),
            "lowered": sum(1 for value in biases.values() if value < 0.0),
            "raised": sum(1 for value in biases.values() if value > 0.0),
            "clamp_min": float(args.clamp_min),
            "clamp_max": float(args.clamp_max),
            "scale": float(args.scale),
            "min_observations": int(args.min_observations),
        },
    }
    _write_json(args.out, payload)
    print(
        f"wrote {args.out}: pitches={payload['summary']['pitches']} "
        f"lowered={payload['summary']['lowered']} raised={payload['summary']['raised']}"
    )


def _pitch_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    direct = data.get("by_pitch")
    if isinstance(direct, list):
        return [row for row in direct if isinstance(row, dict)]
    calibration = data.get("pitch_calibration")
    if isinstance(calibration, dict) and isinstance(calibration.get("by_pitch"), list):
        return [row for row in calibration["by_pitch"] if isinstance(row, dict)]
    raise SystemExit("eval JSON does not contain pitch_calibration.by_pitch")


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


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
