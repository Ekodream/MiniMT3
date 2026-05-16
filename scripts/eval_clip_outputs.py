#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import mir_eval.transcription as mir_transcription

from minimt3.eval.metrics import note_arrays


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize short inference outputs.")
    parser.add_argument("--pred_dir", default="outputs/eval_best")
    parser.add_argument("--ref_dir", default="outputs/eval_refs")
    args = parser.parse_args()

    for pred_path in sorted(glob.glob(str(Path(args.pred_dir) / "*.mid"))):
        pred = Path(pred_path)
        ref = Path(args.ref_dir) / pred.name
        if not ref.exists():
            print("METRIC", pred.stem, "missing_ref", ref)
            continue
        ref_intervals, ref_pitches, _ = note_arrays(ref)
        pred_intervals, pred_pitches, _ = note_arrays(pred)
        onset = mir_transcription.precision_recall_f1_overlap(
            ref_intervals,
            ref_pitches,
            pred_intervals,
            pred_pitches,
            offset_ratio=None,
        )
        offset = mir_transcription.precision_recall_f1_overlap(
            ref_intervals,
            ref_pitches,
            pred_intervals,
            pred_pitches,
            offset_ratio=0.2,
        )
        print(
            "METRIC",
            pred.stem,
            "pred",
            len(pred_intervals),
            "ref",
            len(ref_intervals),
            "note_p/r/f1",
            round(float(onset[0]), 4),
            round(float(onset[1]), 4),
            round(float(onset[2]), 4),
            "offset_f1",
            round(float(offset[2]), 4),
        )

    for debug_path in sorted(glob.glob(str(Path(args.pred_dir) / "*.json"))):
        debug = Path(debug_path)
        data = json.loads(debug.read_text(encoding="utf-8"))
        window = data["windows"][0] if data.get("windows") else {}
        print(
            "DEBUG",
            debug.stem,
            "eos",
            window.get("eos_hit"),
            "stop",
            window.get("stop_reason"),
            "families",
            window.get("token_family_counts"),
            "time",
            round(float(window.get("decode_wall_time", 0.0)), 2),
            "tok/s",
            round(float(window.get("tokens_per_second", 0.0)), 2),
            "notes",
            len(data.get("notes", [])),
            "pedals",
            len(data.get("pedals", [])),
            "invalid",
            round(float(window.get("invalid_event_rate", 0.0)), 4),
        )


if __name__ == "__main__":
    main()
