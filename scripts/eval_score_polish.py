#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from minimt3.symbolic.score_render import validate_musicxml
from minimt3.utils import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ScorePolish demo/debug outputs.")
    parser.add_argument("--out_dir", default="outputs/demo")
    parser.add_argument("--debug_json", action="append")
    args = parser.parse_args()

    debug_paths = [Path(p) for p in args.debug_json or []]
    if not debug_paths:
        debug_paths = sorted(Path(args.out_dir).glob("*_debug.json"))
    if not debug_paths:
        raise SystemExit("No *_debug.json files found.")

    totals: dict[str, float] = {
        "items": 0.0,
        "musicxml_ok": 0.0,
        "notes": 0.0,
        "score_notes": 0.0,
        "quantization_error_seconds": 0.0,
        "long_note_rate": 0.0,
        "density_pruned_rate": 0.0,
        "overlap_trim_rate": 0.0,
        "chord_collapse_rate": 0.0,
        "hand_crossings": 0.0,
    }
    for path in debug_paths:
        data = read_json(path)
        score = data.get("score_polish") or {}
        metrics = score.get("metrics") or {}
        musicxml_path = data.get("musicxml")
        ok = False
        if musicxml_path:
            try:
                validate_musicxml(musicxml_path)
                ok = True
            except Exception:
                ok = False
        totals["items"] += 1
        totals["musicxml_ok"] += 1.0 if ok else 0.0
        totals["notes"] += float(data.get("notes", 0))
        totals["score_notes"] += float(data.get("score_notes", 0))
        for key in (
            "quantization_error_seconds",
            "long_note_rate",
            "density_pruned_rate",
            "overlap_trim_rate",
            "chord_collapse_rate",
            "hand_crossings",
        ):
            totals[key] += float(metrics.get(key, 0.0))
        print(
            "score_item "
            f"path={path} musicxml_ok={ok} notes={data.get('notes', 0)} score_notes={data.get('score_notes', 0)} "
            f"key={score.get('key_signature')} tempo={score.get('tempo_bpm')} metrics={metrics}",
            flush=True,
        )

    count = max(1.0, totals["items"])
    print(
        "score_summary "
        f"items={int(totals['items'])} musicxml_ok_rate={totals['musicxml_ok'] / count:.3f} "
        f"avg_notes={totals['notes'] / count:.1f} avg_score_notes={totals['score_notes'] / count:.1f} "
        f"avg_quant_error={totals['quantization_error_seconds'] / count:.4f} "
        f"avg_long_note_rate={totals['long_note_rate'] / count:.4f} "
        f"avg_density_pruned_rate={totals['density_pruned_rate'] / count:.4f} "
        f"avg_overlap_trim_rate={totals['overlap_trim_rate'] / count:.4f} "
        f"avg_chord_collapse_rate={totals['chord_collapse_rate'] / count:.4f} "
        f"avg_hand_crossings={totals['hand_crossings'] / count:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
