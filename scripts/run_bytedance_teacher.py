#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from minimt3.audio.preprocess import load_audio
from minimt3.utils import ensure_dir, read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ByteDance piano-transcription teacher MIDI for AMT clips.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split", default="")
    parser.add_argument("--items", type=int)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only_hard", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary_json")
    args = parser.parse_args()

    try:
        from piano_transcription_inference import PianoTranscription, sample_rate
    except Exception as exc:
        raise SystemExit(
            "Missing piano_transcription_inference. Install it in the active environment with: "
            "pip install piano-transcription-inference"
        ) from exc

    rows = [
        row
        for row in read_json(args.manifest)
        if (not args.split or row.get("split") == args.split)
        and row.get("audio_exists", True)
        and (not args.only_hard or str(row.get("hard_category", "base")) != "base")
    ]
    rows = rows[max(0, int(args.start_index)) :]
    if args.items:
        rows = rows[: max(0, int(args.items))]
    if not rows:
        raise SystemExit("No manifest rows selected.")

    out_dir = ensure_dir(args.out_dir)
    transcriptor = PianoTranscription(device=str(args.device))
    done = 0
    skipped = 0
    failed: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        clip_id = str(row.get("clip_id", idx))
        out_path = out_dir / f"{clip_id}.mid"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        start = float(row.get("start_sec", 0.0))
        end = float(row.get("end_sec", start + float(row.get("duration", 0.0) or 0.0)))
        duration = max(0.01, end - start)
        try:
            waveform = load_audio(
                row["audio"],
                sample_rate=int(sample_rate),
                offset_seconds=start,
                duration_seconds=duration,
            )
            audio = waveform.mean(dim=0).detach().cpu().numpy().astype(np.float32, copy=False)
            transcriptor.transcribe(audio, str(out_path))
            done += 1
        except Exception as exc:
            failed.append({"clip_id": clip_id, "audio": row.get("audio"), "error": repr(exc)})
        if (done + skipped + len(failed)) % 25 == 0:
            print(
                f"teacher_progress selected={len(rows)} done={done} skipped={skipped} failed={len(failed)}",
                flush=True,
            )

    summary = {
        "manifest": args.manifest,
        "out_dir": str(out_dir),
        "selected": len(rows),
        "done": done,
        "skipped": skipped,
        "failed": failed,
    }
    if args.summary_json:
        write_json(args.summary_json, summary)
    print(
        f"teacher_summary selected={len(rows)} done={done} skipped={skipped} failed={len(failed)} out={out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
