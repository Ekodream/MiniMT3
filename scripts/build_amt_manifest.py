#!/usr/bin/env python
from __future__ import annotations

import argparse

from minimt3.utils import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed dense-AMT clip manifests from MAESTRO index.")
    parser.add_argument("--index", default="data/cache/maestro_index.json")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", required=True)
    parser.add_argument("--clip_seconds", type=float, default=2.0)
    parser.add_argument("--stride_seconds", type=float, default=4.0)
    parser.add_argument("--max_clips", type=int, default=0)
    parser.add_argument("--max_clips_per_piece", type=int, default=0)
    args = parser.parse_args()

    rows = [
        row
        for row in read_json(args.index)
        if row.get("split") == args.split
        and row.get("audio_exists", True)
        and row.get("midi_exists", True)
        and float(row.get("duration") or 0.0) > 0
    ]
    clips = []
    for piece_idx, row in enumerate(rows):
        duration = float(row.get("duration") or 0.0)
        max_start = max(0.0, duration - args.clip_seconds)
        starts = []
        current = 0.0 if max_start == 0 else min(max_start, args.clip_seconds)
        while current <= max_start + 1e-6:
            starts.append(current)
            current += args.stride_seconds
            if args.max_clips_per_piece and len(starts) >= args.max_clips_per_piece:
                break
        if not starts:
            starts = [0.0]
        for clip_idx, start in enumerate(starts):
            end = min(duration, start + args.clip_seconds)
            clips.append(
                {
                    "clip_id": f"{args.split}_{piece_idx:04d}_{clip_idx:04d}",
                    "split": args.split,
                    "audio": row["audio"],
                    "midi": row["midi"],
                    "start_sec": round(start, 3),
                    "end_sec": round(end, 3),
                    "duration": round(end - start, 3),
                    "composer": row.get("composer", ""),
                    "title": row.get("title", ""),
                    "audio_exists": row.get("audio_exists", True),
                    "midi_exists": row.get("midi_exists", True),
                }
            )
            if args.max_clips and len(clips) >= args.max_clips:
                write_json(args.out, clips)
                print(f"wrote {len(clips)} clips to {args.out}")
                return
    write_json(args.out, clips)
    print(f"wrote {len(clips)} clips to {args.out}")


if __name__ == "__main__":
    main()
