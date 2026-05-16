#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from minimt3.data import build_fixed_clip_manifest
from minimt3.utils import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic fixed validation/debug clips.")
    parser.add_argument("--index", default="data/cache/maestro_index.json")
    parser.add_argument("--out_dir", default="data/cache")
    parser.add_argument("--val_split", default="validation")
    parser.add_argument("--debug_split", default="validation")
    parser.add_argument("--clip_seconds", type=float, default=20.0)
    parser.add_argument("--val_count", type=int, default=50)
    parser.add_argument("--debug_count", type=int, default=8)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    rows = read_json(args.index)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val = build_fixed_clip_manifest(rows, args.val_split, args.clip_seconds, args.val_count)
    debug = build_fixed_clip_manifest(rows, args.debug_split, args.clip_seconds, args.debug_count)
    suffix = f"_{args.suffix}" if args.suffix else ""
    val_path = out_dir / f"maestro_val_clips{suffix}.json"
    debug_path = out_dir / f"maestro_debug_clips{suffix}.json"
    write_json(val_path, val)
    write_json(debug_path, debug)
    print(f"Wrote {len(val)} fixed validation clips to {val_path}")
    print(f"Wrote {len(debug)} fixed debug clips to {debug_path}")


if __name__ == "__main__":
    main()
