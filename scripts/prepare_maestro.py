#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from minimt3.data import index_maestro
from minimt3.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Index MAESTRO metadata for MiniMT3-Piano.")
    parser.add_argument("--data_dir", required=True, help="Directory containing MAESTRO v3 files.")
    parser.add_argument("--out", default="data/cache", help="Output cache directory.")
    args = parser.parse_args()

    rows = index_maestro(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "maestro_index.json"
    write_json(out_path, rows)
    counts = Counter(row["split"] for row in rows)
    missing_audio = sum(not row["audio_exists"] for row in rows)
    missing_midi = sum(not row["midi_exists"] for row in rows)
    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"Splits: {dict(counts)}")
    print(f"Missing audio: {missing_audio}; missing MIDI: {missing_midi}")


if __name__ == "__main__":
    main()
