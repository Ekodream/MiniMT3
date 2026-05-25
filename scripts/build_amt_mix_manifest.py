#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
from collections import defaultdict

from minimt3.utils import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a regular/hard ratio manifest from an existing hardmix manifest.")
    parser.add_argument("--source_manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--hard_fraction", type=float, default=0.30)
    parser.add_argument("--max_hard_clips", type=int)
    parser.add_argument("--seed", type=int, default=186)
    args = parser.parse_args()

    rows = [
        row
        for row in read_json(args.source_manifest)
        if row.get("split") == args.split
        and row.get("audio_exists", True)
        and row.get("midi_exists", True)
    ]
    base = [row for row in rows if str(row.get("hard_category", "base")) == "base"]
    hard = [row for row in rows if str(row.get("hard_category", "base")) != "base"]
    if not base:
        raise SystemExit("source manifest has no base rows")
    rng = random.Random(int(args.seed))
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in hard:
        by_category[str(row.get("hard_category", "hard"))].append(row)
    for values in by_category.values():
        values.sort(key=lambda row: str(row.get("clip_id", "")))
        rng.shuffle(values)

    requested = int(round(len(base) * args.hard_fraction / max(1e-6, 1.0 - args.hard_fraction)))
    if args.max_hard_clips is not None:
        requested = min(requested, max(0, int(args.max_hard_clips)))
    requested = min(requested, len(hard))
    selected: list[dict] = []
    categories = sorted(by_category)
    while len(selected) < requested:
        added = False
        for category in categories:
            if by_category[category] and len(selected) < requested:
                selected.append(by_category[category].pop())
                added = True
        if not added:
            break
    selected.sort(key=lambda row: str(row.get("clip_id", "")))
    mixed = base + selected
    write_json(args.out, mixed)
    print(
        "mix_manifest "
        f"source={args.source_manifest} out={args.out} base={len(base)} hard={len(selected)} total={len(mixed)} "
        f"hard_fraction={len(selected) / max(1, len(mixed)):.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
