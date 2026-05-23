#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict

from minimt3.symbolic.events import NoteEvent, load_midi_events
from minimt3.utils import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small hard-case manifest for score readability checks.")
    parser.add_argument("--index", default="data/cache/maestro_index.json")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--out", required=True)
    parser.add_argument("--clip_seconds", type=float, default=30.0)
    parser.add_argument("--stride_seconds", type=float, default=30.0)
    parser.add_argument("--max_clips", type=int, default=30)
    parser.add_argument("--max_per_category", type=int, default=8)
    args = parser.parse_args()

    rows = [
        row
        for row in read_json(args.index)
        if row.get("split") == args.split
        and row.get("audio_exists", True)
        and row.get("midi_exists", True)
        and float(row.get("duration") or 0.0) > 0
    ]
    candidates: dict[str, list[tuple[float, dict]]] = defaultdict(list)
    for piece_idx, row in enumerate(rows):
        duration = float(row.get("duration") or 0.0)
        max_start = max(0.0, duration - args.clip_seconds)
        start = 0.0 if max_start <= 0.0 else min(max_start, args.clip_seconds)
        clip_idx = 0
        while start <= max_start + 1e-6:
            end = min(duration, start + args.clip_seconds)
            notes, pedals = load_midi_events(row["midi"], start=start, end=end)
            features = _clip_features(notes, len(pedals), end - start)
            base = {
                "clip_id": f"{args.split}_score_{piece_idx:04d}_{clip_idx:04d}",
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
                "score_quality_features": features,
            }
            _add_candidates(candidates, base, features)
            clip_idx += 1
            start += args.stride_seconds

    selected = []
    seen = set()
    category_order = ["dense_chords", "low_long_notes", "pedal_legato", "weak_accompaniment", "fast_arpeggio"]
    while len(selected) < args.max_clips:
        added = False
        for category in category_order:
            pool = sorted(candidates.get(category, []), key=lambda item: item[0], reverse=True)
            used_in_category = sum(1 for row in selected if row.get("score_quality_category") == category)
            if used_in_category >= args.max_per_category:
                continue
            while pool:
                _, row = pool.pop(0)
                key = (row["audio"], row["start_sec"], row["end_sec"])
                if key in seen:
                    continue
                row = dict(row)
                row["score_quality_category"] = category
                selected.append(row)
                seen.add(key)
                added = True
                break
            candidates[category] = pool
            if len(selected) >= args.max_clips:
                break
        if not added:
            break
    write_json(args.out, selected)
    print(f"wrote {len(selected)} score-quality clips to {args.out}")


def _clip_features(notes: list[NoteEvent], pedal_count: int, duration: float) -> dict[str, float]:
    if not notes:
        return {
            "notes": 0.0,
            "onsets_per_second": 0.0,
            "max_polyphony": 0.0,
            "chord_notes": 0.0,
            "long_notes": 0.0,
            "low_long_notes": 0.0,
            "weak_notes": 0.0,
            "pedals": float(pedal_count),
        }
    groups = _group_by_start(notes, tolerance=0.06)
    max_polyphony = max(len(group) for group in groups)
    chord_notes = sum(len(group) for group in groups if len(group) >= 3)
    long_notes = sum(1 for note in notes if note.end - note.start >= 2.0)
    low_long_notes = sum(1 for note in notes if note.pitch <= 48 and note.end - note.start >= 1.5)
    weak_notes = sum(1 for note in notes if note.velocity <= 45)
    return {
        "notes": float(len(notes)),
        "onsets_per_second": len(groups) / max(1e-6, duration),
        "max_polyphony": float(max_polyphony),
        "chord_notes": float(chord_notes),
        "long_notes": float(long_notes),
        "low_long_notes": float(low_long_notes),
        "weak_notes": float(weak_notes),
        "pedals": float(pedal_count),
    }


def _add_candidates(candidates: dict[str, list[tuple[float, dict]]], row: dict, features: dict[str, float]) -> None:
    if features["chord_notes"] >= 12 or features["max_polyphony"] >= 5:
        candidates["dense_chords"].append((features["chord_notes"] + features["max_polyphony"], row))
    if features["low_long_notes"] >= 2:
        candidates["low_long_notes"].append((features["low_long_notes"] + 0.1 * features["long_notes"], row))
    if features["pedals"] >= 2 and features["long_notes"] >= 4:
        candidates["pedal_legato"].append((features["pedals"] + features["long_notes"], row))
    if features["weak_notes"] >= 12:
        candidates["weak_accompaniment"].append((features["weak_notes"], row))
    if features["onsets_per_second"] >= 5.0 and features["max_polyphony"] <= 4:
        candidates["fast_arpeggio"].append((features["onsets_per_second"], row))


def _group_by_start(notes: list[NoteEvent], tolerance: float) -> list[list[NoteEvent]]:
    groups: list[list[NoteEvent]] = []
    for note in sorted(notes, key=lambda item: (item.start, item.pitch)):
        if groups and note.start - groups[-1][0].start <= tolerance:
            groups[-1].append(note)
        else:
            groups.append([note])
    return groups


if __name__ == "__main__":
    main()
