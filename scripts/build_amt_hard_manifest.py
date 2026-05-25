#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict

from minimt3.symbolic.events import NoteEvent, PedalEvent, load_midi_events
from minimt3.utils import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a train-only hard-mix manifest for dense AMT fine-tuning.")
    parser.add_argument("--index", default="data/cache/maestro_index.json")
    parser.add_argument("--base_manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--clip_seconds", type=float, default=8.0)
    parser.add_argument("--stride_seconds", type=float, default=4.0)
    parser.add_argument("--max_hard_clips", type=int, default=48000)
    parser.add_argument("--max_per_piece", type=int, default=80)
    parser.add_argument("--max_per_category", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=184)
    args = parser.parse_args()

    base = [
        row
        for row in read_json(args.base_manifest)
        if row.get("split") == args.split
        and row.get("audio_exists", True)
        and row.get("midi_exists", True)
    ]
    seen = {(row["audio"], round(float(row["start_sec"]), 3), round(float(row["end_sec"]), 3)) for row in base}
    index_rows = [
        row
        for row in read_json(args.index)
        if row.get("split") == args.split
        and row.get("audio_exists", True)
        and row.get("midi_exists", True)
        and float(row.get("duration") or 0.0) > 0
    ]

    candidates: dict[str, list[tuple[float, dict]]] = defaultdict(list)
    for piece_idx, row in enumerate(index_rows):
        duration = float(row.get("duration") or 0.0)
        if duration <= 0.0:
            continue
        notes, pedals = load_midi_events(row["midi"], start=0.0, end=duration)
        notes_by_start = sorted(notes, key=lambda item: item.start)
        pedals_by_start = sorted(pedals, key=lambda item: item.start)
        active_notes: list[NoteEvent] = []
        active_pedals: list[PedalEvent] = []
        note_pos = 0
        pedal_pos = 0
        max_start = max(0.0, duration - args.clip_seconds)
        start = 0.0
        clip_idx = 0
        while start <= max_start + 1e-6:
            end = min(duration, start + args.clip_seconds)
            while note_pos < len(notes_by_start) and notes_by_start[note_pos].start < end:
                active_notes.append(notes_by_start[note_pos])
                note_pos += 1
            while pedal_pos < len(pedals_by_start) and pedals_by_start[pedal_pos].start < end:
                active_pedals.append(pedals_by_start[pedal_pos])
                pedal_pos += 1
            active_notes = [note for note in active_notes if note.end > start]
            active_pedals = [pedal for pedal in active_pedals if pedal.end > start]
            key = (row["audio"], round(start, 3), round(end, 3))
            if key not in seen:
                clip_notes = _clip_notes(active_notes, start, end)
                clip_pedals = _clip_pedals(active_pedals, start, end)
                features = _clip_features(clip_notes, clip_pedals, end - start)
                base_row = {
                    "clip_id": f"{args.split}_hard_{piece_idx:04d}_{clip_idx:04d}",
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
                    "hard_features": features,
                }
                _add_candidates(candidates, base_row, features)
            clip_idx += 1
            start += args.stride_seconds

    for category in list(candidates):
        candidates[category].sort(key=lambda item: (-item[0], str(item[1]["clip_id"])))

    selected = []
    selected_by_piece: dict[str, int] = defaultdict(int)
    category_order = [
        "dense_chords",
        "low_long_pedal",
        "mid_long_notes",
        "weak_notes",
        "very_short_notes",
        "fast_arpeggio",
    ]
    while len(selected) < args.max_hard_clips:
        added = False
        for category in category_order:
            if sum(1 for row in selected if row["hard_category"] == category) >= args.max_per_category:
                continue
            pool = candidates.get(category, [])
            while pool:
                _, row = pool.pop(0)
                piece_key = str(row["audio"])
                key = (row["audio"], round(float(row["start_sec"]), 3), round(float(row["end_sec"]), 3))
                if key in seen or selected_by_piece[piece_key] >= args.max_per_piece:
                    continue
                row = dict(row)
                row["hard_category"] = category
                selected.append(row)
                seen.add(key)
                selected_by_piece[piece_key] += 1
                added = True
                break
            if len(selected) >= args.max_hard_clips:
                break
        if not added:
            break

    mixed = base + selected
    write_json(args.out, mixed)
    print(f"wrote {len(mixed)} clips to {args.out} base={len(base)} hard={len(selected)}")


def _clip_notes(notes: list[NoteEvent], start: float, end: float) -> list[NoteEvent]:
    out = []
    for note in notes:
        if note.start >= end or note.end <= start:
            continue
        out.append(NoteEvent(note.pitch, max(start, note.start) - start, min(end, note.end) - start, note.velocity))
    return out


def _clip_pedals(pedals: list[PedalEvent], start: float, end: float) -> list[PedalEvent]:
    out = []
    for pedal in pedals:
        if pedal.start >= end or pedal.end <= start:
            continue
        out.append(PedalEvent(max(start, pedal.start) - start, min(end, pedal.end) - start))
    return out


def _clip_features(notes: list[NoteEvent], pedals: list[PedalEvent], duration: float) -> dict[str, float]:
    if not notes:
        return {
            "notes": 0.0,
            "onsets_per_second": 0.0,
            "max_polyphony": 0.0,
            "chord_notes": 0.0,
            "long_notes": 0.0,
            "low_long_notes": 0.0,
            "weak_notes": 0.0,
            "very_short_notes": 0.0,
            "pedal_seconds": sum(max(0.0, p.end - p.start) for p in pedals),
        }
    groups = _group_by_start(notes, tolerance=0.055)
    long_notes = sum(1 for note in notes if note.end - note.start >= 1.0)
    low_long_notes = sum(1 for note in notes if note.pitch <= 52 and note.end - note.start >= 1.0)
    return {
        "notes": float(len(notes)),
        "onsets_per_second": len(groups) / max(1e-6, duration),
        "max_polyphony": float(max(len(group) for group in groups)),
        "chord_notes": float(sum(len(group) for group in groups if len(group) >= 3)),
        "long_notes": float(long_notes),
        "low_long_notes": float(low_long_notes),
        "weak_notes": float(sum(1 for note in notes if note.velocity <= 45)),
        "very_short_notes": float(sum(1 for note in notes if note.end - note.start <= 0.125)),
        "pedal_seconds": sum(max(0.0, p.end - p.start) for p in pedals),
    }


def _add_candidates(candidates: dict[str, list[tuple[float, dict]]], row: dict, features: dict[str, float]) -> None:
    if features["chord_notes"] >= 10 or features["max_polyphony"] >= 5:
        candidates["dense_chords"].append((features["chord_notes"] + 2.0 * features["max_polyphony"], row))
    if features["low_long_notes"] >= 1 and features["pedal_seconds"] >= 1.0:
        candidates["low_long_pedal"].append((3.0 * features["low_long_notes"] + features["pedal_seconds"], row))
    if features["long_notes"] >= 3:
        candidates["mid_long_notes"].append((features["long_notes"] + 0.2 * features["notes"], row))
    if features["weak_notes"] >= 8:
        candidates["weak_notes"].append((features["weak_notes"], row))
    if features["very_short_notes"] >= 12:
        candidates["very_short_notes"].append((features["very_short_notes"], row))
    if features["onsets_per_second"] >= 4.5 and features["max_polyphony"] <= 4:
        candidates["fast_arpeggio"].append((features["onsets_per_second"] + 0.05 * features["notes"], row))


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
