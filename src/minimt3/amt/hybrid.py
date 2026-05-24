from __future__ import annotations

from dataclasses import asdict, dataclass

from minimt3.symbolic.events import NoteEvent


@dataclass(frozen=True)
class HybridRescueConfig:
    enabled: bool = False
    mode: str = "chord_long"
    chord_window_seconds: float = 0.085
    long_window_seconds: float = 0.14
    duplicate_window_seconds: float = 0.065
    duplicate_overlap_ratio: float = 0.35
    min_velocity: int = 50
    min_duration_seconds: float = 0.08
    long_min_duration_seconds: float = 0.65
    bass_pitch_max: int = 55
    max_added_ratio: float = 0.05
    max_added_per_second: float = 0.55
    max_added_per_base_onset: int = 2
    max_total_notes_per_second: float = 28.0

    def to_json(self) -> dict[str, float | int | bool | str]:
        return asdict(self)


def hybrid_rescue_notes(
    base_notes: list[NoteEvent],
    assistant_notes: list[NoteEvent],
    duration: float,
    config: HybridRescueConfig | None = None,
) -> tuple[list[NoteEvent], dict[str, float]]:
    cfg = config or HybridRescueConfig()
    if not cfg.enabled or not assistant_notes:
        return list(base_notes), _stats(0, len(assistant_notes), len(base_notes), len(base_notes))
    base = sorted(base_notes, key=lambda n: (n.start, n.pitch, n.end))
    candidates = sorted(assistant_notes, key=lambda n: (-n.velocity, -(n.end - n.start), n.start, n.pitch))
    max_added_by_ratio = int(max(0.0, cfg.max_added_ratio) * max(1, len(base)))
    max_added_by_time = int(max(0.0, cfg.max_added_per_second) * max(0.1, duration))
    max_added = max(0, min(max_added_by_ratio, max_added_by_time))
    max_total_notes = int(max(1.0, cfg.max_total_notes_per_second) * max(0.1, duration))
    if max_added <= 0 or len(base) >= max_total_notes:
        return base, _stats(0, len(assistant_notes), len(base_notes), len(base))

    base_groups = _group_by_start(base, cfg.chord_window_seconds)
    added: list[NoteEvent] = []
    added_by_group: dict[int, int] = {}
    rejected_duplicate = 0
    rejected_context = 0
    rejected_budget = 0
    for cand in candidates:
        if len(added) >= max_added or len(base) + len(added) >= max_total_notes:
            rejected_budget += 1
            continue
        if cand.velocity < cfg.min_velocity or cand.end - cand.start < cfg.min_duration_seconds:
            rejected_context += 1
            continue
        if _has_duplicate(cand, base, cfg) or _has_duplicate(cand, added, cfg):
            rejected_duplicate += 1
            continue
        group_idx, group = _nearest_group(cand, base_groups, cfg.chord_window_seconds)
        accept = False
        if group is not None:
            group_size = len(group)
            near_chord = group_size >= 2 or _polyphony_at(base, cand.start, cfg.chord_window_seconds) >= 2
            pitch_new_to_group = all(abs(cand.pitch - item.pitch) >= 2 for item in group)
            if near_chord and pitch_new_to_group:
                accept = True
            long_low = (
                cand.pitch <= cfg.bass_pitch_max
                and cand.end - cand.start >= cfg.long_min_duration_seconds
                and abs(cand.start - group[0].start) <= cfg.long_window_seconds
            )
            if long_low:
                accept = True
        else:
            long_low = cand.pitch <= cfg.bass_pitch_max and cand.end - cand.start >= cfg.long_min_duration_seconds * 1.5
            support = _polyphony_at(base, cand.start, cfg.long_window_seconds) >= 1
            accept = bool(long_low and support)
            group_idx = -1

        if not accept:
            rejected_context += 1
            continue
        if group_idx >= 0 and added_by_group.get(group_idx, 0) >= cfg.max_added_per_base_onset:
            rejected_budget += 1
            continue
        added.append(NoteEvent(cand.pitch, cand.start, cand.end, cand.velocity))
        if group_idx >= 0:
            added_by_group[group_idx] = added_by_group.get(group_idx, 0) + 1

    merged = sorted(base + added, key=lambda n: (n.start, n.pitch, n.end))
    stats = _stats(len(added), len(assistant_notes), len(base_notes), len(merged))
    stats.update(
        {
            "hybrid_rejected_duplicate": float(rejected_duplicate),
            "hybrid_rejected_context": float(rejected_context),
            "hybrid_rejected_budget": float(rejected_budget),
            "hybrid_added_ratio": len(added) / max(1.0, float(len(base_notes))),
        }
    )
    return merged, stats


def _stats(added: int, assistant: int, before: int, after: int) -> dict[str, float]:
    return {
        "hybrid_added_notes": float(added),
        "hybrid_assistant_candidates": float(assistant),
        "hybrid_base_notes": float(before),
        "hybrid_output_notes": float(after),
    }


def _group_by_start(notes: list[NoteEvent], tolerance: float) -> list[list[NoteEvent]]:
    groups: list[list[NoteEvent]] = []
    for note in sorted(notes, key=lambda n: (n.start, n.pitch, n.end)):
        if groups and abs(note.start - groups[-1][0].start) <= tolerance:
            groups[-1].append(note)
        else:
            groups.append([note])
    return groups


def _nearest_group(
    note: NoteEvent,
    groups: list[list[NoteEvent]],
    tolerance: float,
) -> tuple[int, list[NoteEvent] | None]:
    best_idx = -1
    best_group = None
    best_error = tolerance + 1e-9
    for idx, group in enumerate(groups):
        error = abs(note.start - group[0].start)
        if error <= tolerance and error < best_error:
            best_idx = idx
            best_group = group
            best_error = error
    return best_idx, best_group


def _has_duplicate(note: NoteEvent, notes: list[NoteEvent], cfg: HybridRescueConfig) -> bool:
    for other in notes:
        if note.pitch != other.pitch:
            continue
        if abs(note.start - other.start) <= cfg.duplicate_window_seconds:
            return True
        if _overlap_ratio(note, other) >= cfg.duplicate_overlap_ratio:
            return True
    return False


def _overlap_ratio(a: NoteEvent, b: NoteEvent) -> float:
    overlap = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    denom = max(1e-6, min(a.end - a.start, b.end - b.start))
    return overlap / denom


def _polyphony_at(notes: list[NoteEvent], start: float, tolerance: float) -> int:
    return sum(1 for note in notes if abs(note.start - start) <= tolerance)
