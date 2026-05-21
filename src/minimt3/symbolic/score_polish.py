from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import median

from minimt3.symbolic.events import NoteEvent, PedalEvent


MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
MAJOR_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MINOR_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]


@dataclass
class ScorePolishConfig:
    key_signature: str | None = None
    time_signature: str = "4/4"
    tempo_bpm: float | None = None
    beat_divisions: tuple[int, ...] = (2, 3, 4)
    chord_tolerance_seconds: float = 0.055
    arpeggio_min_gap_seconds: float = 0.08
    arpeggio_max_gap_seconds: float = 0.25
    min_note_beats: float = 0.25
    max_note_beats: float = 4.0
    same_pitch_margin_seconds: float = 0.02
    min_velocity: int = 6
    max_chord_notes: int = 10
    max_notes_per_beat: int = 8
    trim_score_overlaps: bool = True
    max_overlap_beats: float = 0.0
    prune_pedal_resonance: bool = True


@dataclass
class ScorePolishResult:
    notes: list[NoteEvent]
    right_notes: list[NoteEvent]
    left_notes: list[NoteEvent]
    key_signature: str
    time_signature: str
    tempo_bpm: float
    seconds_per_quarter: float
    beat_divisions: tuple[int, ...]
    metrics: dict[str, float]

    def to_json(self) -> dict:
        return {
            "notes": [asdict(n) for n in self.notes],
            "right_notes": len(self.right_notes),
            "left_notes": len(self.left_notes),
            "key_signature": self.key_signature,
            "time_signature": self.time_signature,
            "tempo_bpm": self.tempo_bpm,
            "seconds_per_quarter": self.seconds_per_quarter,
            "beat_divisions": list(self.beat_divisions),
            "metrics": self.metrics,
        }


def polish_score_notes(
    notes: list[NoteEvent],
    pedals: list[PedalEvent] | None = None,
    config: ScorePolishConfig | None = None,
) -> ScorePolishResult:
    cfg = config or ScorePolishConfig()
    pedals = pedals or []
    base = _filter_score_notes(notes, cfg)
    key_signature = cfg.key_signature or estimate_key_signature(base)
    tempo_bpm = float(cfg.tempo_bpm or estimate_tempo_bpm(base))
    seconds_per_quarter = 60.0 / max(20.0, tempo_bpm)
    pruned, long_note_rate = _prune_long_notes(base, seconds_per_quarter, cfg)
    density_limited, density_pruned_rate = _limit_note_density(pruned, seconds_per_quarter, cfg)
    quantized, quant_error, collapse_rate = _quantize_to_beat_grid(density_limited, seconds_per_quarter, cfg)
    right, left, hand_crossings = assign_hands_dp(quantized, cfg)
    right, right_overlap_trim_rate = _trim_hand_overlaps(right, seconds_per_quarter, cfg)
    left, left_overlap_trim_rate = _trim_hand_overlaps(left, seconds_per_quarter, cfg)
    score_notes = sorted(right + left, key=lambda n: (n.start, n.pitch, n.end))
    metrics = {
        "quantization_error_seconds": quant_error,
        "long_note_rate": long_note_rate,
        "density_pruned_rate": density_pruned_rate,
        "overlap_trim_rate": (right_overlap_trim_rate + left_overlap_trim_rate) / 2.0,
        "chord_collapse_rate": collapse_rate,
        "hand_crossings": float(hand_crossings),
        "score_notes": float(len(score_notes)),
        "pedal_regions": float(len(pedals)),
    }
    return ScorePolishResult(
        notes=score_notes,
        right_notes=right,
        left_notes=left,
        key_signature=key_signature,
        time_signature=cfg.time_signature,
        tempo_bpm=tempo_bpm,
        seconds_per_quarter=seconds_per_quarter,
        beat_divisions=cfg.beat_divisions,
        metrics=metrics,
    )


def estimate_key_signature(notes: list[NoteEvent]) -> str:
    if not notes:
        return "C major"
    hist = [0.0] * 12
    for note in notes:
        weight = max(0.05, note.end - note.start) * max(1, note.velocity)
        hist[int(note.pitch) % 12] += weight
    best_name = "C major"
    best_score = -math.inf
    for tonic in range(12):
        major = _profile_score(hist, MAJOR_PROFILE, tonic)
        minor = _profile_score(hist, MINOR_PROFILE, tonic)
        if major > best_score:
            best_score = major
            best_name = f"{MAJOR_NAMES[tonic]} major"
        if minor > best_score:
            best_score = minor
            best_name = f"{MINOR_NAMES[tonic]} minor"
    return best_name


def estimate_tempo_bpm(notes: list[NoteEvent], min_bpm: int = 50, max_bpm: int = 180) -> float:
    onsets = sorted({round(n.start, 4) for n in notes})
    if len(onsets) < 4:
        return 120.0
    sample = onsets[: min(len(onsets), 400)]
    best_bpm = 120
    best_score = math.inf
    for bpm in range(min_bpm, max_bpm + 1):
        spq = 60.0 / bpm
        step = spq / 4.0
        phases = [0.0]
        phases.extend((t % step) for t in sample[:16])
        phases.append(median(t % step for t in sample))
        score = min(_grid_error(sample, step, phase) for phase in phases)
        score += 0.0004 * abs(bpm - 100)
        if score < best_score:
            best_score = score
            best_bpm = bpm
    return float(best_bpm)


def assign_hands_dp(
    notes: list[NoteEvent],
    config: ScorePolishConfig | None = None,
) -> tuple[list[NoteEvent], list[NoteEvent], int]:
    cfg = config or ScorePolishConfig()
    groups = _group_by_start(notes, tolerance=1e-5)
    if not groups:
        return [], [], 0
    candidates = [_hand_candidates(group) for group in groups]
    costs: list[list[float]] = []
    backs: list[list[int]] = []
    costs.append([cand["local_cost"] for cand in candidates[0]])
    backs.append([-1] * len(candidates[0]))
    for idx in range(1, len(groups)):
        row_costs: list[float] = []
        row_backs: list[int] = []
        for cand in candidates[idx]:
            best_cost = math.inf
            best_prev = 0
            for prev_idx, prev in enumerate(candidates[idx - 1]):
                total = costs[idx - 1][prev_idx] + cand["local_cost"] + _hand_transition_cost(prev, cand)
                if total < best_cost:
                    best_cost = total
                    best_prev = prev_idx
            row_costs.append(best_cost)
            row_backs.append(best_prev)
        costs.append(row_costs)
        backs.append(row_backs)

    choice = min(range(len(costs[-1])), key=lambda i: costs[-1][i])
    selected = [0] * len(groups)
    for idx in range(len(groups) - 1, -1, -1):
        selected[idx] = choice
        choice = backs[idx][choice]

    right: list[NoteEvent] = []
    left: list[NoteEvent] = []
    crossings = 0
    for group, group_candidates, cand_idx in zip(groups, candidates, selected):
        cand = group_candidates[cand_idx]
        left_notes = [group[i] for i in cand["left_indices"]]
        right_notes = [group[i] for i in cand["right_indices"]]
        if left_notes and right_notes and max(n.pitch for n in left_notes) > min(n.pitch for n in right_notes):
            crossings += 1
        left.extend(left_notes)
        right.extend(right_notes)
    right.sort(key=lambda n: (n.start, n.pitch, n.end))
    left.sort(key=lambda n: (n.start, n.pitch, n.end))
    return right, left, crossings


def _filter_score_notes(notes: list[NoteEvent], cfg: ScorePolishConfig) -> list[NoteEvent]:
    clean = [
        NoteEvent(int(n.pitch), max(0.0, n.start), max(n.start, n.end), max(1, min(127, int(n.velocity))))
        for n in notes
        if n.end > n.start and int(n.velocity) >= cfg.min_velocity
    ]
    clean.sort(key=lambda n: (n.start, n.pitch, n.end))
    groups = _group_by_start(clean, tolerance=cfg.chord_tolerance_seconds)
    limited: list[NoteEvent] = []
    for group in groups:
        limited.extend(sorted(group, key=lambda n: (n.velocity, n.end - n.start), reverse=True)[: cfg.max_chord_notes])
    limited.sort(key=lambda n: (n.start, n.pitch, n.end))
    return limited


def _prune_long_notes(
    notes: list[NoteEvent],
    seconds_per_quarter: float,
    cfg: ScorePolishConfig,
) -> tuple[list[NoteEvent], float]:
    if not notes:
        return [], 0.0
    max_duration = max(cfg.min_note_beats * seconds_per_quarter, cfg.max_note_beats * seconds_per_quarter)
    by_pitch: dict[int, list[NoteEvent]] = {}
    for note in notes:
        by_pitch.setdefault(note.pitch, []).append(note)
    next_start: dict[int, float | None] = {}
    for pitch_notes in by_pitch.values():
        pitch_notes.sort(key=lambda n: (n.start, n.end))
        for idx, note in enumerate(pitch_notes):
            next_start[id(note)] = pitch_notes[idx + 1].start if idx + 1 < len(pitch_notes) else None

    pruned: list[NoteEvent] = []
    long_count = 0
    for note in notes:
        end = note.end
        next_pitch_start = next_start.get(id(note))
        if next_pitch_start is not None:
            end = min(end, max(note.start + 0.01, next_pitch_start - cfg.same_pitch_margin_seconds))
        if cfg.prune_pedal_resonance and end - note.start > max_duration:
            long_count += 1
            end = note.start + max_duration
        if end > note.start:
            pruned.append(NoteEvent(note.pitch, note.start, end, note.velocity))
    pruned.sort(key=lambda n: (n.start, n.pitch, n.end))
    return pruned, long_count / max(1, len(notes))


def _quantize_to_beat_grid(
    notes: list[NoteEvent],
    seconds_per_quarter: float,
    cfg: ScorePolishConfig,
) -> tuple[list[NoteEvent], float, float]:
    if not notes:
        return [], 0.0, 0.0
    divisions = tuple(sorted({max(1, int(d)) for d in cfg.beat_divisions}))
    min_step = seconds_per_quarter / max(divisions)
    min_duration = max(0.03, cfg.min_note_beats * seconds_per_quarter)
    quantized: list[NoteEvent] = []
    total_error = 0.0
    collapse_risk = 0
    prev_orig: NoteEvent | None = None
    prev_q_start: float | None = None
    for note in sorted(notes, key=lambda n: (n.start, n.pitch, n.end)):
        q_start = _quantize_seconds(note.start, seconds_per_quarter, divisions)
        q_end = _quantize_seconds(note.end, seconds_per_quarter, divisions)
        if prev_orig is not None and prev_q_start is not None:
            gap = note.start - prev_orig.start
            pitch_move = note.pitch - prev_orig.pitch
            is_arpeggio_gap = cfg.arpeggio_min_gap_seconds <= gap <= cfg.arpeggio_max_gap_seconds
            if is_arpeggio_gap and pitch_move != 0 and q_start <= prev_q_start:
                collapse_risk += 1
                q_start = prev_q_start + min_step
        if q_end <= q_start:
            q_end = q_start + min_duration
        elif q_end - q_start < min_duration:
            q_end = q_start + min_duration
        total_error += abs(q_start - note.start) + 0.5 * abs(q_end - note.end)
        quantized.append(NoteEvent(note.pitch, q_start, q_end, note.velocity))
        prev_orig = note
        prev_q_start = q_start

    deduped: dict[tuple[int, float], NoteEvent] = {}
    for note in quantized:
        key = (note.pitch, round(note.start, 6))
        if key not in deduped or note.end - note.start > deduped[key].end - deduped[key].start:
            deduped[key] = note
    out = list(deduped.values())
    out.sort(key=lambda n: (n.start, n.pitch, n.end))
    return out, total_error / max(1, len(notes)), collapse_risk / max(1, len(notes))


def _limit_note_density(
    notes: list[NoteEvent],
    seconds_per_quarter: float,
    cfg: ScorePolishConfig,
) -> tuple[list[NoteEvent], float]:
    if not notes or cfg.max_notes_per_beat <= 0:
        return notes, 0.0
    by_beat: dict[int, list[NoteEvent]] = {}
    for note in notes:
        beat = int(note.start / max(1e-6, seconds_per_quarter))
        by_beat.setdefault(beat, []).append(note)
    kept: list[NoteEvent] = []
    pruned = 0
    for beat_notes in by_beat.values():
        if len(beat_notes) <= cfg.max_notes_per_beat:
            kept.extend(beat_notes)
            continue
        ranked = sorted(beat_notes, key=lambda n: (n.velocity, n.end - n.start), reverse=True)
        keep = ranked[: cfg.max_notes_per_beat]
        kept.extend(keep)
        pruned += len(beat_notes) - len(keep)
    kept.sort(key=lambda n: (n.start, n.pitch, n.end))
    return kept, pruned / max(1, len(notes))


def _trim_hand_overlaps(
    notes: list[NoteEvent],
    seconds_per_quarter: float,
    cfg: ScorePolishConfig,
) -> tuple[list[NoteEvent], float]:
    if not notes or not cfg.trim_score_overlaps:
        return notes, 0.0
    groups = _group_by_start(notes, tolerance=1e-5)
    trimmed: list[NoteEvent] = []
    changed = 0
    min_duration = max(0.03, cfg.min_note_beats * seconds_per_quarter)
    overlap_allowance = max(0.0, cfg.max_overlap_beats) * seconds_per_quarter
    for idx, group in enumerate(groups):
        next_start = groups[idx + 1][0].start if idx + 1 < len(groups) else None
        for note in group:
            end = note.end
            if next_start is not None:
                cap = max(note.start + min_duration, next_start + overlap_allowance)
                if end > cap:
                    end = cap
                    changed += 1
            if end <= note.start:
                end = note.start + min_duration
            trimmed.append(NoteEvent(note.pitch, note.start, end, note.velocity))
    trimmed.sort(key=lambda n: (n.start, n.pitch, n.end))
    return trimmed, changed / max(1, len(notes))


def _quantize_seconds(seconds: float, seconds_per_quarter: float, divisions: tuple[int, ...]) -> float:
    best = 0.0
    best_error = math.inf
    for div in divisions:
        step = seconds_per_quarter / div
        candidate = round(seconds / step) * step
        error = abs(candidate - seconds) + 0.0005 * div
        if error < best_error:
            best_error = error
            best = candidate
    return max(0.0, best)


def _group_by_start(notes: list[NoteEvent], tolerance: float) -> list[list[NoteEvent]]:
    groups: list[list[NoteEvent]] = []
    for note in sorted(notes, key=lambda n: (n.start, n.pitch, n.end)):
        if groups and abs(note.start - groups[-1][0].start) <= tolerance:
            groups[-1].append(note)
        else:
            groups.append([note])
    for group in groups:
        group.sort(key=lambda n: n.pitch)
    return groups


def _hand_candidates(group: list[NoteEvent]) -> list[dict[str, object]]:
    ordered = sorted(range(len(group)), key=lambda i: group[i].pitch)
    candidates: list[dict[str, object]] = []
    for split in range(len(ordered) + 1):
        left_indices = ordered[:split]
        right_indices = ordered[split:]
        left_notes = [group[i] for i in left_indices]
        right_notes = [group[i] for i in right_indices]
        local = 0.0
        for note in left_notes:
            local += max(0, note.pitch - 64) * 0.8 + max(0, 38 - note.pitch) * 0.2
        for note in right_notes:
            local += max(0, 55 - note.pitch) * 0.8 + max(0, note.pitch - 96) * 0.2
        if left_notes and right_notes and max(n.pitch for n in left_notes) > min(n.pitch for n in right_notes):
            local += 50.0
        if not left_notes:
            local += max(0.0, 64.0 - _mean_pitch(right_notes)) * 0.25
        if not right_notes:
            local += max(0.0, _mean_pitch(left_notes) - 55.0) * 0.25
        candidates.append(
            {
                "left_indices": left_indices,
                "right_indices": right_indices,
                "left_center": _mean_pitch(left_notes) if left_notes else None,
                "right_center": _mean_pitch(right_notes) if right_notes else None,
                "local_cost": local,
            }
        )
    return candidates


def _hand_transition_cost(prev: dict[str, object], curr: dict[str, object]) -> float:
    cost = 0.0
    for hand in ("left", "right"):
        prev_center = prev[f"{hand}_center"]
        curr_center = curr[f"{hand}_center"]
        if prev_center is not None and curr_center is not None:
            cost += abs(float(curr_center) - float(prev_center)) * 0.12
        elif prev_center is not None or curr_center is not None:
            cost += 1.5
    return cost


def _mean_pitch(notes: list[NoteEvent]) -> float:
    if not notes:
        return 60.0
    return sum(n.pitch for n in notes) / len(notes)


def _profile_score(hist: list[float], profile: list[float], tonic: int) -> float:
    rotated = [profile[(pc - tonic) % 12] for pc in range(12)]
    return _cosine(hist, rotated)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a <= 1e-9 or norm_b <= 1e-9:
        return 0.0
    return dot / (norm_a * norm_b)


def _grid_error(values: list[float], step: float, phase: float) -> float:
    errors = []
    for value in values:
        q = phase + round((value - phase) / step) * step
        errors.append(abs(value - q))
    return sum(errors) / max(1, len(errors))
