from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import median

from minimt3.symbolic.events import NoteEvent, PedalEvent


MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
MAJOR_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MINOR_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]
PITCH_CLASS = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}
MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}


@dataclass
class ScorePolishConfig:
    key_signature: str | None = None
    time_signature: str = "auto"
    tempo_bpm: float | None = None
    beat_divisions: tuple[int, ...] = (2, 4)
    allow_tuplets: bool = False
    chord_tolerance_seconds: float = 0.055
    arpeggio_min_gap_seconds: float = 0.08
    arpeggio_max_gap_seconds: float = 0.25
    min_note_beats: float = 0.25
    max_note_beats: float = 4.0
    bass_max_note_beats: float = 8.0
    bass_protect_pitch: int = 48
    protect_bass_long_notes: bool = True
    protect_chord_tone_durations: bool = True
    same_pitch_margin_seconds: float = 0.02
    min_velocity: int = 6
    max_chord_notes: int = 10
    max_notes_per_beat: int = 8
    trim_score_overlaps: bool = True
    max_overlap_beats: float = 0.0
    prune_pedal_resonance: bool = True
    filter_key_outliers: bool = True
    non_key_max_velocity: int = 54
    non_key_max_note_beats: float = 0.75
    filter_isolated_notes: bool = True
    isolated_max_velocity: int = 48
    isolated_max_note_beats: float = 0.75
    isolation_window_beats: float = 1.0
    isolation_pitch_window: int = 12
    fill_short_rests: bool = True
    max_short_rest_beats: float = 0.5
    align_score_start: bool = True
    leading_rest_threshold_beats: float = 0.5
    start_offset_beats: float | None = None
    start_offset_seconds: float | None = None
    chord_snap_seconds: float = 0.075
    chord_snap_max_spread_beats: float = 0.25


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
    time_signature = infer_time_signature(base, tempo_bpm, cfg) if cfg.time_signature == "auto" else cfg.time_signature
    beat_divisions = _score_beat_divisions(cfg, time_signature)
    base, start_shift = _align_score_start(base, seconds_per_quarter, cfg)
    base, key_outlier_rate = _filter_key_outliers(base, key_signature, seconds_per_quarter, cfg)
    base, isolated_rate = _filter_isolated_notes(base, seconds_per_quarter, cfg)
    pruned, long_note_rate = _prune_long_notes(base, seconds_per_quarter, cfg)
    density_limited, density_pruned_rate = _limit_note_density(pruned, seconds_per_quarter, cfg)
    snapped, chord_snap_rate = _snap_near_chords(density_limited, seconds_per_quarter, cfg)
    quantized, quant_error, collapse_rate = _quantize_to_beat_grid(
        snapped,
        seconds_per_quarter,
        cfg,
        beat_divisions=beat_divisions,
    )
    right, left, hand_crossings = assign_hands_dp(quantized, cfg)
    right, right_fill_rate = _fill_short_rests(right, seconds_per_quarter, cfg)
    left, left_fill_rate = _fill_short_rests(left, seconds_per_quarter, cfg)
    right, right_overlap_trim_rate = _trim_hand_overlaps(right, seconds_per_quarter, cfg)
    left, left_overlap_trim_rate = _trim_hand_overlaps(left, seconds_per_quarter, cfg)
    score_notes = sorted(right + left, key=lambda n: (n.start, n.pitch, n.end))
    metrics = {
        "quantization_error_seconds": quant_error,
        "long_note_rate": long_note_rate,
        "density_pruned_rate": density_pruned_rate,
        "key_outlier_pruned_rate": key_outlier_rate,
        "isolated_pruned_rate": isolated_rate,
        "score_start_shift_seconds": start_shift,
        "chord_snap_rate": chord_snap_rate,
        "short_rest_fill_rate": (right_fill_rate + left_fill_rate) / 2.0,
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
        time_signature=time_signature,
        tempo_bpm=tempo_bpm,
        seconds_per_quarter=seconds_per_quarter,
        beat_divisions=beat_divisions,
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


def infer_time_signature(notes: list[NoteEvent], tempo_bpm: float, cfg: ScorePolishConfig | None = None) -> str:
    cfg = cfg or ScorePolishConfig()
    if not notes:
        return "4/4"
    spq = 60.0 / max(20.0, tempo_bpm)
    onsets = sorted({round(n.start, 4) for n in notes})
    if len(onsets) < 12:
        return "4/4"
    iois = [b - a for a, b in zip(onsets, onsets[1:]) if 0.04 <= b - a <= 1.5 * spq]
    if not iois:
        return "4/4"
    med_ioi = median(iois)
    eighth_like = 0.38 * spq <= med_ioi <= 0.68 * spq
    sixteenth_like = 0.18 * spq <= med_ioi < 0.38 * spq
    if eighth_like or sixteenth_like:
        compound_score = _compound_group_score(onsets, spq)
        if compound_score >= 0.52:
            return "12/8"
    return "4/4"


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


def _filter_key_outliers(
    notes: list[NoteEvent],
    key_signature: str,
    seconds_per_quarter: float,
    cfg: ScorePolishConfig,
) -> tuple[list[NoteEvent], float]:
    if not notes or not cfg.filter_key_outliers:
        return notes, 0.0
    scale = _scale_pitch_classes(key_signature)
    if scale is None:
        return notes, 0.0
    groups = _group_by_start(notes, tolerance=cfg.chord_tolerance_seconds)
    keep: list[NoteEvent] = []
    pruned = 0
    short_limit = max(0.03, cfg.non_key_max_note_beats * seconds_per_quarter)
    for group in groups:
        for note in group:
            in_key = int(note.pitch) % 12 in scale
            protected = len(group) >= 2 or note.velocity > cfg.non_key_max_velocity or note.end - note.start > short_limit
            if not in_key and not protected:
                pruned += 1
            else:
                keep.append(note)
    keep.sort(key=lambda n: (n.start, n.pitch, n.end))
    return keep, pruned / max(1, len(notes))


def _filter_isolated_notes(
    notes: list[NoteEvent],
    seconds_per_quarter: float,
    cfg: ScorePolishConfig,
) -> tuple[list[NoteEvent], float]:
    if not notes or not cfg.filter_isolated_notes:
        return notes, 0.0
    groups = _group_by_start(notes, tolerance=cfg.chord_tolerance_seconds)
    group_size = {id(note): len(group) for group in groups for note in group}
    time_window = max(0.05, cfg.isolation_window_beats * seconds_per_quarter)
    short_limit = max(0.03, cfg.isolated_max_note_beats * seconds_per_quarter)
    keep: list[NoteEvent] = []
    pruned = 0
    ordered = sorted(notes, key=lambda n: (n.start, n.pitch, n.end))
    for idx, note in enumerate(ordered):
        protected = (
            group_size.get(id(note), 1) >= 2
            or note.velocity > cfg.isolated_max_velocity
            or note.end - note.start > short_limit
        )
        if protected:
            keep.append(note)
            continue
        supported = False
        for other in ordered[max(0, idx - 10) : min(len(ordered), idx + 11)]:
            if other is note:
                continue
            if abs(other.start - note.start) <= time_window and abs(other.pitch - note.pitch) <= cfg.isolation_pitch_window:
                supported = True
                break
        if supported:
            keep.append(note)
        else:
            pruned += 1
    keep.sort(key=lambda n: (n.start, n.pitch, n.end))
    return keep, pruned / max(1, len(notes))


def _align_score_start(
    notes: list[NoteEvent],
    seconds_per_quarter: float,
    cfg: ScorePolishConfig,
) -> tuple[list[NoteEvent], float]:
    if not notes or not cfg.align_score_start:
        return notes, 0.0
    if cfg.start_offset_seconds is not None:
        shift = max(0.0, float(cfg.start_offset_seconds))
    elif cfg.start_offset_beats is not None:
        shift = max(0.0, float(cfg.start_offset_beats) * seconds_per_quarter)
    else:
        first_start = min(n.start for n in notes)
        threshold = max(0.0, cfg.leading_rest_threshold_beats) * seconds_per_quarter
        shift = first_start if first_start >= threshold else 0.0
    if shift <= 1e-6:
        return notes, 0.0
    shifted = [
        NoteEvent(n.pitch, max(0.0, n.start - shift), max(0.0, n.end - shift), n.velocity)
        for n in notes
        if n.end - shift > 0.0
    ]
    shifted = [n for n in shifted if n.end > n.start]
    shifted.sort(key=lambda n: (n.start, n.pitch, n.end))
    return shifted, shift


def _snap_near_chords(
    notes: list[NoteEvent],
    seconds_per_quarter: float,
    cfg: ScorePolishConfig,
) -> tuple[list[NoteEvent], float]:
    if not notes or cfg.chord_snap_seconds <= 0.0:
        return notes, 0.0
    max_spread = min(
        max(0.0, cfg.chord_snap_seconds),
        max(0.0, cfg.chord_snap_max_spread_beats) * seconds_per_quarter,
    )
    if max_spread <= 1e-6:
        return notes, 0.0
    ordered = sorted(notes, key=lambda n: (n.start, n.pitch, n.end))
    groups: list[list[NoteEvent]] = []
    current: list[NoteEvent] = []
    for item in ordered:
        if current and item.start - current[0].start > max_spread:
            groups.append(current)
            current = []
        current.append(item)
    if current:
        groups.append(current)

    snapped: list[NoteEvent] = []
    changed = 0
    for group in groups:
        if len(group) < 2:
            snapped.extend(group)
            continue
        group = sorted(group, key=lambda n: (n.start, n.pitch, n.end))
        span = group[-1].start - group[0].start
        pitch_moves = [b.pitch - a.pitch for a, b in zip(group, group[1:])]
        monotonic = all(move > 0 for move in pitch_moves) or all(move < 0 for move in pitch_moves)
        is_arpeggio = monotonic and span >= cfg.arpeggio_min_gap_seconds
        if is_arpeggio:
            snapped.extend(group)
            continue
        anchor = min(group, key=lambda n: (n.start, -n.velocity)).start
        for note in group:
            if abs(note.start - anchor) > 1e-6:
                changed += 1
            snapped.append(NoteEvent(note.pitch, anchor, max(anchor + 0.01, note.end), note.velocity))
    snapped.sort(key=lambda n: (n.start, n.pitch, n.end))
    return snapped, changed / max(1, len(notes))


def _prune_long_notes(
    notes: list[NoteEvent],
    seconds_per_quarter: float,
    cfg: ScorePolishConfig,
) -> tuple[list[NoteEvent], float]:
    if not notes:
        return [], 0.0
    max_duration = max(cfg.min_note_beats * seconds_per_quarter, cfg.max_note_beats * seconds_per_quarter)
    bass_max_duration = max(max_duration, cfg.bass_max_note_beats * seconds_per_quarter)
    chord_groups = _group_by_start(notes, tolerance=cfg.chord_tolerance_seconds)
    chord_size = {id(note): len(group) for group in chord_groups for note in group}
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
        note_max_duration = max_duration
        if cfg.protect_bass_long_notes and int(note.pitch) <= int(cfg.bass_protect_pitch):
            note_max_duration = bass_max_duration
        if cfg.protect_chord_tone_durations and chord_size.get(id(note), 1) >= 2:
            note_max_duration = max(note_max_duration, max_duration * 1.5)
        if cfg.prune_pedal_resonance and end - note.start > note_max_duration:
            long_count += 1
            end = note.start + note_max_duration
        if end > note.start:
            pruned.append(NoteEvent(note.pitch, note.start, end, note.velocity))
    pruned.sort(key=lambda n: (n.start, n.pitch, n.end))
    return pruned, long_count / max(1, len(notes))


def _quantize_to_beat_grid(
    notes: list[NoteEvent],
    seconds_per_quarter: float,
    cfg: ScorePolishConfig,
    beat_divisions: tuple[int, ...] | None = None,
) -> tuple[list[NoteEvent], float, float]:
    if not notes:
        return [], 0.0, 0.0
    divisions = tuple(sorted({max(1, int(d)) for d in (beat_divisions or cfg.beat_divisions)}))
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


def _fill_short_rests(
    notes: list[NoteEvent],
    seconds_per_quarter: float,
    cfg: ScorePolishConfig,
) -> tuple[list[NoteEvent], float]:
    if not notes or not cfg.fill_short_rests:
        return notes, 0.0
    groups = _group_by_start(notes, tolerance=1e-5)
    changed = 0
    max_gap = max(0.0, cfg.max_short_rest_beats) * seconds_per_quarter
    filled: list[NoteEvent] = []
    for idx, group in enumerate(groups):
        next_start = groups[idx + 1][0].start if idx + 1 < len(groups) else None
        group_end = max(n.end for n in group)
        target_end = group_end
        if next_start is not None:
            gap = next_start - group_end
            if 0.0 < gap <= max_gap:
                target_end = next_start
        for note in group:
            end = note.end
            if target_end > end and target_end - note.start <= cfg.max_note_beats * seconds_per_quarter * 1.25:
                end = target_end
                changed += 1
            filled.append(NoteEvent(note.pitch, note.start, end, note.velocity))
    filled.sort(key=lambda n: (n.start, n.pitch, n.end))
    return filled, changed / max(1, len(notes))


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


def _score_beat_divisions(cfg: ScorePolishConfig, time_signature: str) -> tuple[int, ...]:
    divisions = tuple(sorted({max(1, int(d)) for d in cfg.beat_divisions})) or (2, 4)
    if not cfg.allow_tuplets:
        divisions = tuple(d for d in divisions if d % 3 != 0)
    if not divisions:
        divisions = (2, 4)
    if str(time_signature).strip() in {"6/8", "9/8", "12/8"} and not cfg.allow_tuplets:
        divisions = tuple(sorted(set(divisions + (2, 4))))
    return divisions


def _compound_group_score(onsets: list[float], seconds_per_quarter: float) -> float:
    eighth = seconds_per_quarter / 2.0
    if eighth <= 0:
        return 0.0
    hits = 0
    total = 0
    for onset in onsets:
        eighth_index = int(round(onset / eighth))
        err = abs(onset - eighth_index * eighth)
        if err <= 0.18 * eighth:
            total += 1
            if eighth_index % 3 in {0, 1, 2}:
                hits += 1
    if total == 0:
        return 0.0
    dense = min(1.0, total / max(1.0, (onsets[-1] - onsets[0]) / eighth * 0.35))
    return dense * hits / total


def _scale_pitch_classes(key_signature: str) -> set[int] | None:
    parts = str(key_signature).strip().replace("-", "b").split()
    if not parts:
        return None
    mode = parts[-1].lower() if parts[-1].lower() in {"major", "minor"} else "major"
    tonic = " ".join(parts[:-1]) if mode in {"major", "minor"} and len(parts) > 1 else parts[0]
    tonic = tonic.replace("♭", "b").replace("♯", "#")
    pc = PITCH_CLASS.get(tonic)
    if pc is None:
        return None
    scale = MINOR_SCALE if mode == "minor" else MAJOR_SCALE
    return {(pc + degree) % 12 for degree in scale}


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
