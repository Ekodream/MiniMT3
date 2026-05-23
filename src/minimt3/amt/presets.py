from __future__ import annotations

from copy import deepcopy
from typing import Any


DECODE_PRESETS: dict[str, dict[str, Any]] = {
    "analysis_midi": {
        "onset_threshold": 0.36,
        "frame_threshold": 0.18,
        "offset_threshold": 0.22,
        "min_note_seconds": 0.04,
        "max_notes_per_second": 38.0,
        "max_polyphony": 12,
        "min_onset_gap_seconds": 0.03,
        "onset_peak_prominence": 0.008,
        "max_notes_per_start_window": 10,
        "start_window_seconds": 0.06,
        "disable_chord_recovery": True,
        "consume_note_energy": True,
        "energy_neighbor_pitches": 0,
        "energy_overlap_ratio": 0.45,
        "infer_onsets_from_frame_diff": True,
        "frame_diff_n": 2,
        "frame_diff_scale": 0.85,
        "score_profile": "analysis_midi",
    },
    "practice_score": {
        "onset_threshold": 0.46,
        "frame_threshold": 0.22,
        "offset_threshold": 0.26,
        "min_note_seconds": 0.055,
        "max_notes_per_second": 24.0,
        "max_polyphony": 10,
        "min_onset_gap_seconds": 0.035,
        "onset_peak_prominence": 0.014,
        "max_notes_per_start_window": 7,
        "start_window_seconds": 0.055,
        "disable_chord_recovery": False,
        "chord_onset_threshold": 0.54,
        "chord_frame_threshold": 0.26,
        "chord_window_frames": 2,
        "chord_score_ratio": 0.92,
        "consume_note_energy": True,
        "energy_neighbor_pitches": 0,
        "energy_overlap_ratio": 0.55,
        "infer_onsets_from_frame_diff": True,
        "frame_diff_n": 2,
        "frame_diff_scale": 0.70,
        "performance_min_note_seconds": 0.06,
        "performance_min_velocity": 5,
        "score_min_note_seconds": 0.125,
        "score_min_velocity": 8,
        "score_max_chord_notes": 10,
        "score_max_notes_per_beat": 5,
        "score_max_note_beats": 6.0,
        "score_chord_tolerance_seconds": 0.080,
        "score_chord_snap_seconds": 0.080,
        "score_chord_snap_max_spread_beats": 0.25,
        "score_profile": "practice_score",
    },
    "v13_recall": {
        "alias_of": "analysis_midi",
    },
    "v13_practice_score": {
        "alias_of": "practice_score",
    },
    "v14_practice_score": {
        "alias_of": "practice_score",
    },
}


def resolve_decode_preset(name: str | None) -> dict[str, Any]:
    if not name:
        return {}
    key = name.strip()
    if key.lower() in {"none", "config"}:
        return {}
    if key not in DECODE_PRESETS:
        known = ", ".join(sorted(DECODE_PRESETS))
        raise ValueError(f"Unknown decode preset '{name}'. Known presets: {known}")
    preset = deepcopy(DECODE_PRESETS[key])
    alias = preset.pop("alias_of", None)
    if alias:
        base = resolve_decode_preset(str(alias))
        base.update(preset)
        return base
    return preset


def apply_decode_preset(decode_cfg: dict[str, Any], name: str | None) -> dict[str, Any]:
    merged = dict(decode_cfg or {})
    merged.update(resolve_decode_preset(name))
    return merged
