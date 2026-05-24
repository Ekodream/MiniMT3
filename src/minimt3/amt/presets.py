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
        "score_chord_snap_seconds": 0.100,
        "score_chord_snap_max_spread_beats": 0.22,
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
    "v15_f1": {
        "onset_threshold": 0.42,
        "frame_threshold": 0.20,
        "offset_threshold": 0.24,
        "min_note_seconds": 0.045,
        "max_notes_per_second": 30.0,
        "max_polyphony": 12,
        "min_onset_gap_seconds": 0.03,
        "onset_peak_prominence": 0.010,
        "max_notes_per_start_window": 9,
        "start_window_seconds": 0.055,
        "disable_chord_recovery": False,
        "chord_onset_threshold": 0.50,
        "chord_frame_threshold": 0.24,
        "chord_window_frames": 2,
        "chord_score_ratio": 0.86,
        "consume_note_energy": True,
        "energy_neighbor_pitches": 0,
        "energy_overlap_ratio": 0.52,
        "infer_onsets_from_frame_diff": True,
        "frame_diff_n": 2,
        "frame_diff_scale": 0.78,
        "score_profile": "v15_f1",
    },
    "v15_rescue": {
        "onset_threshold": 0.56,
        "frame_threshold": 0.26,
        "offset_threshold": 0.28,
        "min_note_seconds": 0.055,
        "max_notes_per_second": 18.0,
        "max_polyphony": 10,
        "min_onset_gap_seconds": 0.035,
        "onset_peak_prominence": 0.020,
        "max_notes_per_start_window": 6,
        "start_window_seconds": 0.055,
        "disable_chord_recovery": False,
        "chord_onset_threshold": 0.64,
        "chord_frame_threshold": 0.30,
        "chord_window_frames": 1,
        "chord_score_ratio": 0.94,
        "consume_note_energy": True,
        "energy_neighbor_pitches": 0,
        "energy_overlap_ratio": 0.58,
        "infer_onsets_from_frame_diff": False,
        "frame_diff_n": 2,
        "frame_diff_scale": 0.50,
        "score_profile": "v15_rescue",
    },
}


SCORE_PRESETS: dict[str, dict[str, Any]] = {
    "performance_midi": {
        "score_profile": "performance_midi",
        "time_signature": "auto",
        "score_beat_divisions": "2,4",
        "score_voice_mode": "single",
        "score_split_ties": True,
        "score_hide_filler_rests": False,
        "score_min_note_seconds": 0.06,
        "score_min_note_beats": 0.125,
        "score_max_note_beats": 8.0,
        "score_max_notes_per_beat": 8,
        "score_max_short_rest_beats": 0.25,
    },
    "score_auto_safe": {
        "score_profile": "score_auto_safe",
        "time_signature": "4/4",
        "score_beat_divisions": "2,4",
        "score_voice_mode": "dual_staff_2voice",
        "score_split_ties": True,
        "score_hide_filler_rests": True,
        "score_min_note_seconds": 0.125,
        "score_min_note_beats": 0.25,
        "score_max_note_beats": 6.0,
        "score_max_notes_per_beat": 5,
        "score_chord_tolerance_seconds": 0.08,
        "score_chord_snap_seconds": 0.10,
        "score_chord_snap_max_spread_beats": 0.22,
        "score_max_short_rest_beats": 0.5,
    },
    "score_demo_4_4": {
        "alias_of": "score_auto_safe",
        "score_profile": "score_demo_4_4",
        "time_signature": "4/4",
        "tempo_bpm": 100.0,
        "score_beat_divisions": "2,4",
        "score_max_notes_per_beat": 5,
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


def resolve_score_preset(name: str | None) -> dict[str, Any]:
    if not name:
        return {}
    key = name.strip()
    if key.lower() in {"none", "config"}:
        return {}
    if key not in SCORE_PRESETS:
        known = ", ".join(sorted(SCORE_PRESETS))
        raise ValueError(f"Unknown score preset '{name}'. Known presets: {known}")
    preset = deepcopy(SCORE_PRESETS[key])
    alias = preset.pop("alias_of", None)
    if alias:
        base = resolve_score_preset(str(alias))
        base.update(preset)
        return base
    return preset


def apply_score_preset(score_cfg: dict[str, Any], name: str | None) -> dict[str, Any]:
    merged = dict(score_cfg or {})
    merged.update(resolve_score_preset(name))
    return merged
