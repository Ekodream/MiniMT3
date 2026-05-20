#!/usr/bin/env python
from __future__ import annotations

import argparse

import torch

from minimt3.amt.data import DenseAMTDataset
from minimt3.amt.decode import decode_dense_notes
from minimt3.amt.model import DenseAMT, DenseAMTConfig
from minimt3.amt.targets import DenseTargetConfig
from minimt3.audio.features import LogMelConfig
from minimt3.symbolic.events import NoteEvent, load_midi_events
from minimt3.utils import read_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold-sweep eval for dense AMT checkpoints.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--items", type=int, default=16)
    parser.add_argument("--cache_dir")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--onset_thresholds", default="0.45,0.55,0.65,0.75")
    parser.add_argument("--frame_thresholds", default="0.30,0.40,0.50")
    parser.add_argument("--offset_thresholds", default="0.30,0.40,0.50")
    parser.add_argument("--max_polyphony", type=int)
    parser.add_argument("--max_notes_per_second", type=float)
    parser.add_argument("--min_onset_gap_seconds", type=float)
    parser.add_argument("--min_frame_at_onset", type=float)
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["config"]
    model = DenseAMT(DenseAMTConfig(**cfg.get("model", {}))).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    dataset = DenseAMTDataset(
        args.manifest,
        feature_config=LogMelConfig(**cfg.get("audio", {})),
        split=args.split,
        max_items=args.items,
        cache_dir=args.cache_dir,
        target_config=DenseTargetConfig(**cfg.get("targets", {})),
    )
    combos = [
        (onset_t, frame_t, offset_t)
        for onset_t in _floats(args.onset_thresholds)
        for frame_t in _floats(args.frame_thresholds)
        for offset_t in _floats(args.offset_thresholds)
    ]
    decode_cfg = cfg.get("decode", {})
    max_polyphony = int(args.max_polyphony or decode_cfg.get("max_polyphony", 12))
    max_notes_per_second = float(args.max_notes_per_second or decode_cfg.get("max_notes_per_second", 45.0))
    min_onset_gap_seconds = float(
        args.min_onset_gap_seconds
        if args.min_onset_gap_seconds is not None
        else decode_cfg.get("min_onset_gap_seconds", 0.06)
    )
    min_frame_at_onset = float(
        args.min_frame_at_onset
        if args.min_frame_at_onset is not None
        else decode_cfg.get("min_frame_at_onset", 0.0)
    )
    totals = {combo: {"note_f1": 0.0, "offset_f1": 0.0, "pred": 0, "ref": 0} for combo in combos}
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            row = sample["meta"]
            duration = float(row["end_sec"]) - float(row["start_sec"])
            out = model(sample["features"].unsqueeze(0).to(device))
            ref_notes, _ = load_midi_events(row["midi"], start=float(row["start_sec"]), end=float(row["end_sec"]))
            best = None
            for combo in combos:
                notes = decode_dense_notes(
                    out,
                    duration=duration,
                    onset_threshold=combo[0],
                    frame_threshold=combo[1],
                    offset_threshold=combo[2],
                    max_notes_per_second=max_notes_per_second,
                    max_polyphony=max_polyphony,
                    min_onset_gap_seconds=min_onset_gap_seconds,
                    min_frame_at_onset=min_frame_at_onset,
                )
                metric = note_metrics(notes, ref_notes)
                totals[combo]["note_f1"] += metric["note_f1"]
                totals[combo]["offset_f1"] += metric["offset_f1"]
                totals[combo]["pred"] += len(notes)
                totals[combo]["ref"] += len(ref_notes)
                cand = (metric["note_f1"], metric["offset_f1"], len(notes), combo)
                if best is None or cand > best:
                    best = cand
            print(
                "amt_item "
                f"index={idx} clip_id={row.get('clip_id', idx)} "
                f"best_note_f1={best[0]:.4f} best_offset_f1={best[1]:.4f} "
                f"pred_notes={best[2]} thresholds={best[3]} ref_notes={len(ref_notes)}",
                flush=True,
            )
    count = max(1, len(dataset))
    best_combo = None
    best_score = -1e9
    for combo, values in sorted(totals.items()):
        note_f1 = values["note_f1"] / count
        offset_f1 = values["offset_f1"] / count
        pred_ref = values["pred"] / max(1, values["ref"])
        score = 10.0 * note_f1 + offset_f1 - 0.5 * min(abs(pred_ref - 1.0), 2.0)
        print(
            "amt_summary "
            f"onset_t={combo[0]:.2f} frame_t={combo[1]:.2f} offset_t={combo[2]:.2f} "
            f"note_f1={note_f1:.4f} offset_f1={offset_f1:.4f} "
            f"pred_ref_ratio={pred_ref:.3f} pred_notes={values['pred']} ref_notes={values['ref']} "
            f"score={score:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_combo = combo
    print(f"amt_best thresholds={best_combo} score={best_score:.4f}", flush=True)


def _floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def note_metrics(pred_notes: list[NoteEvent], ref_notes: list[NoteEvent]) -> dict[str, float]:
    try:
        import mir_eval.transcription
        import numpy as np
    except ImportError:
        return {"note_f1": 0.0, "offset_f1": 0.0}
    ref_intervals, ref_pitches = note_arrays(ref_notes, np)
    pred_intervals, pred_pitches = note_arrays(pred_notes, np)
    onset = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals,
        ref_pitches,
        pred_intervals,
        pred_pitches,
        offset_ratio=None,
    )
    offset = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals,
        ref_pitches,
        pred_intervals,
        pred_pitches,
        offset_ratio=0.2,
    )
    return {"note_f1": float(onset[2]), "offset_f1": float(offset[2])}


def note_arrays(notes: list[NoteEvent], np_module):
    if not notes:
        return np_module.zeros((0, 2)), np_module.zeros((0,), dtype=int)
    return (
        np_module.array([[n.start, n.end] for n in notes], dtype=float),
        np_module.array([n.pitch for n in notes], dtype=int),
    )


if __name__ == "__main__":
    main()
