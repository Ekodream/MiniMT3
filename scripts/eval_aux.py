#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from minimt3.data import MaestroDataset
from minimt3.pipeline import feature_config_from_model, load_checkpoint
from minimt3.symbolic.events import NoteEvent, PITCH_MIN, load_midi_events


@dataclass
class AuxPrediction:
    notes: list[NoteEvent]
    onset_threshold: float
    frame_threshold: float


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate onset/frame auxiliary heads on fixed clips.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--manifest", default="data/cache/maestro_val_clips_abs2.json")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--items", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--onset_thresholds", default="0.20,0.30,0.40,0.50")
    parser.add_argument("--frame_thresholds", default="0.20,0.30,0.40")
    parser.add_argument("--min_note_seconds", type=float, default=0.04)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, codec, model_config = load_checkpoint(args.ckpt, device=device)
    dataset = MaestroDataset(
        args.manifest,
        split=args.split,
        codec=codec,
        feature_config=feature_config_from_model(model_config),
        train_seconds=args.seconds,
        max_items=args.items,
        sampling="fixed",
    )
    onset_thresholds = _parse_thresholds(args.onset_thresholds)
    frame_thresholds = _parse_thresholds(args.frame_thresholds)
    totals = {
        (onset_t, frame_t): {"note_f1": 0.0, "offset_f1": 0.0, "pred": 0, "ref": 0}
        for onset_t in onset_thresholds
        for frame_t in frame_thresholds
    }

    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            row = dataset.rows[index]
            features = sample["features"].to(device)
            memory = model.encode(features)
            onset_probs = torch.sigmoid(model.onset_head(memory))[0].cpu()
            frame_probs = torch.sigmoid(model.frame_head(memory))[0].cpu()
            duration = float(row.get("end_sec", args.seconds)) - float(row.get("start_sec", 0.0))
            ref_notes, _ = load_midi_events(
                row["midi"],
                start=float(row.get("start_sec", 0.0)),
                end=float(row.get("end_sec", row.get("duration", args.seconds) or args.seconds)),
            )
            best_item = None
            for onset_t in onset_thresholds:
                for frame_t in frame_thresholds:
                    pred = decode_aux_notes(
                        onset_probs,
                        frame_probs,
                        duration=max(0.01, duration),
                        onset_threshold=onset_t,
                        frame_threshold=frame_t,
                        min_note_seconds=args.min_note_seconds,
                    )
                    metrics = note_metrics(pred.notes, ref_notes)
                    key = (onset_t, frame_t)
                    totals[key]["note_f1"] += metrics["note_f1"]
                    totals[key]["offset_f1"] += metrics["offset_f1"]
                    totals[key]["pred"] += len(pred.notes)
                    totals[key]["ref"] += len(ref_notes)
                    candidate = (metrics["note_f1"], metrics["offset_f1"], onset_t, frame_t, len(pred.notes))
                    if best_item is None or candidate > best_item:
                        best_item = candidate
            print(
                "aux_item "
                f"index={index} clip_id={row.get('clip_id', index)} "
                f"best_note_f1={best_item[0]:.4f} best_offset_f1={best_item[1]:.4f} "
                f"onset_t={best_item[2]:.2f} frame_t={best_item[3]:.2f} "
                f"pred_notes={best_item[4]} ref_notes={len(ref_notes)}"
            )

    count = max(1, len(dataset))
    best_key = None
    best_score = -1.0
    for key, values in sorted(totals.items()):
        note_f1 = values["note_f1"] / count
        offset_f1 = values["offset_f1"] / count
        pred_ref = values["pred"] / max(1, values["ref"])
        score = note_f1 - 0.05 * min(abs(pred_ref - 1.0), 2.0)
        print(
            "aux_summary "
            f"onset_t={key[0]:.2f} frame_t={key[1]:.2f} "
            f"note_f1={note_f1:.4f} offset_f1={offset_f1:.4f} "
            f"pred_ref_ratio={pred_ref:.3f} pred_notes={values['pred']} ref_notes={values['ref']}"
        )
        if score > best_score:
            best_score = score
            best_key = key
    print(f"aux_best onset_t={best_key[0]:.2f} frame_t={best_key[1]:.2f} score={best_score:.4f}")


def _parse_thresholds(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def decode_aux_notes(
    onset_probs: torch.Tensor,
    frame_probs: torch.Tensor,
    duration: float,
    onset_threshold: float,
    frame_threshold: float,
    min_note_seconds: float,
) -> AuxPrediction:
    frames, pitches = onset_probs.shape
    frame_seconds = duration / max(1, frames)
    min_frames = max(1, int(round(min_note_seconds / max(1e-6, frame_seconds))))
    notes: list[NoteEvent] = []
    for pitch_idx in range(pitches):
        active_start: int | None = None
        last_frame = 0
        for frame_idx in range(frames):
            onset = float(onset_probs[frame_idx, pitch_idx]) >= onset_threshold
            active_frame = float(frame_probs[frame_idx, pitch_idx]) >= frame_threshold
            if active_start is None:
                if onset:
                    active_start = frame_idx
                    last_frame = frame_idx
                continue
            if active_frame or frame_idx - active_start < min_frames:
                last_frame = frame_idx
                continue
            _append_note(notes, pitch_idx, active_start, last_frame + 1, frame_seconds)
            active_start = frame_idx if onset else None
            last_frame = frame_idx
        if active_start is not None:
            _append_note(notes, pitch_idx, active_start, max(last_frame + 1, active_start + min_frames), frame_seconds)
    notes.sort(key=lambda n: (n.start, n.pitch, n.end))
    return AuxPrediction(notes, onset_threshold, frame_threshold)


def _append_note(notes: list[NoteEvent], pitch_idx: int, start_frame: int, end_frame: int, frame_seconds: float) -> None:
    start = start_frame * frame_seconds
    end = max(start + frame_seconds, end_frame * frame_seconds)
    notes.append(NoteEvent(PITCH_MIN + pitch_idx, start, end, velocity=80))


def note_metrics(pred_notes: list[NoteEvent], ref_notes: list[NoteEvent]) -> dict[str, float]:
    try:
        import mir_eval.transcription
        import numpy as np
    except ImportError:
        return simple_note_metrics(pred_notes, ref_notes)

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
    intervals = np_module.array([[n.start, n.end] for n in notes], dtype=float)
    pitches = np_module.array([n.pitch for n in notes], dtype=int)
    return intervals, pitches


def simple_note_metrics(pred_notes: list[NoteEvent], ref_notes: list[NoteEvent]) -> dict[str, float]:
    matched = set()
    hits = 0
    for pred in pred_notes:
        for idx, ref in enumerate(ref_notes):
            if idx in matched:
                continue
            if pred.pitch == ref.pitch and abs(pred.start - ref.start) <= 0.05:
                matched.add(idx)
                hits += 1
                break
    precision = hits / max(1, len(pred_notes))
    recall = hits / max(1, len(ref_notes))
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {"note_f1": f1, "offset_f1": 0.0}


if __name__ == "__main__":
    main()
