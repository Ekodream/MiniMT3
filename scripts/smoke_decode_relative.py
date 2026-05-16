#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from minimt3.data import MaestroDataset
from minimt3.decode.beam_search import greedy_decode
from minimt3.pipeline import feature_config_from_model, load_checkpoint
from minimt3.symbolic.events import NoteEvent, load_midi_events


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test relative-time cached decoding on fixed clips.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--manifest", default="data/cache/maestro_debug_clips_rel8.json")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--items", type=int, default=2)
    parser.add_argument("--max_tokens", type=int, default=900)
    parser.add_argument("--max_time_seconds", type=float, default=8.5)
    parser.add_argument("--eos_bias_after_seconds", type=float)
    parser.add_argument("--eos_logit_bias", type=float, default=0.0)
    parser.add_argument("--eos_bias_after_token_ratio", type=float)
    parser.add_argument("--force_eos_on_loop", action="store_true")
    parser.add_argument("--max_tokens_since_shift", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, codec, model_config = load_checkpoint(args.ckpt, device=device)
    dataset = MaestroDataset(
        args.manifest,
        split=args.split,
        codec=codec,
        feature_config=feature_config_from_model(model_config),
        train_seconds=8.0,
        max_items=args.items,
        sampling="fixed",
    )

    totals = {
        "eos": 0,
        "loop": 0,
        "pred_notes": 0,
        "ref_notes": 0,
        "note_f1": 0.0,
        "offset_f1": 0.0,
        "tok_s": 0.0,
    }
    for index in range(len(dataset)):
        sample = dataset[index]
        row = dataset.rows[index]
        tokens, stats = greedy_decode(
            model,
            sample["features"].to(device),
            codec,
            max_tokens=args.max_tokens,
            constrained=True,
            repetition_penalty=args.repetition_penalty,
            max_time_seconds=args.max_time_seconds,
            eos_bias_after_seconds=args.eos_bias_after_seconds,
            eos_logit_bias=args.eos_logit_bias,
            eos_bias_after_token_ratio=args.eos_bias_after_token_ratio,
            force_eos_on_loop=args.force_eos_on_loop,
            max_tokens_since_shift=args.max_tokens_since_shift,
            return_stats=True,
        )
        decoded = codec.decode(tokens, stop_reason=stats.stop_reason)
        ref_notes, _ = load_midi_events(
            row["midi"],
            start=float(row.get("start_sec", 0.0)),
            end=float(row.get("end_sec", row.get("duration", 0.0) or 0.0)),
        )
        metrics = _note_metrics(decoded.notes, ref_notes)
        invalid_rate = decoded.invalid_events / max(1, decoded.total_events)
        loop = "loop" in stats.stop_reason
        totals["eos"] += int(decoded.eos_hit)
        totals["loop"] += int(loop)
        totals["pred_notes"] += len(decoded.notes)
        totals["ref_notes"] += len(ref_notes)
        totals["note_f1"] += metrics["note_f1"]
        totals["offset_f1"] += metrics["offset_f1"]
        totals["tok_s"] += stats.tokens_per_second
        print(
            "smoke_decode "
            f"item={index} clip_id={row.get('clip_id', index)} "
            f"eos={decoded.eos_hit} stop={stats.stop_reason} "
            f"pred_notes={len(decoded.notes)} ref_notes={len(ref_notes)} "
            f"note_f1={metrics['note_f1']:.4f} offset_f1={metrics['offset_f1']:.4f} "
            f"invalid={invalid_rate:.4f} decode_s={stats.wall_time:.2f} "
            f"tok_s={stats.tokens_per_second:.2f}"
        )

    count = max(1, len(dataset))
    print(
        "smoke_decode_summary "
        f"items={count} eos_hit_rate={totals['eos'] / count:.3f} "
        f"loop_rate={totals['loop'] / count:.3f} "
        f"pred_ref_ratio={totals['pred_notes'] / max(1, totals['ref_notes']):.3f} "
        f"note_f1={totals['note_f1'] / count:.4f} "
        f"offset_f1={totals['offset_f1'] / count:.4f} "
        f"avg_tok_s={totals['tok_s'] / count:.2f} "
        f"pred_notes={totals['pred_notes']} ref_notes={totals['ref_notes']}"
    )


def _note_metrics(pred_notes: list[NoteEvent], ref_notes: list[NoteEvent]) -> dict[str, float]:
    try:
        import mir_eval.transcription
        import numpy as np
    except ImportError:
        return _simple_note_metrics(pred_notes, ref_notes)

    ref_intervals, ref_pitches = _note_arrays(ref_notes, np)
    pred_intervals, pred_pitches = _note_arrays(pred_notes, np)
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


def _note_arrays(notes: list[NoteEvent], np_module):
    if not notes:
        return np_module.zeros((0, 2)), np_module.zeros((0,), dtype=int)
    intervals = np_module.array([[n.start, n.end] for n in notes], dtype=float)
    pitches = np_module.array([n.pitch for n in notes], dtype=int)
    return intervals, pitches


def _simple_note_metrics(pred_notes: list[NoteEvent], ref_notes: list[NoteEvent]) -> dict[str, float]:
    matched = set()
    hits = 0
    for pred in pred_notes:
        for idx, ref in enumerate(ref_notes):
            if idx in matched or pred.pitch != ref.pitch or abs(pred.start - ref.start) > 0.05:
                continue
            matched.add(idx)
            hits += 1
            break
    precision = hits / max(1, len(pred_notes))
    recall = hits / max(1, len(ref_notes))
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {"note_f1": f1, "offset_f1": 0.0}


if __name__ == "__main__":
    main()
