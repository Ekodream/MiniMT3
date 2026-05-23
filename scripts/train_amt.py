#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime as dt
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from minimt3.amt.data import DenseAMTCollator, DenseAMTDataset
from minimt3.amt.decode import decode_dense_notes
from minimt3.amt.loss import DenseAMTLoss
from minimt3.amt.model import DenseAMT, DenseAMTConfig
from minimt3.amt.presets import apply_decode_preset
from minimt3.amt.targets import DenseTargetConfig
from minimt3.audio.features import LogMelConfig
from minimt3.symbolic.events import NoteEvent, load_midi_events
from minimt3.utils import ensure_dir, read_yaml, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train v5 dense AMT model.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = read_yaml(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    is_ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if is_ddp:
        dist.init_process_group(
            backend=cfg.get("dist_backend", "nccl"),
            timeout=dt.timedelta(minutes=int(cfg.get("dist_timeout_minutes", 120))),
        )
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    rank = dist.get_rank() if is_ddp else 0
    if cfg.get("allow_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    audio_cfg = LogMelConfig(**cfg.get("audio", {}))
    target_cfg = DenseTargetConfig(**cfg.get("targets", {}))
    train_ds = DenseAMTDataset(
        cfg["train_manifest"],
        feature_config=audio_cfg,
        split=cfg.get("train_split", "train"),
        max_items=cfg.get("max_items"),
        cache_dir=cfg.get("train_cache_dir"),
        target_config=target_cfg,
    )
    val_ds = DenseAMTDataset(
        cfg["val_manifest"],
        feature_config=audio_cfg,
        split=cfg.get("val_split", "validation"),
        max_items=cfg.get("val_max_items"),
        cache_dir=cfg.get("val_cache_dir"),
        target_config=target_cfg,
    )
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if is_ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("batch_size", 32)),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(cfg.get("num_workers", 4)) > 0,
        prefetch_factor=int(cfg.get("prefetch_factor", 2)) if int(cfg.get("num_workers", 4)) > 0 else None,
        collate_fn=DenseAMTCollator(),
        drop_last=is_ddp,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.get("eval_batch_size", 8)),
        shuffle=False,
        num_workers=int(cfg.get("eval_workers", 2)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(cfg.get("eval_workers", 2)) > 0,
        collate_fn=DenseAMTCollator(),
    )

    model_cfg = DenseAMTConfig(**cfg.get("model", {}))
    model = DenseAMT(model_cfg).to(device)
    if cfg.get("init_from"):
        load_compatible_weights(
            cfg["init_from"],
            model,
            device,
            rank,
            expand=bool(cfg.get("init_expand", False)),
        )
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    loss_cfg = cfg.get("loss", {})
    criterion = DenseAMTLoss(**loss_cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 5e-5)),
        betas=tuple(cfg.get("betas", [0.9, 0.98])),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )
    max_steps = int(cfg.get("max_steps", 1000))
    warmup = int(cfg.get("warmup_steps", 100))
    min_lr = float(cfg.get("min_lr", 1e-5))
    base_lr = float(cfg.get("lr", 5e-5))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_factor(step, warmup, max_steps, base_lr, min_lr),
    )
    precision = str(cfg.get("precision", "bf16")).lower()
    use_amp = torch.cuda.is_available() and precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available() and precision == "fp16")

    out_dir = ensure_dir(cfg.get("output_dir", "outputs/ckpt_amt_v5"))
    best_score = -math.inf
    best_step = 0
    global_step = 0
    epochs = int(cfg.get("epochs", 100))
    log_interval = int(cfg.get("log_interval", 20))
    eval_interval = int(cfg.get("eval_interval", 100))
    for epoch in range(epochs):
        if max_steps and global_step >= max_steps:
            break
        if train_sampler:
            train_sampler.set_epoch(epoch)
        model.train()
        accum: dict[str, list[float]] = {}
        iterator = tqdm(train_loader, disable=rank != 0, desc=f"epoch {epoch + 1}")
        for batch in iterator:
            if max_steps and global_step >= max_steps:
                break
            features = batch["features"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                model_out = model(features)
                loss_out = criterion(model_out, batch)
                loss = loss_out.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            for key, value in {"loss": float(loss.detach().cpu()), **loss_out.logs}.items():
                accum.setdefault(key, []).append(value)
            if rank == 0 and global_step % log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                logs = " ".join(f"{k}:{sum(v[-log_interval:]) / len(v[-log_interval:]):.3f}" for k, v in accum.items())
                print(f"step={global_step} train {logs} lr={lr:.3e}", flush=True)
            if global_step % eval_interval == 0:
                stop_tensor = torch.zeros(1, dtype=torch.int32, device=device)
                if rank == 0:
                    eval_model = model.module if hasattr(model, "module") else model
                    val = evaluate(eval_model, val_loader, criterion, device, use_amp, amp_dtype, cfg)
                    print(f"step={global_step} val_loss={val['loss']:.4f} {val['losses']}", flush=True)
                    debug = debug_decode(eval_model, val_ds, device, cfg, global_step)
                    score = selection_score(debug)
                    print(f"checkpoint_selection score={score:.5f} best={best_score:.5f}", flush=True)
                    min_delta = float(cfg.get("early_stop_min_delta", 1e-5))
                    improved = score > best_score + min_delta
                    if improved:
                        best_score = score
                        best_step = global_step
                        save_checkpoint(out_dir / "best.pt", eval_model, cfg, global_step, best_score)
                    if global_step % int(cfg.get("save_interval", 300)) == 0:
                        save_checkpoint(out_dir / f"step_{global_step}.pt", eval_model, cfg, global_step, score)
                    patience_evals = int(cfg.get("early_stop_patience_evals", 0) or 0)
                    early_stop_min_steps = int(cfg.get("early_stop_min_steps", 0) or 0)
                    if (
                        patience_evals > 0
                        and best_step > 0
                        and global_step >= early_stop_min_steps
                        and global_step - best_step >= patience_evals * eval_interval
                    ):
                        print(
                            "early_stop "
                            f"step={global_step} best_step={best_step} "
                            f"patience_evals={patience_evals}",
                            flush=True,
                        )
                        stop_tensor.fill_(1)
                if is_ddp:
                    dist.broadcast(stop_tensor, src=0)
                    dist.barrier()
                if int(stop_tensor.item()) != 0:
                    global_step = max_steps if max_steps else global_step
                    break
    if rank == 0:
        eval_model = model.module if hasattr(model, "module") else model
        save_checkpoint(out_dir / "last.pt", eval_model, cfg, global_step, best_score)
    if is_ddp:
        dist.destroy_process_group()


def lr_factor(step: int, warmup: int, max_steps: int, base_lr: float, min_lr: float) -> float:
    if warmup > 0 and step < warmup:
        return max(1e-4, (step + 1) / warmup)
    if max_steps <= warmup:
        return 1.0
    progress = min(1.0, max(0.0, (step - warmup) / max(1, max_steps - warmup)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_factor = min_lr / max(base_lr, 1e-12)
    return min_factor + (1.0 - min_factor) * cosine


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp: bool, amp_dtype: torch.dtype, cfg: dict[str, Any]) -> dict[str, Any]:
    model.eval()
    losses = []
    logs: dict[str, list[float]] = {}
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(features)
            loss_out = criterion(out, batch)
        losses.append(float(loss_out.loss.detach().cpu()))
        for key, value in loss_out.logs.items():
            logs.setdefault(key, []).append(value)
    model.train()
    return {"loss": sum(losses) / max(1, len(losses)), "losses": {k: sum(v) / len(v) for k, v in logs.items()}}


@torch.no_grad()
def debug_decode(model, dataset: DenseAMTDataset, device, cfg: dict[str, Any], step: int) -> dict[str, float]:
    was_training = model.training
    model.eval()
    decode_cfg = apply_decode_preset(cfg.get("decode", {}), cfg.get("decode_preset"))
    combos = _debug_decode_combos(cfg, decode_cfg)
    max_items = min(int(cfg.get("debug_items", 8)), len(dataset))
    totals = {
        combo: {"note_f1": 0.0, "offset_f1": 0.0, "pred": 0, "ref": 0, "bad_clips": 0}
        for combo in combos
    }
    for idx in range(max_items):
        sample = dataset[idx]
        features = sample["features"].unsqueeze(0).to(device)
        row = sample["meta"]
        duration = float(row.get("end_sec", row.get("duration", 0.0))) - float(row.get("start_sec", 0.0))
        out = model(features)
        chord_onset_threshold = decode_cfg.get("chord_onset_threshold")
        if bool(decode_cfg.get("disable_chord_recovery", False)):
            chord_onset_threshold = None
        ref_notes, _ = load_midi_events(row["midi"], start=float(row["start_sec"]), end=float(row["end_sec"]))
        best_item = None
        for combo in combos:
            notes = decode_dense_notes(
                out,
                duration=duration,
                onset_threshold=combo[0],
                frame_threshold=combo[1],
                offset_threshold=combo[2],
                min_note_seconds=float(decode_cfg.get("min_note_seconds", 0.04)),
                max_notes_per_second=float(decode_cfg.get("max_notes_per_second", 45.0)),
                max_polyphony=int(decode_cfg.get("max_polyphony", 12)),
                min_onset_gap_seconds=float(decode_cfg.get("min_onset_gap_seconds", 0.06)),
                min_frame_at_onset=float(decode_cfg.get("min_frame_at_onset", 0.0)),
                onset_frame_fusion_weight=float(decode_cfg.get("onset_frame_fusion_weight", 0.0)),
                chord_onset_threshold=chord_onset_threshold,
                chord_frame_threshold=float(decode_cfg.get("chord_frame_threshold", 0.35)),
                chord_window_frames=int(decode_cfg.get("chord_window_frames", 1)),
                chord_score_ratio=float(decode_cfg.get("chord_score_ratio", 0.75)),
                onset_peak_prominence=float(decode_cfg.get("onset_peak_prominence", 0.0)),
                max_notes_per_start_window=decode_cfg.get("max_notes_per_start_window"),
                start_window_seconds=float(decode_cfg.get("start_window_seconds", 0.08)),
                use_duration_head=bool(decode_cfg.get("use_duration_head", True)),
                max_duration_seconds=float(decode_cfg.get("max_duration_seconds", 8.0)),
                duration_extension_weight=float(decode_cfg.get("duration_extension_weight", 1.0)),
                time_shift_clip_frames=float(cfg.get("targets", {}).get("time_shift_clip_frames", 1.0)),
                consume_note_energy=bool(decode_cfg.get("consume_note_energy", False)),
                energy_neighbor_pitches=int(decode_cfg.get("energy_neighbor_pitches", 1)),
                energy_overlap_ratio=float(decode_cfg.get("energy_overlap_ratio", 0.5)),
                infer_onsets_from_frame_diff=bool(decode_cfg.get("infer_onsets_from_frame_diff", False)),
                frame_diff_n=int(decode_cfg.get("frame_diff_n", 2)),
                frame_diff_scale=float(decode_cfg.get("frame_diff_scale", 1.0)),
            )
            eval_notes, eval_ref_notes = _maybe_center_crop_notes(notes, ref_notes, row, cfg)
            metric = note_metrics(eval_notes, eval_ref_notes)
            totals[combo]["note_f1"] += metric["note_f1"]
            totals[combo]["offset_f1"] += metric["offset_f1"]
            totals[combo]["pred"] += len(eval_notes)
            totals[combo]["ref"] += len(eval_ref_notes)
            if metric["note_f1"] < float(cfg.get("bad_clip_f1_threshold", 0.05)):
                totals[combo]["bad_clips"] += 1
            candidate = (metric["note_f1"], metric["offset_f1"], -len(eval_notes), combo, len(eval_notes), len(eval_ref_notes))
            if best_item is None or candidate[:4] > best_item[:4]:
                best_item = candidate
        assert best_item is not None
        print(
            "debug_amt "
            f"step={step} item={idx} clip_id={row.get('clip_id', idx)} "
            f"pred_notes={best_item[4]} ref_notes={best_item[5]} "
            f"note_f1={best_item[0]:.4f} offset_f1={best_item[1]:.4f} thresholds={best_item[3]}",
            flush=True,
        )
    count = max(1, max_items)
    best_summary = None
    best_score = -math.inf
    for combo, values in sorted(totals.items()):
        pred_ref = values["pred"] / max(1, values["ref"])
        summary = {
            "note_f1": values["note_f1"] / count,
            "offset_f1": values["offset_f1"] / count,
            "pred_ref_ratio": pred_ref,
            "bad_clip_rate": values["bad_clips"] / count,
            "pred_notes": values["pred"],
            "ref_notes": values["ref"],
            "thresholds": combo,
        }
        score = selection_score(summary)
        if len(combos) > 1:
            print(
                "debug_amt_sweep "
                f"step={step} thresholds={combo} note_f1={summary['note_f1']:.4f} "
                f"offset_f1={summary['offset_f1']:.4f} pred_ref_ratio={pred_ref:.3f} "
                f"score={score:.5f}",
                flush=True,
            )
        if score > best_score:
            best_score = score
            best_summary = summary
    assert best_summary is not None
    print(
        "debug_amt_summary "
        f"step={step} thresholds={best_summary['thresholds']} "
        f"note_f1={best_summary['note_f1']:.4f} offset_f1={best_summary['offset_f1']:.4f} "
        f"pred_ref_ratio={best_summary['pred_ref_ratio']:.3f} bad_clip_rate={best_summary['bad_clip_rate']:.3f} "
        f"pred_notes={best_summary['pred_notes']} ref_notes={best_summary['ref_notes']}",
        flush=True,
    )
    if was_training:
        model.train()
    return best_summary


def _debug_decode_combos(cfg: dict[str, Any], decode_cfg: dict[str, Any]) -> list[tuple[float, float, float]]:
    sweep = cfg.get("selection_sweep") or cfg.get("eval", {}).get("selection_sweep")
    if not sweep or not bool(sweep.get("enabled", False)):
        return [
            (
                float(decode_cfg.get("onset_threshold", 0.45)),
                float(decode_cfg.get("frame_threshold", 0.35)),
                float(decode_cfg.get("offset_threshold", 0.35)),
            )
        ]
    onset_values = _float_values(sweep.get("onset_thresholds", decode_cfg.get("onset_threshold", 0.45)))
    frame_values = _float_values(sweep.get("frame_thresholds", decode_cfg.get("frame_threshold", 0.35)))
    offset_values = _float_values(sweep.get("offset_thresholds", decode_cfg.get("offset_threshold", 0.35)))
    return [(o, f, off) for o in onset_values for f in frame_values for off in offset_values]


def _float_values(value: Any) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def selection_score(debug: dict[str, float]) -> float:
    pred_ref_ratio = float(debug.get("pred_ref_ratio", 0.0))
    if pred_ref_ratio <= 0:
        return -math.inf
    ratio_error = abs(math.log(max(1e-6, pred_ref_ratio)))
    over_generation = max(0.0, pred_ref_ratio - 1.35)
    under_generation = max(0.0, 0.55 - pred_ref_ratio)
    return (
        10.0 * float(debug.get("note_f1", 0.0))
        + float(debug.get("offset_f1", 0.0))
        - 1.15 * ratio_error
        - 0.65 * over_generation
        - 0.35 * under_generation
        - 1.0 * float(debug.get("bad_clip_rate", 0.0))
    )


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
    return (
        np_module.array([[n.start, n.end] for n in notes], dtype=float),
        np_module.array([n.pitch for n in notes], dtype=int),
    )


def _maybe_center_crop_notes(
    pred_notes: list[NoteEvent],
    ref_notes: list[NoteEvent],
    row: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[list[NoteEvent], list[NoteEvent]]:
    margin = float(cfg.get("targets", {}).get("supervision_margin_seconds", 0.0) or 0.0)
    if margin <= 0.0 or not bool(cfg.get("decode", {}).get("eval_center_only", False)):
        return pred_notes, ref_notes
    duration = float(row.get("end_sec", 0.0)) - float(row.get("start_sec", 0.0))
    if duration <= margin * 2.0:
        return pred_notes, ref_notes
    lo = margin
    hi = duration - margin

    def keep(note: NoteEvent) -> bool:
        return lo <= note.start < hi

    return [note for note in pred_notes if keep(note)], [note for note in ref_notes if keep(note)]


def simple_note_metrics(pred_notes: list[NoteEvent], ref_notes: list[NoteEvent]) -> dict[str, float]:
    matched = set()
    hits = 0
    for pred in pred_notes:
        for idx, ref in enumerate(ref_notes):
            if idx not in matched and pred.pitch == ref.pitch and abs(pred.start - ref.start) <= 0.05:
                matched.add(idx)
                hits += 1
                break
    precision = hits / max(1, len(pred_notes))
    recall = hits / max(1, len(ref_notes))
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {"note_f1": f1, "offset_f1": 0.0}


def load_compatible_weights(path, model: DenseAMT, device, rank: int, expand: bool = False) -> None:
    ckpt = torch.load(path, map_location=device)
    source = ckpt.get("model", ckpt)
    current = model.state_dict()
    compatible = {k: v for k, v in source.items() if k in current and current[k].shape == v.shape}
    expanded: dict[str, torch.Tensor] = {}
    if expand:
        for key, source_value in source.items():
            if key not in current or key in compatible:
                continue
            target_value = current[key]
            if not torch.is_floating_point(source_value) or not torch.is_floating_point(target_value):
                continue
            if source_value.ndim != target_value.ndim:
                continue
            expanded_value = _expanded_initial_value(target_value, key)
            _copy_expanded_tensor(expanded_value, source_value, key)
            expanded[key] = expanded_value
        compatible.update(expanded)
    incompatible = model.load_state_dict(compatible, strict=False)
    if rank == 0:
        print(
            f"initialized compatible AMT weights from {path} "
            f"keys={len(compatible)} expanded={len(expanded)}",
            flush=True,
        )
        if incompatible.missing_keys:
            print(f"init_missing_keys={incompatible.missing_keys[:20]}", flush=True)


def _copy_expanded_tensor(target: torch.Tensor, source: torch.Tensor, key: str) -> None:
    """Copy source weights into the overlapping subspace of a wider target tensor."""
    if "in_proj_weight" in key and target.ndim == 2 and source.ndim == 2:
        _copy_qkv_rows(target, source)
        return
    if "in_proj_bias" in key and target.ndim == 1 and source.ndim == 1:
        _copy_qkv_rows(target.unsqueeze(1), source.unsqueeze(1))
        return
    if _is_gru_gate_tensor(key, target, source):
        _copy_gru_gates(target, source)
        return
    slices = tuple(slice(0, min(t, s)) for t, s in zip(target.shape, source.shape))
    target[slices].copy_(source[slices])


def _expanded_initial_value(target: torch.Tensor, key: str) -> torch.Tensor:
    if target.ndim == 1 and key.endswith("weight") and any(part in key.lower() for part in ("norm", "layernorm")):
        return target.clone()
    return torch.zeros_like(target)


def _copy_qkv_rows(target: torch.Tensor, source: torch.Tensor) -> None:
    target_rows = target.shape[0] // 3
    source_rows = source.shape[0] // 3
    cols = min(target.shape[1], source.shape[1])
    rows = min(target_rows, source_rows)
    for idx in range(3):
        target[idx * target_rows : idx * target_rows + rows, :cols].copy_(
            source[idx * source_rows : idx * source_rows + rows, :cols]
        )


def _is_gru_gate_tensor(key: str, target: torch.Tensor, source: torch.Tensor) -> bool:
    if not any(part in key for part in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")):
        return False
    return target.ndim in {1, 2} and source.ndim == target.ndim and target.shape[0] % 3 == 0 and source.shape[0] % 3 == 0


def _copy_gru_gates(target: torch.Tensor, source: torch.Tensor) -> None:
    target_rows = target.shape[0] // 3
    source_rows = source.shape[0] // 3
    rows = min(target_rows, source_rows)
    if target.ndim == 1:
        for idx in range(3):
            target[idx * target_rows : idx * target_rows + rows].copy_(
                source[idx * source_rows : idx * source_rows + rows]
            )
    else:
        cols = min(target.shape[1], source.shape[1])
        for idx in range(3):
            target[idx * target_rows : idx * target_rows + rows, :cols].copy_(
                source[idx * source_rows : idx * source_rows + rows, :cols]
            )


def save_checkpoint(path: str | Path, model: DenseAMT, cfg: dict[str, Any], step: int, score: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg, "step": step, "score": score}, path)
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
