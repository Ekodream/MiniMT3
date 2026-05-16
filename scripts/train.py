#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from minimt3.audio.features import LogMelConfig
from minimt3.data import Collator, MaestroDataset, summarize_token_targets
from minimt3.decode.beam_search import greedy_decode
from minimt3.model.loss import WeightedSeq2SeqLoss
from minimt3.pipeline import build_codec, build_model
from minimt3.symbolic.events import NoteEvent, load_midi_events
from minimt3.utils import ensure_dir, read_yaml, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MiniMT3-Piano.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    cfg = read_yaml(args.config)
    model_cfg = read_yaml(cfg["model_config"])
    seed_everything(int(cfg.get("seed", 42)))

    is_ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if is_ddp:
        dist.init_process_group(backend=cfg.get("dist_backend", "nccl"))
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    rank = dist.get_rank() if is_ddp else 0

    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.get("allow_tf32", True))
    torch.backends.cudnn.allow_tf32 = bool(cfg.get("allow_tf32", True))

    codec = build_codec(model_cfg)
    audio_cfg = LogMelConfig(**model_cfg.get("audio", {}))
    train_ds = MaestroDataset(
        cfg["metadata"],
        split=cfg.get("split", "train"),
        codec=codec,
        feature_config=audio_cfg,
        train_seconds=float(cfg.get("train_seconds", 20.0)),
        max_items=cfg.get("max_items"),
        sampling=cfg.get("train_sampling", "random"),
        seed=int(cfg.get("seed", 42)) + rank,
    )
    val_sampling = cfg.get("val_sampling", "fixed")
    if val_sampling != "fixed":
        raise ValueError("validation must use val_sampling: fixed; random val_loss is meaningless")
    val_ds = MaestroDataset(
        cfg.get("val_metadata", cfg["metadata"]),
        split=cfg.get("val_split", "validation"),
        codec=codec,
        feature_config=audio_cfg,
        train_seconds=float(cfg.get("val_seconds", cfg.get("train_seconds", 20.0))),
        max_items=cfg.get("val_max_items", 64),
        sampling=val_sampling,
        seed=int(cfg.get("seed", 42)),
    )

    if rank == 0:
        train_summary = summarize_token_targets(train_ds, codec, max_items=32)
        val_summary = summarize_token_targets(val_ds, codec, max_items=min(32, len(val_ds)))
        print("train token summary:", train_summary)
        print("fixed val token summary:", val_summary)
        _validate_token_summary("train", train_summary, cfg)
        _validate_token_summary("fixed val", val_summary, cfg)

    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if is_ddp else None
    collate = Collator(codec.pad_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("batch_size", 2)),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(cfg.get("num_workers", 2)),
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(cfg.get("num_workers", 2)) > 0,
        prefetch_factor=int(cfg.get("prefetch_factor", 2)) if int(cfg.get("num_workers", 2)) > 0 else None,
        drop_last=is_ddp,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.get("eval_batch_size", cfg.get("batch_size", 2))),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 2)),
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(cfg.get("num_workers", 2)) > 0,
        prefetch_factor=int(cfg.get("prefetch_factor", 2)) if int(cfg.get("num_workers", 2)) > 0 else None,
    )

    model = build_model(model_cfg, codec).to(device)
    if cfg.get("compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model)
    if is_ddp:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
    criterion = WeightedSeq2SeqLoss(
        codec,
        label_smoothing=float(cfg.get("label_smoothing", 0.05)),
        family_weights=cfg.get("family_weights"),
        eos_aux_weight=float(cfg.get("eos_aux_weight", 0.0)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 3e-4)),
        betas=tuple(cfg.get("betas", [0.9, 0.98])),
        weight_decay=float(cfg.get("weight_decay", 1e-2)),
    )
    total_updates = _estimate_updates(cfg, train_loader)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=_warmup_cosine_lambda(
            int(cfg.get("warmup_steps", 4000)),
            total_updates,
            min_lr_ratio=float(cfg.get("min_lr", 0.0)) / float(cfg.get("lr", 3e-4)),
        ),
    )
    precision = cfg.get("precision", "bf16")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_amp = torch.cuda.is_available() and precision in {"bf16", "fp16"}
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available() and precision == "fp16")
    out_dir = ensure_dir(cfg.get("output_dir", "outputs/ckpt"))
    global_step = 0
    best_val = float("inf")
    last_val: dict[str, Any] | None = None
    if args.resume:
        global_step, best_val = load_training_state(
            args.resume, model, optimizer, scheduler, scaler, device, rank
        )

    grad_accum = int(cfg.get("grad_accum", 1))
    max_steps = int(cfg.get("max_steps") or 0)
    for epoch in range(int(cfg.get("epochs", 5))):
        if max_steps and global_step >= max_steps:
            break
        train_ds.set_epoch(epoch)
        if train_sampler:
            train_sampler.set_epoch(epoch)
        model.train()
        iterator = tqdm(train_loader, disable=rank != 0, desc=f"epoch {epoch + 1}")
        optimizer.zero_grad(set_to_none=True)
        for micro_step, batch in enumerate(iterator):
            features = batch["features"].to(device, non_blocking=True)
            tokens = batch["tokens"].to(device, non_blocking=True)
            decoder_in = tokens[:, :-1]
            target = tokens[:, 1:]
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(features, decoder_in)
                loss_out = criterion(logits, target)
                loss = loss_out.loss / grad_accum
            scaler.scale(loss).backward()
            if (micro_step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if rank == 0 and global_step % int(cfg.get("log_interval", 20)) == 0:
                    lr = scheduler.get_last_lr()[0]
                    fam = " ".join(f"{k}:{v:.3f}" for k, v in loss_out.family_losses.items())
                    iterator.set_postfix(loss=f"{loss_out.loss.item():.4f}", lr=f"{lr:.2e}")
                    print(f"step={global_step} train_loss={loss_out.loss.item():.4f} lr={lr:.3e} {fam}")
                if rank == 0 and global_step % int(cfg.get("eval_interval", 500)) == 0:
                    val = evaluate(model, val_loader, criterion, device, use_amp, amp_dtype)
                    last_val = val
                    print(f"step={global_step} fixed_val_loss={val['loss']:.4f} {val['families']}")
                    maybe_run_debug_decode(model, val_ds, codec, cfg, global_step, device)
                    if val["loss"] < best_val:
                        best_val = val["loss"]
                        save_checkpoint(
                            out_dir / "best.pt",
                            model,
                            model_cfg,
                            codec,
                            optimizer,
                            scheduler,
                            scaler,
                            global_step,
                            best_val,
                            cfg,
                        )
                if rank == 0 and global_step % int(cfg.get("save_interval", 1000)) == 0:
                    save_checkpoint(
                        out_dir / f"step_{global_step}.pt",
                        model,
                        model_cfg,
                        codec,
                        optimizer,
                        scheduler,
                        scaler,
                        global_step,
                        last_val["loss"] if last_val else None,
                        cfg,
                    )
                if max_steps and global_step >= max_steps:
                    break

    if rank == 0:
        val = evaluate(model, val_loader, criterion, device, use_amp, amp_dtype)
        print(f"step={global_step} final_fixed_val_loss={val['loss']:.4f} {val['families']}")
        save_checkpoint(
            out_dir / "last.pt",
            model,
            model_cfg,
            codec,
            optimizer,
            scheduler,
            scaler,
            global_step,
            val["loss"],
            cfg,
        )
        if val["loss"] < best_val:
            save_checkpoint(
                out_dir / "best.pt",
                model,
                model_cfg,
                codec,
                optimizer,
                scheduler,
                scaler,
                global_step,
                val["loss"],
                cfg,
            )
    if is_ddp:
        dist.destroy_process_group()


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp: bool, amp_dtype: torch.dtype) -> dict[str, Any]:
    model.eval()
    losses = []
    family_accum: dict[str, list[float]] = {}
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        tokens = batch["tokens"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(features, tokens[:, :-1])
            loss_out = criterion(logits, tokens[:, 1:])
        losses.append(float(loss_out.loss.item()))
        for key, value in loss_out.family_losses.items():
            family_accum.setdefault(key, []).append(value)
    model.train()
    return {
        "loss": sum(losses) / max(1, len(losses)),
        "families": {k: sum(v) / len(v) for k, v in family_accum.items()},
    }


def _validate_token_summary(name: str, summary: dict[str, Any], cfg: dict[str, Any]) -> None:
    if summary["eos_rate"] < 1.0:
        raise ValueError(f"{name} target eos_rate must be 1.0, got {summary['eos_rate']:.4f}")
    max_target_tokens = int(cfg.get("max_target_tokens") or 0)
    if max_target_tokens and summary["max_target_length"] > max_target_tokens:
        raise ValueError(
            f"{name} max target length {summary['max_target_length']} exceeds max_target_tokens={max_target_tokens}"
        )


@torch.no_grad()
def maybe_run_debug_decode(
    model,
    dataset: MaestroDataset,
    codec,
    cfg: dict[str, Any],
    global_step: int,
    device: torch.device,
) -> None:
    debug_cfg = cfg.get("debug_decode", {})
    if not debug_cfg or not debug_cfg.get("enabled", False):
        return

    module = model.module if hasattr(model, "module") else model
    max_items = min(int(debug_cfg.get("max_items", 2)), len(dataset))
    max_tokens = int(debug_cfg.get("max_tokens", 1600))
    constrained = bool(debug_cfg.get("constrained", True))
    repetition_penalty = float(debug_cfg.get("repetition_penalty", 1.15))
    loop_window = int(debug_cfg.get("loop_window", 16))
    loop_repeats = int(debug_cfg.get("loop_repeats", 4))
    max_time_seconds = debug_cfg.get("max_time_seconds")
    if max_time_seconds is not None:
        max_time_seconds = float(max_time_seconds)

    rows = getattr(dataset, "rows", [])
    items = []
    module.eval()
    for index in range(max_items):
        sample = dataset[index]
        row = rows[index] if index < len(rows) else {}
        features = sample["features"].to(device, non_blocking=True)
        tokens, stats = greedy_decode(
            module,
            features,
            codec,
            max_tokens=max_tokens,
            constrained=constrained,
            repetition_penalty=repetition_penalty,
            loop_window=loop_window,
            loop_repeats=loop_repeats,
            max_time_seconds=max_time_seconds,
            eos_bias_after_seconds=debug_cfg.get("eos_bias_after_seconds"),
            eos_logit_bias=float(debug_cfg.get("eos_logit_bias", 0.0)),
            eos_bias_after_token_ratio=debug_cfg.get("eos_bias_after_token_ratio"),
            force_eos_on_loop=bool(debug_cfg.get("force_eos_on_loop", False)),
            max_tokens_since_shift=debug_cfg.get("max_tokens_since_shift"),
            return_stats=True,
        )
        decoded = codec.decode(tokens, stop_reason=stats.stop_reason)
        ref_notes, _ = load_midi_events(
            row.get("midi"),
            start=float(row.get("start_sec", 0.0)),
            end=float(row.get("end_sec", row.get("duration", 0.0) or 0.0)),
        )
        metric = _note_metrics(decoded.notes, ref_notes)
        invalid_rate = decoded.invalid_events / max(1, decoded.total_events)
        is_loop = "loop" in stats.stop_reason
        item = {
            "eos": float(decoded.eos_hit),
            "loop": float(is_loop),
            "pred_notes": len(decoded.notes),
            "ref_notes": len(ref_notes),
            "note_f1": metric["note_f1"],
            "offset_f1": metric["offset_f1"],
        }
        items.append(item)
        print(
            "debug_decode "
            f"step={global_step} item={index} clip_id={row.get('clip_id', index)} "
            f"eos={decoded.eos_hit} stop={stats.stop_reason} "
            f"pred_notes={len(decoded.notes)} ref_notes={len(ref_notes)} "
            f"note_f1={metric['note_f1']:.4f} offset_f1={metric['offset_f1']:.4f} "
            f"invalid={invalid_rate:.4f} decode_s={stats.wall_time:.2f} "
            f"tok_s={stats.tokens_per_second:.2f}"
        )
    module.train()

    if not items:
        return
    eos_rate = sum(x["eos"] for x in items) / len(items)
    loop_rate = sum(x["loop"] for x in items) / len(items)
    note_f1 = sum(x["note_f1"] for x in items) / len(items)
    offset_f1 = sum(x["offset_f1"] for x in items) / len(items)
    pred_notes = sum(x["pred_notes"] for x in items)
    ref_notes = sum(x["ref_notes"] for x in items)
    pred_ref_ratio = pred_notes / max(1, ref_notes)
    print(
        "debug_decode_summary "
        f"step={global_step} eos_hit_rate={eos_rate:.3f} loop_rate={loop_rate:.3f} "
        f"pred_ref_ratio={pred_ref_ratio:.3f} note_f1={note_f1:.4f} "
        f"offset_f1={offset_f1:.4f} pred_notes={pred_notes} ref_notes={ref_notes}"
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


def save_checkpoint(
    path: Path,
    model,
    model_cfg: dict,
    codec,
    optimizer,
    scheduler,
    scaler,
    step: int,
    val_loss: float | None,
    train_cfg: dict,
) -> None:
    module = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "model": module.state_dict(),
            "model_config": model_cfg,
            "codec_config": {
                "time_shift_ms": codec.time_shift_ms,
                "max_time_shift_steps": codec.max_time_shift_steps,
                "velocity_bins": codec.velocity_bins,
                "time_mode": codec.time_mode,
            },
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "val_loss": val_loss,
            "train_config": train_cfg,
        },
        path,
    )
    print(f"saved {path}")


def load_training_state(path, model, optimizer, scheduler, scaler, device, rank: int) -> tuple[int, float]:
    ckpt = torch.load(path, map_location=device)
    module = model.module if hasattr(model, "module") else model
    module.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    if rank == 0:
        print(f"resumed {path} at step={ckpt.get('step', 0)}")
    return int(ckpt.get("step", 0)), float(ckpt.get("val_loss") or math.inf)


def _estimate_updates(cfg: dict, loader: DataLoader) -> int:
    if cfg.get("max_steps"):
        return max(int(cfg["max_steps"]), int(cfg.get("warmup_steps", 4000)) + 1)
    steps_per_epoch = max(1, len(loader) // int(cfg.get("grad_accum", 1)))
    return max(steps_per_epoch * int(cfg.get("epochs", 5)), int(cfg.get("warmup_steps", 4000)) + 1)


def _warmup_cosine_lambda(warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.0):
    min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))

    def fn(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return fn


if __name__ == "__main__":
    main()
