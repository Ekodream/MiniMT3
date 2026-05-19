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
from torch.nn import functional as F
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
from minimt3.symbolic.events import EventCodec, NoteEvent, load_midi_events
from minimt3.utils import ensure_dir, read_yaml, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MiniMT3-Piano.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--resume", default="")
    parser.add_argument("--init_from", default="")
    args = parser.parse_args()
    cfg = read_yaml(args.config)
    model_cfg = read_yaml(cfg["model_config"])
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

    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.get("allow_tf32", True))
    torch.backends.cudnn.allow_tf32 = bool(cfg.get("allow_tf32", True))

    codec = build_codec(model_cfg)
    audio_cfg = LogMelConfig(**model_cfg.get("audio", {}))
    include_ties = bool(cfg.get("include_ties", False))
    aux_cfg = cfg.get("auxiliary", {})
    aux_enabled = bool(aux_cfg.get("enabled", False))
    train_ds = MaestroDataset(
        cfg["metadata"],
        split=cfg.get("split", "train"),
        codec=codec,
        feature_config=audio_cfg,
        train_seconds=float(cfg.get("train_seconds", 20.0)),
        max_items=cfg.get("max_items"),
        sampling=cfg.get("train_sampling", "random"),
        seed=int(cfg.get("seed", 42)) + rank,
        include_ties=include_ties,
        aux_targets=aux_enabled,
        onset_width_frames=int(aux_cfg.get("onset_width_frames", 1)),
        target_cover_to_end=bool(cfg.get("target_cover_to_end", False)),
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
        include_ties=include_ties,
        aux_targets=aux_enabled,
        onset_width_frames=int(aux_cfg.get("onset_width_frames", 1)),
        target_cover_to_end=bool(cfg.get("target_cover_to_end", False)),
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
            find_unused_parameters=not aux_enabled,
            gradient_as_bucket_view=True,
        )
    init_from = args.init_from or cfg.get("init_from", "")
    if init_from and not args.resume:
        load_model_weights(
            init_from,
            model,
            device,
            rank,
            codec=codec,
            skip_families=set(cfg.get("init_skip_families", [])),
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
    best_score = -math.inf
    last_val: dict[str, Any] | None = None
    last_debug: dict[str, Any] | None = None
    bad_debug_evals = 0
    if args.resume:
        global_step, best_val, best_score = load_training_state(
            args.resume, model, optimizer, scheduler, scaler, device, rank
        )

    grad_accum = int(cfg.get("grad_accum", 1))
    max_steps = int(cfg.get("max_steps") or 0)
    stop_training = False
    for epoch in range(int(cfg.get("epochs", 5))):
        if stop_training or (max_steps and global_step >= max_steps):
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
                model_out = model(features, decoder_in, return_aux=aux_enabled)
                logits = model_out["logits"] if aux_enabled else model_out
                loss_out = criterion(logits, target)
                total_loss = loss_out.loss
                if aux_enabled:
                    aux_loss, aux_logs = auxiliary_loss(model_out, batch, device, aux_cfg)
                    total_loss = total_loss + aux_loss
                    for key, value in aux_logs.items():
                        loss_out.family_losses[key] = value
                loss = total_loss / grad_accum
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
                    iterator.set_postfix(loss=f"{total_loss.item():.4f}", lr=f"{lr:.2e}")
                    print(f"step={global_step} train_loss={total_loss.item():.4f} lr={lr:.3e} {fam}")
                if global_step % int(cfg.get("eval_interval", 500)) == 0:
                    stop_now = False
                    stop_tensor = torch.zeros(1, dtype=torch.int32, device=device)
                    if rank == 0:
                        eval_model = model.module if hasattr(model, "module") else model
                        val = evaluate(eval_model, val_loader, criterion, device, use_amp, amp_dtype, aux_cfg)
                        last_val = val
                        print(f"step={global_step} fixed_val_loss={val['loss']:.4f} {val['families']}")
                        debug_metrics = maybe_run_debug_decode(eval_model, val_ds, codec, cfg, global_step, device)
                        last_debug = debug_metrics
                        if _should_save_best(cfg, val, debug_metrics, best_val, best_score):
                            selection_score = _selection_score(cfg, val, debug_metrics)
                            best_val = min(best_val, val["loss"])
                            best_score = max(best_score, selection_score)
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
                                debug_metrics=debug_metrics,
                                selection_metric=cfg.get("checkpoint_metric", "fixed_val_loss"),
                                selection_score=selection_score,
                            )
                        if _is_bad_debug_eval(cfg, debug_metrics, global_step):
                            bad_debug_evals += 1
                            print(
                                f"debug_guard bad_eval_count={bad_debug_evals} "
                                f"patience={cfg.get('early_stop', {}).get('patience', 0)}"
                            )
                        else:
                            bad_debug_evals = 0
                        stop_now = _should_early_stop(cfg, bad_debug_evals)
                        stop_tensor.fill_(1 if stop_now else 0)
                    if is_ddp:
                        dist.broadcast(stop_tensor, src=0)
                        stop_training = bool(stop_tensor.item())
                    else:
                        stop_training = stop_now
                    if stop_training:
                        if rank == 0:
                            print(f"early_stop triggered at step={global_step}")
                        break
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
        if stop_training:
            break

    if rank == 0:
        if is_ddp and last_val is not None:
            val = last_val
            debug_metrics = last_debug
            print(f"step={global_step} final_reuse_fixed_val_loss={val['loss']:.4f} {val['families']}")
        else:
            eval_model = model.module if hasattr(model, "module") else model
            val = evaluate(eval_model, val_loader, criterion, device, use_amp, amp_dtype, aux_cfg)
            print(f"step={global_step} final_fixed_val_loss={val['loss']:.4f} {val['families']}")
            debug_metrics = maybe_run_debug_decode(eval_model, val_ds, codec, cfg, global_step, device)
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
            debug_metrics=debug_metrics,
            selection_metric=cfg.get("checkpoint_metric", "fixed_val_loss"),
            selection_score=_selection_score(cfg, val, debug_metrics),
        )
        if (not is_ddp or last_val is None) and _should_save_best(cfg, val, debug_metrics, best_val, best_score):
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
                debug_metrics=debug_metrics,
                selection_metric=cfg.get("checkpoint_metric", "fixed_val_loss"),
                selection_score=_selection_score(cfg, val, debug_metrics),
            )
    if is_ddp:
        dist.destroy_process_group()


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    aux_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model.eval()
    losses = []
    family_accum: dict[str, list[float]] = {}
    aux_enabled = bool(aux_cfg and aux_cfg.get("enabled", False))
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        tokens = batch["tokens"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            model_out = model(features, tokens[:, :-1], return_aux=aux_enabled)
            logits = model_out["logits"] if aux_enabled else model_out
            loss_out = criterion(logits, tokens[:, 1:])
            total_loss = loss_out.loss
            if aux_enabled:
                aux_loss, aux_logs = auxiliary_loss(model_out, batch, device, aux_cfg or {})
                total_loss = total_loss + aux_loss
                for key, value in aux_logs.items():
                    loss_out.family_losses[key] = value
        losses.append(float(total_loss.item()))
        for key, value in loss_out.family_losses.items():
            family_accum.setdefault(key, []).append(value)
    model.train()
    return {
        "loss": sum(losses) / max(1, len(losses)),
        "families": {k: sum(v) / len(v) for k, v in family_accum.items()},
    }


def auxiliary_loss(
    model_out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    device: torch.device,
    aux_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    onset_logits = model_out["onset_logits"]
    frame_logits = model_out["frame_logits"]
    onset_target = batch["onset_targets"].to(device, non_blocking=True)
    frame_target = batch["frame_targets"].to(device, non_blocking=True)
    mask = batch["aux_mask"].to(device, non_blocking=True).unsqueeze(-1).float()

    max_len = min(onset_logits.shape[1], onset_target.shape[1])
    onset_logits = onset_logits[:, :max_len]
    frame_logits = frame_logits[:, :max_len]
    onset_target = onset_target[:, :max_len]
    frame_target = frame_target[:, :max_len]
    mask = mask[:, :max_len]

    onset_pos_weight = torch.full(
        (onset_logits.shape[-1],),
        float(aux_cfg.get("onset_pos_weight", 18.0)),
        device=device,
    )
    frame_pos_weight = torch.full(
        (frame_logits.shape[-1],),
        float(aux_cfg.get("frame_pos_weight", 4.0)),
        device=device,
    )
    onset_raw = F.binary_cross_entropy_with_logits(
        onset_logits,
        onset_target,
        reduction="none",
        pos_weight=onset_pos_weight,
    )
    frame_raw = F.binary_cross_entropy_with_logits(
        frame_logits,
        frame_target,
        reduction="none",
        pos_weight=frame_pos_weight,
    )
    denom = (mask.sum() * onset_logits.shape[-1]).clamp_min(1.0)
    onset_loss = (onset_raw * mask).sum() / denom
    frame_loss = (frame_raw * mask).sum() / denom
    weighted = float(aux_cfg.get("onset_weight", 0.15)) * onset_loss + float(
        aux_cfg.get("frame_weight", 0.05)
    ) * frame_loss
    return weighted, {
        "ONSET_AUX": float(onset_loss.detach().cpu()),
        "FRAME_AUX": float(frame_loss.detach().cpu()),
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
) -> dict[str, Any] | None:
    debug_cfg = cfg.get("debug_decode", {})
    if not debug_cfg or not debug_cfg.get("enabled", False):
        return None

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
            min_time_for_eos=float(debug_cfg.get("min_time_for_eos", 0.5)),
            max_time_seconds=max_time_seconds,
            eos_bias_after_seconds=debug_cfg.get("eos_bias_after_seconds"),
            eos_logit_bias=float(debug_cfg.get("eos_logit_bias", 0.0)),
            eos_bias_after_token_ratio=debug_cfg.get("eos_bias_after_token_ratio"),
            force_eos_on_loop=bool(debug_cfg.get("force_eos_on_loop", False)),
            max_tokens_since_shift=debug_cfg.get("max_tokens_since_shift"),
            max_same_time_events=debug_cfg.get("max_same_time_events"),
            max_same_time_note_ons=debug_cfg.get("max_same_time_note_ons"),
            max_same_time_pitch_repeats=debug_cfg.get("max_same_time_pitch_repeats"),
            max_note_on_rate=debug_cfg.get("max_note_on_rate"),
            note_on_budget_floor=int(debug_cfg.get("note_on_budget_floor", 0)),
            note_on_logit_bias=float(debug_cfg.get("note_on_logit_bias", 0.0)),
            positive_velocity_logit_bias=float(debug_cfg.get("positive_velocity_logit_bias", 0.0)),
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
        natural_eos = decoded.eos_hit and stats.stop_reason == "eos"
        forced_eos = "forced_eos" in stats.stop_reason
        is_loop = _is_loop_stop(stats.stop_reason)
        item = {
            "eos": float(decoded.eos_hit),
            "natural_eos": float(natural_eos),
            "forced_eos": float(forced_eos),
            "loop": float(is_loop),
            "pred_notes": len(decoded.notes),
            "ref_notes": len(ref_notes),
            "note_f1": metric["note_f1"],
            "offset_f1": metric["offset_f1"],
            "invalid_rate": invalid_rate,
            "decode_s": stats.wall_time,
            "tokens_per_second": stats.tokens_per_second,
            "stop_reason": stats.stop_reason,
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
        return None
    eos_rate = sum(x["eos"] for x in items) / len(items)
    natural_eos_rate = sum(x["natural_eos"] for x in items) / len(items)
    forced_eos_rate = sum(x["forced_eos"] for x in items) / len(items)
    loop_rate = sum(x["loop"] for x in items) / len(items)
    note_f1 = sum(x["note_f1"] for x in items) / len(items)
    offset_f1 = sum(x["offset_f1"] for x in items) / len(items)
    pred_notes = sum(x["pred_notes"] for x in items)
    ref_notes = sum(x["ref_notes"] for x in items)
    pred_ref_ratio = pred_notes / max(1, ref_notes)
    invalid_rate = sum(x["invalid_rate"] for x in items) / len(items)
    avg_decode_s = sum(x["decode_s"] for x in items) / len(items)
    avg_tok_s = sum(x["tokens_per_second"] for x in items) / len(items)
    stop_counts: dict[str, int] = {}
    for item in items:
        reason = str(item["stop_reason"])
        stop_counts[reason] = stop_counts.get(reason, 0) + 1
    summary = {
        "eos_hit_rate": eos_rate,
        "natural_eos_rate": natural_eos_rate,
        "forced_eos_rate": forced_eos_rate,
        "loop_rate": loop_rate,
        "pred_ref_ratio": pred_ref_ratio,
        "note_f1": note_f1,
        "offset_f1": offset_f1,
        "pred_notes": pred_notes,
        "ref_notes": ref_notes,
        "invalid_event_rate": invalid_rate,
        "avg_decode_s": avg_decode_s,
        "avg_tokens_per_second": avg_tok_s,
        "stop_reason_counts": stop_counts,
        "items": items,
    }
    print(
        "debug_decode_summary "
        f"step={global_step} eos_hit_rate={eos_rate:.3f} natural_eos_rate={natural_eos_rate:.3f} "
        f"forced_eos_rate={forced_eos_rate:.3f} loop_rate={loop_rate:.3f} "
        f"pred_ref_ratio={pred_ref_ratio:.3f} note_f1={note_f1:.4f} "
        f"offset_f1={offset_f1:.4f} pred_notes={pred_notes} ref_notes={ref_notes} "
        f"invalid={invalid_rate:.4f} tok_s={avg_tok_s:.2f} stops={stop_counts}"
    )
    return summary


def _selection_score(
    cfg: dict[str, Any],
    val: dict[str, Any],
    debug_metrics: dict[str, Any] | None,
) -> float:
    metric = cfg.get("checkpoint_metric", "fixed_val_loss")
    if metric != "debug_score":
        return -float(val["loss"])
    if not debug_metrics:
        return -math.inf
    weights = cfg.get("debug_score_weights", {})
    note_w = float(weights.get("note_f1", 1.0))
    offset_w = float(weights.get("offset_f1", 0.25))
    eos_w = float(weights.get("eos_hit_rate", 0.10))
    natural_eos_w = float(weights.get("natural_eos_rate", 0.0))
    loop_penalty = float(weights.get("loop_rate", 0.20))
    forced_eos_penalty = float(weights.get("forced_eos_rate", 0.0))
    pred_ref_penalty = float(weights.get("pred_ref_penalty", 0.05))
    invalid_penalty = float(weights.get("invalid_event_rate", 0.05))
    pred_ref_ratio = float(debug_metrics.get("pred_ref_ratio", 0.0))
    ratio_error = min(abs(pred_ref_ratio - 1.0), 2.0)
    return (
        note_w * float(debug_metrics.get("note_f1", 0.0))
        + offset_w * float(debug_metrics.get("offset_f1", 0.0))
        + eos_w * float(debug_metrics.get("eos_hit_rate", 0.0))
        + natural_eos_w * float(debug_metrics.get("natural_eos_rate", 0.0))
        - loop_penalty * float(debug_metrics.get("loop_rate", 1.0))
        - forced_eos_penalty * float(debug_metrics.get("forced_eos_rate", 0.0))
        - pred_ref_penalty * ratio_error
        - invalid_penalty * float(debug_metrics.get("invalid_event_rate", 1.0))
    )


def _should_save_best(
    cfg: dict[str, Any],
    val: dict[str, Any],
    debug_metrics: dict[str, Any] | None,
    best_val: float,
    best_score: float,
) -> bool:
    metric = cfg.get("checkpoint_metric", "fixed_val_loss")
    if metric == "debug_score":
        score = _selection_score(cfg, val, debug_metrics)
        print(f"checkpoint_selection metric=debug_score score={score:.5f} best={best_score:.5f}")
        return score > best_score
    return float(val["loss"]) < best_val


def _is_loop_stop(stop_reason: str) -> bool:
    loop_markers = ("loop", "too_many_tokens_without_shift", "no_shift")
    return any(marker in stop_reason for marker in loop_markers)


def _is_bad_debug_eval(
    cfg: dict[str, Any],
    debug_metrics: dict[str, Any] | None,
    global_step: int,
) -> bool:
    early_cfg = cfg.get("early_stop", {})
    if not early_cfg or not early_cfg.get("enabled", False) or debug_metrics is None:
        return False
    if global_step < int(early_cfg.get("min_steps", 0)):
        return False

    loop_rate = float(debug_metrics.get("loop_rate", 1.0))
    natural_eos_rate = float(debug_metrics.get("natural_eos_rate", debug_metrics.get("eos_hit_rate", 0.0)))
    pred_ref_ratio = float(debug_metrics.get("pred_ref_ratio", 0.0))
    note_f1 = float(debug_metrics.get("note_f1", 0.0))

    if loop_rate > float(early_cfg.get("max_loop_rate", 1.0)):
        return True
    if natural_eos_rate < float(early_cfg.get("min_natural_eos_rate", 0.0)):
        return True
    if pred_ref_ratio < float(early_cfg.get("min_pred_ref_ratio", 0.0)):
        return True
    if pred_ref_ratio > float(early_cfg.get("max_pred_ref_ratio", 999.0)):
        return True
    min_note_f1 = early_cfg.get("min_note_f1")
    if min_note_f1 is not None and note_f1 < float(min_note_f1):
        return True
    return False


def _should_early_stop(cfg: dict[str, Any], bad_debug_evals: int) -> bool:
    early_cfg = cfg.get("early_stop", {})
    if not early_cfg or not early_cfg.get("enabled", False):
        return False
    return bad_debug_evals >= int(early_cfg.get("patience", 2))


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
    debug_metrics: dict[str, Any] | None = None,
    selection_metric: str = "fixed_val_loss",
    selection_score: float | None = None,
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
            "debug_metrics": debug_metrics,
            "selection_metric": selection_metric,
            "selection_score": selection_score,
            "train_config": train_cfg,
        },
        path,
    )
    print(f"saved {path}")


def load_model_weights(
    path,
    model,
    device,
    rank: int,
    codec: EventCodec | None = None,
    skip_families: set[str] | None = None,
) -> None:
    ckpt = torch.load(path, map_location=device)
    module = model.module if hasattr(model, "module") else model
    current = module.state_dict()
    source_state = dict(ckpt["model"])
    remapped = _remap_vocab_rows(
        source_state,
        current,
        ckpt.get("codec_config"),
        codec,
        skip_families=skip_families,
    )
    compatible = {}
    skipped = []
    for key, value in remapped.items():
        if key in current and current[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)
    incompatible = module.load_state_dict(compatible, strict=False)
    if rank == 0:
        print(f"initialized model weights from {path} at source_step={ckpt.get('step', 'unknown')}")
        if skipped:
            preview = skipped[:16]
            suffix = "..." if len(skipped) > len(preview) else ""
            print(f"init_skipped_shape_or_missing={preview}{suffix}")
        if incompatible.missing_keys:
            print(f"init_missing_keys={incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"init_unexpected_keys={incompatible.unexpected_keys}")


def _remap_vocab_rows(
    source_state: dict[str, torch.Tensor],
    current_state: dict[str, torch.Tensor],
    source_codec_cfg: dict[str, Any] | None,
    target_codec: EventCodec | None,
    skip_families: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    if not source_codec_cfg or target_codec is None:
        return source_state
    source_codec = EventCodec(**source_codec_cfg)
    remapped = dict(source_state)
    skip_families = {str(f).upper() for f in (skip_families or set())}
    shared = [
        (token, source_codec.token_to_id[token], target_id)
        for token, target_id in target_codec.token_to_id.items()
        if token in source_codec.token_to_id
        and target_codec.token_family(token).upper() not in skip_families
    ]
    for key in ("decoder.embedding.weight", "decoder.output.weight", "decoder.output.bias"):
        if key not in source_state or key not in current_state:
            continue
        src = source_state[key]
        dst = current_state[key].clone()
        if src.ndim != dst.ndim:
            continue
        if src.ndim == 2 and src.shape[1] != dst.shape[1]:
            continue
        if src.ndim == 1:
            for _, src_id, dst_id in shared:
                if src_id < src.shape[0] and dst_id < dst.shape[0]:
                    dst[dst_id] = src[src_id]
        else:
            for _, src_id, dst_id in shared:
                if src_id < src.shape[0] and dst_id < dst.shape[0]:
                    dst[dst_id].copy_(src[src_id])
        remapped[key] = dst
    return remapped


def load_training_state(path, model, optimizer, scheduler, scaler, device, rank: int) -> tuple[int, float, float]:
    ckpt = torch.load(path, map_location=device)
    module = model.module if hasattr(model, "module") else model
    module.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    if rank == 0:
        print(f"resumed {path} at step={ckpt.get('step', 0)}")
    return (
        int(ckpt.get("step", 0)),
        float(ckpt.get("val_loss") or math.inf),
        float(ckpt.get("selection_score") if ckpt.get("selection_score") is not None else -math.inf),
    )


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
