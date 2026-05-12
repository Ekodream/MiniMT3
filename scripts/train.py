#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from minimt3.audio.features import LogMelConfig
from minimt3.data import Collator, MaestroDataset
from minimt3.model.loss import Seq2SeqLoss
from minimt3.pipeline import build_codec, build_model
from minimt3.utils import ensure_dir, read_yaml, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MiniMT3-Piano.")
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    cfg = read_yaml(args.config)
    model_cfg = read_yaml(cfg["model_config"])
    seed_everything(int(cfg.get("seed", 42)))

    is_ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if is_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    rank = dist.get_rank() if is_ddp else 0

    codec = build_codec(model_cfg)
    audio_cfg = LogMelConfig(**model_cfg.get("audio", {}))
    train_ds = MaestroDataset(
        cfg["metadata"],
        split=cfg.get("split", "train"),
        codec=codec,
        feature_config=audio_cfg,
        train_seconds=float(cfg.get("train_seconds", 20.0)),
        max_items=cfg.get("max_items"),
    )
    val_ds = MaestroDataset(
        cfg["metadata"],
        split=cfg.get("val_split", "validation"),
        codec=codec,
        feature_config=audio_cfg,
        train_seconds=float(cfg.get("train_seconds", 20.0)),
        max_items=min(int(cfg.get("max_items") or 64), 64) if cfg.get("max_items") else 64,
    )
    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_ddp else None
    collate = Collator(codec.pad_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("batch_size", 2)),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(cfg.get("num_workers", 2)),
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.get("batch_size", 2)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 2)),
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(model_cfg, codec).to(device)
    if is_ddp:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    criterion = Seq2SeqLoss(codec.pad_id)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-2)),
    )
    precision = cfg.get("precision", "bf16")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_amp = torch.cuda.is_available() and precision in {"bf16", "fp16"}
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available() and precision == "fp16")
    out_dir = ensure_dir(cfg.get("output_dir", "outputs/ckpt"))
    global_step = 0
    best_val = float("inf")
    grad_accum = int(cfg.get("grad_accum", 1))

    for epoch in range(int(cfg.get("epochs", 5))):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        model.train()
        iterator = tqdm(train_loader, disable=rank != 0, desc=f"epoch {epoch + 1}")
        optimizer.zero_grad(set_to_none=True)
        for batch in iterator:
            features = batch["features"].to(device)
            tokens = batch["tokens"].to(device)
            decoder_in = tokens[:, :-1]
            target = tokens[:, 1:]
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(features, decoder_in)
                loss = criterion(logits, target) / grad_accum
            scaler.scale(loss).backward()
            if (global_step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if rank == 0 and global_step % int(cfg.get("log_interval", 20)) == 0:
                iterator.set_postfix(loss=f"{loss.item() * grad_accum:.4f}")
            if rank == 0 and global_step % int(cfg.get("eval_interval", 500)) == 0:
                val_loss = evaluate(model, val_loader, criterion, device, use_amp, amp_dtype)
                print(f"step={global_step} val_loss={val_loss:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(out_dir / "best.pt", model, model_cfg, global_step, val_loss)
            if rank == 0 and global_step % int(cfg.get("save_interval", 500)) == 0:
                save_checkpoint(out_dir / f"step_{global_step}.pt", model, model_cfg, global_step, None)

    if rank == 0:
        val_loss = evaluate(model, val_loader, criterion, device, use_amp, amp_dtype)
        save_checkpoint(out_dir / "last.pt", model, model_cfg, global_step, val_loss)
        if val_loss < best_val:
            save_checkpoint(out_dir / "best.pt", model, model_cfg, global_step, val_loss)
    if is_ddp:
        dist.destroy_process_group()


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp: bool, amp_dtype: torch.dtype) -> float:
    model.eval()
    losses = []
    for batch in loader:
        features = batch["features"].to(device)
        tokens = batch["tokens"].to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(features, tokens[:, :-1])
            losses.append(float(criterion(logits, tokens[:, 1:]).item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def save_checkpoint(path: Path, model, model_cfg: dict, step: int, val_loss: float | None) -> None:
    module = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "model": module.state_dict(),
            "model_config": model_cfg,
            "step": step,
            "val_loss": val_loss,
        },
        path,
    )
    print(f"saved {path}")


if __name__ == "__main__":
    main()
