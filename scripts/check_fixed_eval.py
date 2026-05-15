#!/usr/bin/env python
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from minimt3.audio.features import LogMelConfig
from minimt3.data import Collator, MaestroDataset
from minimt3.model.loss import WeightedSeq2SeqLoss
from minimt3.pipeline import build_codec, build_model
from minimt3.utils import read_yaml, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Check fixed validation loss determinism.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    cfg = read_yaml(args.config)
    model_cfg = read_yaml(cfg["model_config"])
    seed_everything(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = build_codec(model_cfg)
    ds = MaestroDataset(
        cfg.get("val_metadata", cfg["metadata"]),
        split=cfg.get("val_split", "validation"),
        codec=codec,
        feature_config=LogMelConfig(**model_cfg.get("audio", {})),
        train_seconds=float(cfg.get("val_seconds", cfg.get("train_seconds", 20.0))),
        max_items=cfg.get("val_max_items", 64),
        sampling="fixed",
    )
    loader = DataLoader(ds, batch_size=int(cfg.get("eval_batch_size", 1)), collate_fn=Collator(codec.pad_id))
    model = build_model(model_cfg, codec).to(device)
    criterion = WeightedSeq2SeqLoss(codec, label_smoothing=float(cfg.get("label_smoothing", 0.05)))
    dtype = torch.bfloat16 if cfg.get("precision", "bf16") == "bf16" else torch.float16
    use_amp = torch.cuda.is_available() and cfg.get("precision", "bf16") in {"bf16", "fp16"}
    first = evaluate(model, loader, criterion, device, use_amp, dtype)["loss"]
    second = evaluate(model, loader, criterion, device, use_amp, dtype)["loss"]
    delta = abs(first - second)
    print({"first": first, "second": second, "delta": delta})
    if delta > args.tolerance:
        raise SystemExit(f"fixed eval loss changed by {delta}, above tolerance {args.tolerance}")


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp: bool, amp_dtype: torch.dtype) -> dict:
    model.eval()
    losses = []
    for batch in loader:
        features = batch["features"].to(device)
        tokens = batch["tokens"].to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(features, tokens[:, :-1])
            losses.append(float(criterion(logits, tokens[:, 1:]).loss.item()))
    return {"loss": sum(losses) / max(1, len(losses))}


if __name__ == "__main__":
    main()
