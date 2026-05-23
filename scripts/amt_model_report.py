#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from minimt3.amt.analysis import manifest_size, model_parameter_count
from minimt3.amt.model import DenseAMT, DenseAMTConfig
from minimt3.utils import read_yaml, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Report dense-AMT parameter counts and manifest sizes.")
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--ckpt", action="append", default=[])
    parser.add_argument("--json_out")
    args = parser.parse_args()

    rows = []
    for config_path in args.config:
        cfg = read_yaml(config_path)
        rows.append(_report_config(config_path, cfg))
    for ckpt_path in args.ckpt:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        rows.append(_report_config(ckpt_path, ckpt["config"], checkpoint_step=ckpt.get("step")))
    for row in rows:
        print(
            "amt_model_report "
            f"name={row['name']} params={row['param_count'] / 1e6:.2f}M "
            f"train_manifest_size={row.get('train_manifest_size')} "
            f"val_manifest_size={row.get('val_manifest_size')} "
            f"global_batch={row.get('global_batch')} max_steps={row.get('max_steps')} "
            f"checkpoint_step={row.get('checkpoint_step')}",
            flush=True,
        )
    if args.json_out:
        write_json(args.json_out, {"items": rows})


def _report_config(path: str, cfg: dict[str, Any], checkpoint_step: int | None = None) -> dict[str, Any]:
    model = DenseAMT(DenseAMTConfig(**cfg.get("model", {})))
    world_size = int(cfg.get("world_size", cfg.get("nproc_per_node", 1)) or 1)
    batch_size = int(cfg.get("batch_size", 0) or 0)
    return {
        "name": Path(path).stem,
        "path": path,
        "param_count": model_parameter_count(model),
        "architecture": cfg.get("model", {}).get("architecture", "transformer"),
        "train_manifest": cfg.get("train_manifest"),
        "val_manifest": cfg.get("val_manifest"),
        "train_manifest_size": manifest_size(cfg.get("train_manifest")),
        "val_manifest_size": manifest_size(cfg.get("val_manifest")),
        "batch_size": batch_size,
        "world_size": world_size,
        "global_batch": batch_size * max(1, world_size),
        "max_steps": cfg.get("max_steps"),
        "checkpoint_step": checkpoint_step,
        "output_dir": cfg.get("output_dir"),
    }


if __name__ == "__main__":
    main()
