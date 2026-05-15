#!/usr/bin/env python
from __future__ import annotations

import argparse

from minimt3.audio.features import LogMelConfig
from minimt3.data import MaestroDataset, summarize_token_targets
from minimt3.pipeline import build_codec
from minimt3.utils import read_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect target token family distribution.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--sampling", choices=["random", "fixed"], default="fixed")
    parser.add_argument("--model_config", default="configs/model.yaml")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--items", type=int, default=64)
    args = parser.parse_args()

    model_cfg = read_yaml(args.model_config)
    codec = build_codec(model_cfg)
    ds = MaestroDataset(
        args.metadata,
        args.split,
        codec,
        LogMelConfig(**model_cfg.get("audio", {})),
        train_seconds=args.seconds,
        max_items=args.items,
        sampling=args.sampling,
    )
    summary = summarize_token_targets(ds, codec, max_items=args.items)
    print(summary)
    if summary["eos_rate"] < 1.0:
        print("WARNING: EOS is missing from at least one target sequence.")
    velocity_ratio = summary["family_ratio"].get("VELOCITY", 0.0)
    if velocity_ratio > 0.45:
        print(f"WARNING: VELOCITY ratio is high ({velocity_ratio:.3f}); check redundant state changes.")


if __name__ == "__main__":
    main()
