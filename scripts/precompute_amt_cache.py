#!/usr/bin/env python
from __future__ import annotations

import argparse

from torch.utils.data import DataLoader
from tqdm import tqdm

from minimt3.amt.data import DenseAMTCollator, DenseAMTDataset
from minimt3.amt.targets import DenseTargetConfig
from minimt3.audio.features import LogMelConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute dense-AMT feature/target cache.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    ds = DenseAMTDataset(
        args.manifest,
        feature_config=LogMelConfig(),
        split=args.split,
        max_items=args.max_items or None,
        cache_dir=args.cache_dir,
        target_config=DenseTargetConfig(),
    )
    loader = DataLoader(
        ds,
        batch_size=16,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=DenseAMTCollator(),
        persistent_workers=args.num_workers > 0,
    )
    for _ in tqdm(loader, desc="precompute"):
        pass
    print(f"cached {len(ds)} items in {args.cache_dir}")


if __name__ == "__main__":
    main()
