#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from minimt3.pipeline import export_transcription, load_checkpoint, load_infer_config, transcribe_audio


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MiniMT3-Piano inference.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="outputs/demo")
    parser.add_argument("--config", default="configs/infer_relative.yaml")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    infer_cfg = load_infer_config(args.config)
    model, codec, model_cfg = load_checkpoint(args.ckpt, device)
    notes, pedals, debug = transcribe_audio(args.audio, model, codec, model_cfg, infer_cfg, device)
    stem = Path(args.audio).stem
    paths = export_transcription(notes, pedals, args.out, stem, infer_cfg, debug)
    print(f"Transcribed {len(notes)} notes and {len(pedals)} pedal regions.")
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
