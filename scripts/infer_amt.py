#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from minimt3.amt.decode import decode_dense_notes
from minimt3.amt.model import DenseAMT, DenseAMTConfig
from minimt3.audio.features import LogMelConfig, LogMelExtractor
from minimt3.audio.preprocess import load_audio
from minimt3.symbolic.midi_io import write_midi
from minimt3.utils import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense-AMT inference.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="outputs/amt_demo")
    parser.add_argument("--window_seconds", type=float, default=2.0)
    parser.add_argument("--overlap_seconds", type=float, default=0.25)
    parser.add_argument("--onset_threshold", type=float, default=0.45)
    parser.add_argument("--frame_threshold", type=float, default=0.35)
    parser.add_argument("--offset_threshold", type=float, default=0.35)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["config"]
    audio_cfg = LogMelConfig(**cfg.get("audio", {}))
    model = DenseAMT(DenseAMTConfig(**cfg.get("model", {}))).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    waveform = load_audio(args.audio, audio_cfg.sample_rate)
    extractor = LogMelExtractor(audio_cfg).to(device)
    sr = audio_cfg.sample_rate
    total_seconds = waveform.shape[-1] / sr
    step = max(0.1, args.window_seconds - args.overlap_seconds)
    starts = []
    t = 0.0
    while t < total_seconds:
        starts.append(t)
        t += step
        if t + 0.05 >= total_seconds:
            break
    notes = []
    debug = []
    with torch.no_grad():
        for start in starts:
            end = min(total_seconds, start + args.window_seconds)
            segment = waveform[:, int(start * sr) : int(end * sr)]
            features = extractor(segment.to(device))
            out = model(features)
            window_notes = decode_dense_notes(
                out,
                duration=end - start,
                onset_threshold=args.onset_threshold,
                frame_threshold=args.frame_threshold,
                offset_threshold=args.offset_threshold,
            )
            for note in window_notes:
                note.start += start
                note.end += start
            notes.extend(window_notes)
            debug.append({"start": start, "end": end, "notes": len(window_notes)})
    notes.sort(key=lambda n: (n.start, n.pitch, n.end))
    out_dir = ensure_dir(args.out)
    stem = Path(args.audio).stem
    midi_path = write_midi(out_dir / f"{stem}.mid", notes, [])
    write_json(out_dir / f"{stem}_debug.json", {"windows": debug, "notes": len(notes)})
    print(f"notes={len(notes)} midi={midi_path}")


if __name__ == "__main__":
    main()
