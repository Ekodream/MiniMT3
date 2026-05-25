#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from minimt3.audio.preprocess import load_audio


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ByteDance piano transcription on one audio file.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out_midi", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--offset_seconds", type=float, default=0.0)
    parser.add_argument("--duration_seconds", type=float)
    args = parser.parse_args()

    try:
        from piano_transcription_inference import PianoTranscription, sample_rate
    except Exception as exc:
        raise SystemExit(
            "Missing piano_transcription_inference. Install it with: pip install piano-transcription-inference"
        ) from exc

    out_midi = Path(args.out_midi)
    out_midi.parent.mkdir(parents=True, exist_ok=True)
    waveform = load_audio(
        args.audio,
        sample_rate=int(sample_rate),
        offset_seconds=float(args.offset_seconds),
        duration_seconds=args.duration_seconds,
    )
    audio = waveform.mean(dim=0).detach().cpu().numpy().astype(np.float32, copy=False)
    transcriptor = PianoTranscription(device=str(args.device))
    transcriptor.transcribe(audio, str(out_midi))
    print(f"teacher_audio out_midi={out_midi}", flush=True)


if __name__ == "__main__":
    main()
