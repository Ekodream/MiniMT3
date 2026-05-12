from __future__ import annotations

from pathlib import Path

import torch
import torchaudio


def load_audio(path: str | Path, sample_rate: int = 16000, mono: bool = True) -> torch.Tensor:
    """Load audio as a float tensor shaped [channels, samples]."""
    waveform, source_sr = torchaudio.load(str(path))
    waveform = waveform.float()
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if source_sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_sr, sample_rate)
    peak = waveform.abs().max()
    if peak > 1.0:
        waveform = waveform / peak
    return waveform


def save_audio(path: str | Path, waveform: torch.Tensor, sample_rate: int = 16000) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), waveform.detach().cpu(), sample_rate)
