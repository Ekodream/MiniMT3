from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torchaudio


def load_audio(
    path: str | Path,
    sample_rate: int = 16000,
    mono: bool = True,
    offset_seconds: float = 0.0,
    duration_seconds: float | None = None,
) -> torch.Tensor:
    """Load audio as a float tensor shaped [channels, samples]."""
    if (offset_seconds > 0.0 or duration_seconds is not None) and str(path).lower().endswith(".wav"):
        try:
            waveform, source_sr = _load_audio_with_scipy(path, offset_seconds, duration_seconds, mmap=True)
            waveform = waveform.float()
            if mono and waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if source_sr != sample_rate:
                waveform = torchaudio.functional.resample(waveform, source_sr, sample_rate)
            peak = waveform.abs().max()
            if peak > 1.0:
                waveform = waveform / peak
            return waveform
        except Exception:
            pass
    try:
        waveform, source_sr = torchaudio.load(str(path))
        if offset_seconds > 0.0 or duration_seconds is not None:
            start = max(0, int(round(float(offset_seconds) * source_sr)))
            end = waveform.shape[-1]
            if duration_seconds is not None:
                end = min(end, start + max(1, int(math.ceil(float(duration_seconds) * source_sr))))
            waveform = waveform[:, start:end]
    except ImportError:
        waveform, source_sr = _load_audio_with_scipy(path, offset_seconds, duration_seconds)
    waveform = waveform.float()
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if source_sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_sr, sample_rate)
    peak = waveform.abs().max()
    if peak > 1.0:
        waveform = waveform / peak
    return waveform


def _load_audio_with_scipy(
    path: str | Path,
    offset_seconds: float = 0.0,
    duration_seconds: float | None = None,
    mmap: bool = False,
) -> tuple[torch.Tensor, int]:
    from scipy.io import wavfile

    source_sr, data = wavfile.read(str(path), mmap=mmap)
    if offset_seconds > 0.0 or duration_seconds is not None:
        start = max(0, int(round(float(offset_seconds) * source_sr)))
        end = data.shape[0]
        if duration_seconds is not None:
            end = min(end, start + max(1, int(math.ceil(float(duration_seconds) * source_sr))))
        data = data[start:end]
    if data.ndim == 1:
        data = data[:, None]
    original_dtype = data.dtype
    data = data.astype(np.float32)
    if np.issubdtype(original_dtype, np.integer):
        max_value = np.iinfo(original_dtype).max
        data = data / float(max_value)
    elif data.max(initial=0.0) > 1.0 or data.min(initial=0.0) < -1.0:
        peak = max(abs(float(data.max(initial=0.0))), abs(float(data.min(initial=0.0))), 1.0)
        data = data / peak
    waveform = torch.from_numpy(data.T.copy())
    return waveform, int(source_sr)


def save_audio(path: str | Path, waveform: torch.Tensor, sample_rate: int = 16000) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), waveform.detach().cpu(), sample_rate)
