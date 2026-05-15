from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchaudio


def load_audio(path: str | Path, sample_rate: int = 16000, mono: bool = True) -> torch.Tensor:
    """Load audio as a float tensor shaped [channels, samples]."""
    try:
        waveform, source_sr = torchaudio.load(str(path))
    except ImportError:
        waveform, source_sr = _load_audio_with_scipy(path)
    waveform = waveform.float()
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if source_sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_sr, sample_rate)
    peak = waveform.abs().max()
    if peak > 1.0:
        waveform = waveform / peak
    return waveform


def _load_audio_with_scipy(path: str | Path) -> tuple[torch.Tensor, int]:
    from scipy.io import wavfile

    source_sr, data = wavfile.read(str(path))
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
