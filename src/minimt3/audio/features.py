from __future__ import annotations

from dataclasses import dataclass

import torch
import torchaudio


@dataclass
class LogMelConfig:
    sample_rate: int = 16000
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 160
    f_min: float = 30.0
    f_max: float = 8000.0


class LogMelExtractor(torch.nn.Module):
    def __init__(self, config: LogMelConfig):
        super().__init__()
        self.config = config
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        mel = self.mel(waveform)
        mel = self.to_db(mel)
        mel = (mel + 80.0) / 80.0
        return mel.clamp(0.0, 1.0)
