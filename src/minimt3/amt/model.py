from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from minimt3.model.encoder import AudioEncoder


@dataclass
class DenseAMTConfig:
    n_mels: int = 128
    d_model: int = 256
    encoder_layers: int = 4
    nhead: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1
    conv_channels: int = 256
    head_hidden: int = 256


class DenseAMT(nn.Module):
    """Onsets-and-frames style AMT model for piano-only transcription."""

    def __init__(self, config: DenseAMTConfig):
        super().__init__()
        self.config = config
        self.encoder = AudioEncoder(
            n_mels=config.n_mels,
            d_model=config.d_model,
            conv_channels=config.conv_channels,
            layers=config.encoder_layers,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
        )
        self.shared = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.head_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden, config.head_hidden),
            nn.GELU(),
        )
        self.onset_head = nn.Linear(config.head_hidden, 88)
        self.frame_head = nn.Linear(config.head_hidden, 88)
        self.offset_head = nn.Linear(config.head_hidden, 88)
        self.velocity_head = nn.Linear(config.head_hidden, 88)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        memory = self.encoder(features)
        hidden = self.shared(memory)
        return {
            "onset_logits": self.onset_head(hidden),
            "frame_logits": self.frame_head(hidden),
            "offset_logits": self.offset_head(hidden),
            "velocity_logits": self.velocity_head(hidden),
        }
