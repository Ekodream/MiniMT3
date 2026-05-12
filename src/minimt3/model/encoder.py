from __future__ import annotations

import torch
from torch import nn


class AudioEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int,
        d_model: int = 256,
        conv_channels: int = 256,
        layers: int = 4,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, conv_channels, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(conv_channels, d_model, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.conv(features)
        x = x.transpose(1, 2)
        x = self.transformer(x)
        return self.norm(x)
