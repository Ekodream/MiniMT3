from __future__ import annotations

import math

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
        position_encoding: str = "none",
        max_positions: int = 4096,
    ):
        super().__init__()
        if position_encoding not in {"none", "learned", "sinusoidal"}:
            raise ValueError("position_encoding must be one of: none, learned, sinusoidal")
        self.position_encoding = position_encoding
        self.max_positions = int(max_positions)
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, conv_channels, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(conv_channels, d_model, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )
        if position_encoding == "learned":
            self.position = nn.Embedding(self.max_positions, d_model)
            nn.init.zeros_(self.position.weight)
        elif position_encoding == "sinusoidal":
            self.register_buffer("position", _sinusoidal_positions(self.max_positions, d_model), persistent=False)
        else:
            self.position = None
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
        if self.position is not None:
            if x.shape[1] > self.max_positions:
                raise ValueError(f"sequence length {x.shape[1]} exceeds max_positions={self.max_positions}")
            pos_ids = torch.arange(x.shape[1], device=x.device)
            if self.position_encoding == "learned":
                x = x + self.position(pos_ids).unsqueeze(0)
            else:
                x = x + self.position[: x.shape[1]].unsqueeze(0).to(dtype=x.dtype)
        x = self.transformer(x)
        return self.norm(x)


def _sinusoidal_positions(max_positions: int, d_model: int) -> torch.Tensor:
    position = torch.arange(max_positions, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_positions, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe
