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
    predict_pedal: bool = False
    predict_duration: bool = False
    onset_conditioned_frame: bool = False
    position_encoding: str = "none"
    max_positions: int = 4096
    recurrent_layers: int = 0
    recurrent_hidden: int = 128
    separate_head_towers: bool = False


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
            position_encoding=config.position_encoding,
            max_positions=config.max_positions,
        )
        if config.recurrent_layers > 0:
            self.temporal = nn.GRU(
                input_size=config.d_model,
                hidden_size=config.recurrent_hidden,
                num_layers=config.recurrent_layers,
                batch_first=True,
                bidirectional=True,
                dropout=config.dropout if config.recurrent_layers > 1 else 0.0,
            )
            self.temporal_proj = nn.Linear(config.recurrent_hidden * 2, config.d_model)
            nn.init.zeros_(self.temporal_proj.weight)
            nn.init.zeros_(self.temporal_proj.bias)
        else:
            self.temporal = None
            self.temporal_proj = None
        self.shared = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.head_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden, config.head_hidden),
            nn.GELU(),
        )
        if config.separate_head_towers:
            self.onset_tower = _head_tower(config.head_hidden, config.dropout)
            self.frame_tower = _head_tower(config.head_hidden, config.dropout)
            self.offset_tower = _head_tower(config.head_hidden, config.dropout)
            self.velocity_tower = _head_tower(config.head_hidden, config.dropout)
        else:
            self.onset_tower = nn.Identity()
            self.frame_tower = nn.Identity()
            self.offset_tower = nn.Identity()
            self.velocity_tower = nn.Identity()
        self.onset_head = nn.Linear(config.head_hidden, 88)
        if config.onset_conditioned_frame:
            self.frame_conditioner = nn.Sequential(
                nn.Linear(config.head_hidden + 88, config.head_hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.head_hidden, config.head_hidden),
            )
            nn.init.zeros_(self.frame_conditioner[-1].weight)
            nn.init.zeros_(self.frame_conditioner[-1].bias)
            self.frame_head = nn.Linear(config.head_hidden, 88)
        else:
            self.frame_conditioner = None
            self.frame_head = nn.Linear(config.head_hidden, 88)
        self.offset_head = nn.Linear(config.head_hidden, 88)
        self.velocity_head = nn.Linear(config.head_hidden, 88)
        self.pedal_head = nn.Linear(config.head_hidden, 1) if config.predict_pedal else None
        self.duration_head = nn.Linear(config.head_hidden, 88) if config.predict_duration else None
        if self.duration_head is not None:
            nn.init.constant_(self.duration_head.bias, -2.2)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        memory = self.encoder(features)
        if self.temporal is not None and self.temporal_proj is not None:
            temporal, _ = self.temporal(memory)
            memory = memory + self.temporal_proj(temporal)
        hidden = self.shared(memory)
        onset_hidden = self.onset_tower(hidden)
        frame_hidden = self.frame_tower(hidden)
        offset_hidden = self.offset_tower(hidden)
        velocity_hidden = self.velocity_tower(hidden)
        onset_logits = self.onset_head(onset_hidden)
        if self.frame_conditioner is not None:
            onset_prob = torch.sigmoid(onset_logits)
            frame_hidden = frame_hidden + self.frame_conditioner(torch.cat([frame_hidden, onset_prob], dim=-1))
        out = {
            "onset_logits": onset_logits,
            "frame_logits": self.frame_head(frame_hidden),
            "offset_logits": self.offset_head(offset_hidden),
            "velocity_logits": self.velocity_head(velocity_hidden),
        }
        if self.pedal_head is not None:
            out["pedal_logits"] = self.pedal_head(hidden)
        if self.duration_head is not None:
            out["duration_logits"] = self.duration_head(hidden)
        return out


def _head_tower(hidden: int, dropout: float) -> nn.Module:
    return ResidualHeadTower(hidden, dropout)


class ResidualHeadTower(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)
