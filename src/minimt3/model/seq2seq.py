from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from minimt3.model.decoder import EventDecoder
from minimt3.model.encoder import AudioEncoder


@dataclass
class MiniMT3Config:
    n_mels: int = 128
    vocab_size: int = 294
    pad_id: int = 0
    d_model: int = 256
    encoder_layers: int = 4
    decoder_layers: int = 4
    nhead: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1
    conv_channels: int = 256


class MiniMT3(nn.Module):
    def __init__(self, config: MiniMT3Config):
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
        self.decoder = EventDecoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            layers=config.decoder_layers,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            pad_id=config.pad_id,
        )
        self.onset_head = nn.Linear(config.d_model, 88)
        self.frame_head = nn.Linear(config.d_model, 88)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features)

    def forward(
        self,
        features: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        memory = self.encode(features)
        logits = self.decoder(decoder_input_ids, memory)
        if not return_aux:
            return logits
        return {
            "logits": logits,
            "onset_logits": self.onset_head(memory),
            "frame_logits": self.frame_head(memory),
        }

    def decode_step(
        self,
        token: torch.Tensor,
        memory: torch.Tensor,
        cache: list[dict[str, torch.Tensor]] | None,
        position: int,
        max_length: int | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        return self.decoder.step(token, memory, cache, position, max_length=max_length)
