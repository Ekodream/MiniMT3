from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from minimt3.model.encoder import AudioEncoder


@dataclass
class DenseAMTConfig:
    architecture: str = "transformer"
    n_mels: int = 128
    d_model: int = 256
    encoder_layers: int = 4
    nhead: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1
    conv_channels: int = 256
    conv_strides: tuple[int, int] | list[int] = (2, 2)
    head_hidden: int = 256
    predict_pedal: bool = False
    predict_duration: bool = False
    predict_time_shifts: bool = False
    onset_conditioned_frame: bool = False
    position_encoding: str = "none"
    max_positions: int = 4096
    recurrent_layers: int = 0
    recurrent_hidden: int = 128
    separate_head_towers: bool = False
    extra_context_layers: int = 0
    extra_context_dim_feedforward: int = 1024
    adapter_context_layers: int = 0
    adapter_context_dim_feedforward: int = 2048
    acoustic_hidden: int = 768
    acoustic_rnn_hidden: int = 256
    acoustic_channels: tuple[int, int, int, int] | list[int] = (48, 64, 96, 128)
    acoustic_context_layers: int = 0
    acoustic_context_dropout: float = 0.1
    acoustic_attention_layers: int = 0
    acoustic_attention_heads: int = 8
    acoustic_attention_dim_feedforward: int = 2048


class DenseAMT(nn.Module):
    """Onsets-and-frames style AMT model for piano-only transcription."""

    def __init__(self, config: DenseAMTConfig):
        super().__init__()
        self.config = config
        if config.architecture == "crnn_ensemble":
            self.backend = ByteDanceStyleCRNN(config)
            return
        if config.architecture != "transformer":
            raise ValueError("DenseAMTConfig.architecture must be 'transformer' or 'crnn_ensemble'")
        self.backend = None
        self.encoder = AudioEncoder(
            n_mels=config.n_mels,
            d_model=config.d_model,
            conv_channels=config.conv_channels,
            conv_strides=config.conv_strides,
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
        self.extra_context = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.d_model,
                    nhead=config.nhead,
                    dim_feedforward=config.extra_context_dim_feedforward,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.extra_context_layers)
            ]
        )
        if self.extra_context:
            self.extra_context_scale = nn.Parameter(torch.zeros(len(self.extra_context)))
        else:
            self.register_parameter("extra_context_scale", None)
        self.adapter_context = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.d_model,
                    nhead=config.nhead,
                    dim_feedforward=config.adapter_context_dim_feedforward,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.adapter_context_layers)
            ]
        )
        if self.adapter_context:
            self.adapter_context_scale = nn.Parameter(torch.zeros(len(self.adapter_context)))
        else:
            self.register_parameter("adapter_context_scale", None)
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
        self.onset_shift_head = nn.Linear(config.head_hidden, 88) if config.predict_time_shifts else None
        self.offset_shift_head = nn.Linear(config.head_hidden, 88) if config.predict_time_shifts else None
        if self.duration_head is not None:
            nn.init.constant_(self.duration_head.bias, -2.2)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.backend is not None:
            return self.backend(features)
        memory = self.encoder(features)
        if self.temporal is not None and self.temporal_proj is not None:
            temporal, _ = self.temporal(memory)
            memory = memory + self.temporal_proj(temporal)
        if self.extra_context:
            for idx, layer in enumerate(self.extra_context):
                refined = layer(memory)
                memory = memory + self.extra_context_scale[idx] * (refined - memory)
        if self.adapter_context:
            for idx, layer in enumerate(self.adapter_context):
                refined = layer(memory)
                memory = memory + self.adapter_context_scale[idx] * (refined - memory)
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
        if self.onset_shift_head is not None and self.offset_shift_head is not None:
            out["onset_shift_logits"] = self.onset_shift_head(onset_hidden)
            out["offset_shift_logits"] = self.offset_shift_head(offset_hidden)
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


class ByteDanceStyleCRNN(nn.Module):
    """Multi-tower CRNN inspired by ByteDance high-resolution piano transcription."""

    def __init__(self, config: DenseAMTConfig):
        super().__init__()
        self.config = config
        channels = tuple(int(x) for x in config.acoustic_channels)
        if len(channels) != 4:
            raise ValueError("acoustic_channels must contain four channel sizes")
        self.bn0 = nn.BatchNorm2d(config.n_mels, momentum=0.01)
        tower_kwargs = {
            "n_mels": config.n_mels,
            "channels": channels,
            "hidden": config.acoustic_hidden,
            "rnn_hidden": config.acoustic_rnn_hidden,
            "context_layers": config.acoustic_context_layers,
            "context_dropout": config.acoustic_context_dropout,
            "attention_layers": config.acoustic_attention_layers,
            "attention_heads": config.acoustic_attention_heads,
            "attention_dim_feedforward": config.acoustic_attention_dim_feedforward,
        }
        self.frame_model = AcousticCRNNTower(classes=88, **tower_kwargs)
        self.onset_model = AcousticCRNNTower(classes=88, **tower_kwargs)
        self.offset_model = AcousticCRNNTower(classes=88, **tower_kwargs)
        self.velocity_model = AcousticCRNNTower(classes=88, **tower_kwargs)
        cond_hidden = int(config.acoustic_rnn_hidden)
        self.onset_condition_gru = nn.GRU(
            input_size=88 * 2,
            hidden_size=cond_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.onset_condition_fc = nn.Linear(cond_hidden * 2, 88)
        self.frame_condition_gru = nn.GRU(
            input_size=88 * 3,
            hidden_size=cond_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.frame_condition_fc = nn.Linear(cond_hidden * 2, 88)
        self.pedal_model = (
            AcousticCRNNTower(classes=1, **tower_kwargs)
            if config.predict_pedal
            else None
        )
        self.onset_shift_head = nn.Linear(cond_hidden * 2, 88) if config.predict_time_shifts else None
        self.offset_shift_head = nn.Linear(config.acoustic_rnn_hidden * 2, 88) if config.predict_time_shifts else None
        self.duration_head = nn.Linear(cond_hidden * 2, 88) if config.predict_duration else None
        self._init_condition_layers()

    def _init_condition_layers(self) -> None:
        nn.init.ones_(self.bn0.weight)
        nn.init.zeros_(self.bn0.bias)
        for gru in (self.onset_condition_gru, self.frame_condition_gru):
            _init_gru(gru)
        for layer in (self.onset_condition_fc, self.frame_condition_fc, self.onset_shift_head, self.offset_shift_head, self.duration_head):
            if layer is not None:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        x = features.transpose(1, 2).unsqueeze(1)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        frame_base_logits, _ = self.frame_model(x)
        onset_base_logits, _ = self.onset_model(x)
        offset_logits, offset_hidden = self.offset_model(x)
        velocity_logits, _ = self.velocity_model(x)

        onset_base_prob = torch.sigmoid(onset_base_logits)
        velocity_prob = torch.sigmoid(velocity_logits)
        onset_cond = torch.cat(
            (onset_base_prob, onset_base_prob.clamp_min(1e-6).sqrt() * velocity_prob.detach()),
            dim=-1,
        )
        onset_hidden, _ = self.onset_condition_gru(onset_cond)
        onset_logits = self.onset_condition_fc(nn.functional.dropout(onset_hidden, p=0.5, training=self.training))

        frame_cond = torch.cat(
            (
                torch.sigmoid(frame_base_logits),
                torch.sigmoid(onset_logits).detach(),
                torch.sigmoid(offset_logits).detach(),
            ),
            dim=-1,
        )
        frame_hidden, _ = self.frame_condition_gru(frame_cond)
        frame_logits = self.frame_condition_fc(nn.functional.dropout(frame_hidden, p=0.5, training=self.training))

        out = {
            "onset_logits": onset_logits,
            "frame_logits": frame_logits,
            "offset_logits": offset_logits,
            "velocity_logits": velocity_logits,
        }
        if self.pedal_model is not None:
            pedal_logits, _ = self.pedal_model(x)
            out["pedal_logits"] = pedal_logits
        if self.onset_shift_head is not None and self.offset_shift_head is not None:
            out["onset_shift_logits"] = self.onset_shift_head(onset_hidden)
            out["offset_shift_logits"] = self.offset_shift_head(offset_hidden)
        if self.duration_head is not None:
            out["duration_logits"] = self.duration_head(frame_hidden)
        return out


class AcousticCRNNTower(nn.Module):
    def __init__(
        self,
        n_mels: int,
        classes: int,
        channels: tuple[int, int, int, int],
        hidden: int,
        rnn_hidden: int,
        context_layers: int = 0,
        context_dropout: float = 0.1,
        attention_layers: int = 0,
        attention_heads: int = 8,
        attention_dim_feedforward: int = 2048,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        in_channels = 1
        for out_channels in channels:
            self.blocks.append(ConvBlock2d(in_channels, out_channels))
            in_channels = out_channels
        freq_bins = max(1, int(n_mels) // (2 ** len(channels)))
        self.fc = nn.Linear(channels[-1] * freq_bins, hidden, bias=False)
        self.bn = nn.BatchNorm1d(hidden, momentum=0.01)
        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=rnn_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.context = (
            TemporalResidualGRU(rnn_hidden * 2, rnn_hidden, context_layers, context_dropout)
            if context_layers > 0
            else None
        )
        self.attention_context = (
            TemporalResidualAttention(
                dim=rnn_hidden * 2,
                layers=attention_layers,
                heads=attention_heads,
                dim_feedforward=attention_dim_feedforward,
                dropout=context_dropout,
            )
            if attention_layers > 0
            else None
        )
        self.out = nn.Linear(rnn_hidden * 2, classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.ones_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)
        _init_gru(self.gru)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x = block(x)
            x = nn.functional.dropout(x, p=0.2, training=self.training)
        x = x.transpose(1, 2).flatten(2)
        x = self.fc(x)
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        x = nn.functional.relu(x)
        x = nn.functional.dropout(x, p=0.5, training=self.training)
        hidden, _ = self.gru(x)
        if self.context is not None:
            hidden = self.context(hidden)
        if self.attention_context is not None:
            hidden = self.attention_context(hidden)
        hidden = nn.functional.dropout(hidden, p=0.5, training=self.training)
        return self.out(hidden), hidden


class TemporalResidualGRU(nn.Module):
    """Lightweight temporal context after the acoustic tower GRU."""

    def __init__(self, dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=hidden,
            num_layers=int(layers),
            batch_first=True,
            bidirectional=True,
            dropout=float(dropout) if int(layers) > 1 else 0.0,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.scale = nn.Parameter(torch.tensor(0.5))
        _init_gru(self.gru)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        refined, _ = self.gru(self.norm(x))
        return x + self.scale * self.dropout(refined)


class TemporalResidualAttention(nn.Module):
    """Residual self-attention context on top of the CRNN tower hidden states."""

    def __init__(self, dim: int, layers: int, heads: int, dim_feedforward: int, dropout: float):
        super().__init__()
        layers = int(layers)
        heads = max(1, int(heads))
        if dim % heads != 0:
            heads = max(1, _largest_divisor_at_most(dim, heads))
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=heads,
                    dim_feedforward=int(dim_feedforward),
                    dropout=float(dropout),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(layers)
            ]
        )
        self.scale = nn.Parameter(torch.zeros(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, layer in enumerate(self.layers):
            refined = layer(x)
            x = x + self.scale[idx] * (refined - x)
        return x


def _largest_divisor_at_most(value: int, limit: int) -> int:
    for candidate in range(max(1, limit), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


class ConvBlock2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.01)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.01)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.ones_(self.bn1.weight)
        nn.init.zeros_(self.bn1.bias)
        nn.init.ones_(self.bn2.weight)
        nn.init.zeros_(self.bn2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.relu_(self.bn1(self.conv1(x)))
        x = nn.functional.relu_(self.bn2(self.conv2(x)))
        return nn.functional.avg_pool2d(x, kernel_size=(1, 2))


def _init_gru(gru: nn.GRU) -> None:
    for name, param in gru.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(param)
        elif "weight_hh" in name:
            nn.init.orthogonal_(param)
        elif "bias" in name:
            nn.init.zeros_(param)
