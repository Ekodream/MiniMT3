from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        return x + self.pe[:, offset : offset + x.shape[1]]


class CachedDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        causal_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        y = self.norm1(x)
        y, _ = self.self_attn(
            y,
            y,
            y,
            attn_mask=causal_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
            is_causal=causal_mask is not None,
        )
        x = residual + self.dropout(y)

        residual = x
        y = self.norm2(x)
        y, _ = self.cross_attn(y, memory, memory, need_weights=False)
        x = residual + self.dropout(y)

        residual = x
        y = self.norm3(x)
        y = self.linear2(self.dropout_ff(F.gelu(self.linear1(y))))
        return residual + self.dropout(y)

    def step(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        cache: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        residual = x
        y = self.norm1(x)
        if cache and "self_k" in cache:
            k = torch.cat([cache["self_k"], y], dim=1)
            v = torch.cat([cache["self_v"], y], dim=1)
        else:
            k = y
            v = y
        y, _ = self.self_attn(y, k, v, need_weights=False)
        x = residual + self.dropout(y)

        residual = x
        y = self.norm2(x)
        y, _ = self.cross_attn(y, memory, memory, need_weights=False)
        x = residual + self.dropout(y)

        residual = x
        y = self.norm3(x)
        y = self.linear2(F.gelu(self.linear1(y)))
        new_cache = {"self_k": k.detach(), "self_v": v.detach()}
        return residual + self.dropout(y), new_cache


class EventDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        layers: int = 4,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [CachedDecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        tgt_key_padding_mask = tokens.eq(self.pad_id)
        seq_len = tokens.shape[1]
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=tokens.device),
            diagonal=1,
        )
        x = self.pos(self.embedding(tokens))
        for layer in self.layers:
            x = layer(x, memory, causal_mask=causal_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return self.output(self.norm(x))

    def init_cache(self, batch_size: int, device: torch.device) -> list[dict[str, torch.Tensor]]:
        del batch_size, device
        return [{} for _ in self.layers]

    def step(
        self,
        token: torch.Tensor,
        memory: torch.Tensor,
        cache: list[dict[str, torch.Tensor]] | None,
        position: int,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        if token.ndim == 1:
            token = token.unsqueeze(1)
        if cache is None:
            cache = self.init_cache(token.shape[0], token.device)
        x = self.pos(self.embedding(token), offset=position)
        new_cache: list[dict[str, torch.Tensor]] = []
        for layer, layer_cache in zip(self.layers, cache):
            x, updated = layer.step(x, memory, layer_cache)
            new_cache.append(updated)
        logits = self.output(self.norm(x))[:, -1]
        return logits, new_cache


DecoderCache = list[dict[str, Any]]
