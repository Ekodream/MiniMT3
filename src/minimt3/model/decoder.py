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
        position: int,
        max_length: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cache = cache or {}
        residual = x
        y = self.norm1(x)
        q, k_new, v_new = _project_self_qkv(self.self_attn, y)
        cache = _append_cache(cache, k_new, v_new, position, max_length=max_length)
        k = cache["self_k"][:, :, : position + 1]
        v = cache["self_v"][:, :, : position + 1]
        y = _scaled_dot_product(self.self_attn, q, k, v)
        x = residual + self.dropout(y)

        residual = x
        y = self.norm2(x)
        q = _project_q(self.cross_attn, y)
        if "cross_k" not in cache or "cross_v" not in cache:
            cache["cross_k"], cache["cross_v"] = _project_kv(self.cross_attn, memory)
        y = _scaled_dot_product(self.cross_attn, q, cache["cross_k"], cache["cross_v"])
        x = residual + self.dropout(y)

        residual = x
        y = self.norm3(x)
        y = self.linear2(F.gelu(self.linear1(y)))
        return residual + self.dropout(y), cache


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

    def init_cache(
        self,
        batch_size: int,
        device: torch.device,
        max_length: int | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        del batch_size, device
        cache: list[dict[str, torch.Tensor]] = [{} for _ in self.layers]
        if max_length is not None:
            for layer_cache in cache:
                layer_cache["max_length"] = torch.tensor(max_length)
        return cache

    def step(
        self,
        token: torch.Tensor,
        memory: torch.Tensor,
        cache: list[dict[str, torch.Tensor]] | None,
        position: int,
        max_length: int | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        if token.ndim == 1:
            token = token.unsqueeze(1)
        if cache is None:
            cache = self.init_cache(token.shape[0], token.device, max_length=max_length)
        x = self.pos(self.embedding(token), offset=position)
        new_cache: list[dict[str, torch.Tensor]] = []
        for layer, layer_cache in zip(self.layers, cache):
            x, updated = layer.step(x, memory, layer_cache, position=position, max_length=max_length)
            new_cache.append(updated)
        logits = self.output(self.norm(x))[:, -1]
        return logits, new_cache


DecoderCache = list[dict[str, Any]]


def _split_projection(attn: nn.MultiheadAttention, index: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    embed_dim = attn.embed_dim
    start = index * embed_dim
    end = (index + 1) * embed_dim
    bias = attn.in_proj_bias[start:end] if attn.in_proj_bias is not None else None
    return attn.in_proj_weight[start:end], bias


def _shape_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch, seq_len, embed_dim = x.shape
    head_dim = embed_dim // num_heads
    return x.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, num_heads, seq_len, head_dim = x.shape
    return x.transpose(1, 2).reshape(batch, seq_len, num_heads * head_dim)


def _project_q(attn: nn.MultiheadAttention, x: torch.Tensor) -> torch.Tensor:
    weight, bias = _split_projection(attn, 0)
    return _shape_heads(F.linear(x, weight, bias), attn.num_heads)


def _project_kv(attn: nn.MultiheadAttention, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    k_weight, k_bias = _split_projection(attn, 1)
    v_weight, v_bias = _split_projection(attn, 2)
    return (
        _shape_heads(F.linear(x, k_weight, k_bias), attn.num_heads),
        _shape_heads(F.linear(x, v_weight, v_bias), attn.num_heads),
    )


def _project_self_qkv(
    attn: nn.MultiheadAttention,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = _project_q(attn, x)
    k, v = _project_kv(attn, x)
    return q, k, v


def _scaled_dot_product(
    attn: nn.MultiheadAttention,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    context = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    return attn.out_proj(_merge_heads(context))


def _append_cache(
    cache: dict[str, torch.Tensor],
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    position: int,
    max_length: int | None,
) -> dict[str, torch.Tensor]:
    batch, num_heads, _, head_dim = k_new.shape
    capacity = int(cache.get("self_k", torch.empty(0)).shape[2]) if "self_k" in cache else 0
    if capacity <= position:
        requested = max_length or max(128, (position + 1) * 2)
        new_capacity = max(position + 1, requested, capacity * 2)
        new_k = k_new.new_empty(batch, num_heads, new_capacity, head_dim)
        new_v = v_new.new_empty(batch, num_heads, new_capacity, head_dim)
        if capacity:
            new_k[:, :, :capacity].copy_(cache["self_k"])
            new_v[:, :, :capacity].copy_(cache["self_v"])
        cache["self_k"] = new_k
        cache["self_v"] = new_v
    cache["self_k"][:, :, position : position + 1].copy_(k_new)
    cache["self_v"][:, :, position : position + 1].copy_(v_new)
    return cache
