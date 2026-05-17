from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from minimt3.decode.constraints import ConstraintState, apply_constraints
from minimt3.model.seq2seq import MiniMT3
from minimt3.symbolic.events import EventCodec


@dataclass
class DecodeStats:
    eos_hit: bool = False
    stop_reason: str = "max_tokens"
    wall_time: float = 0.0
    tokens_per_second: float = 0.0
    topk_snapshots: list[dict] = field(default_factory=list)


@torch.no_grad()
def greedy_decode(
    model: MiniMT3,
    features: torch.Tensor,
    codec: EventCodec,
    max_tokens: int = 4096,
    constrained: bool = True,
    repetition_penalty: float = 1.15,
    loop_window: int = 16,
    loop_repeats: int = 4,
    min_time_for_eos: float = 0.5,
    max_time_seconds: float | None = None,
    eos_bias_after_seconds: float | None = None,
    eos_logit_bias: float = 0.0,
    eos_bias_after_token_ratio: float | None = None,
    force_eos_on_loop: bool = False,
    max_tokens_since_shift: int | None = None,
    max_same_time_events: int | None = None,
    max_same_time_note_ons: int | None = None,
    prefix_tokens: list[int] | None = None,
    return_stats: bool = False,
) -> list[int] | tuple[list[int], DecodeStats]:
    model.eval()
    if features.ndim == 2:
        features = features.unsqueeze(0)
    device = next(model.parameters()).device
    features = features.to(device)
    memory = model.encode(features)
    tokens = [codec.bos_id]
    state = ConstraintState(codec)
    cache = None
    current = torch.tensor([codec.bos_id], dtype=torch.long, device=device)
    stats = DecodeStats()
    started = time.perf_counter()
    position = 0
    for prefix_id in prefix_tokens or []:
        if len(tokens) >= max_tokens:
            break
        _, cache = model.decode_step(current, memory, cache, position, max_length=max_tokens)
        prefix_id = int(prefix_id)
        tokens.append(prefix_id)
        state.update(prefix_id)
        current = torch.tensor([prefix_id], dtype=torch.long, device=device)
        position += 1
    for decode_position in range(max_tokens - len(tokens) + 1):
        logits, cache = model.decode_step(current, memory, cache, position, max_length=max_tokens)
        logits = logits[0]
        if constrained:
            logits = apply_constraints(
                logits,
                state,
                min_time_for_eos=min_time_for_eos,
                max_same_time_events=max_same_time_events,
                max_same_time_note_ons=max_same_time_note_ons,
            )
        logits = _apply_eos_bias(
            logits,
            codec,
            state,
            position=position,
            max_tokens=max_tokens,
            max_time_seconds=max_time_seconds,
            eos_bias_after_seconds=eos_bias_after_seconds,
            eos_logit_bias=eos_logit_bias,
            eos_bias_after_token_ratio=eos_bias_after_token_ratio,
        )
        logits = _apply_repetition_penalty(logits, tokens, repetition_penalty)
        if decode_position < 3:
            values, indices = torch.topk(torch.softmax(logits, dim=-1), k=min(8, logits.numel()))
            stats.topk_snapshots.append(
                {
                    "position": position,
                    "tokens": [codec.token(i) for i in indices.tolist()],
                    "probs": [float(v) for v in values.tolist()],
                }
            )
        next_id = int(torch.argmax(logits).item())
        tokens.append(next_id)
        state.update(next_id)
        if next_id == codec.eos_id:
            stats.eos_hit = True
            stats.stop_reason = "eos"
            break
        if _detect_loop(tokens, loop_window=loop_window, repeats=loop_repeats):
            if force_eos_on_loop and state.current_time >= min_time_for_eos and not state.pending_velocity:
                tokens.append(codec.eos_id)
                stats.eos_hit = True
                stats.stop_reason = "loop_forced_eos"
            else:
                stats.stop_reason = "loop_detected"
            break
        if max_tokens_since_shift is not None and state.tokens_since_shift > max_tokens_since_shift:
            if force_eos_on_loop and state.current_time >= min_time_for_eos and not state.pending_velocity:
                tokens.append(codec.eos_id)
                stats.eos_hit = True
                stats.stop_reason = "no_shift_forced_eos"
            else:
                stats.stop_reason = "too_many_tokens_without_shift"
            break
        if state.repeated_pitch_count > 12 or state.no_time_progress > 64:
            if force_eos_on_loop and state.current_time >= min_time_for_eos and not state.pending_velocity:
                tokens.append(codec.eos_id)
                stats.eos_hit = True
                stats.stop_reason = "state_loop_forced_eos"
            else:
                stats.stop_reason = "state_loop_detected"
            break
        if max_time_seconds is not None and state.current_time >= max_time_seconds:
            if force_eos_on_loop and not state.pending_velocity:
                tokens.append(codec.eos_id)
                stats.eos_hit = True
                stats.stop_reason = "max_time_forced_eos"
            else:
                stats.stop_reason = "max_time"
            break
        current = torch.tensor([next_id], dtype=torch.long, device=device)
        position += 1
    stats.wall_time = time.perf_counter() - started
    stats.tokens_per_second = (len(tokens) - 1) / max(stats.wall_time, 1e-6)
    return (tokens, stats) if return_stats else tokens


def _apply_eos_bias(
    logits: torch.Tensor,
    codec: EventCodec,
    state: ConstraintState,
    position: int,
    max_tokens: int,
    max_time_seconds: float | None,
    eos_bias_after_seconds: float | None,
    eos_logit_bias: float,
    eos_bias_after_token_ratio: float | None,
) -> torch.Tensor:
    if eos_logit_bias <= 0 or state.pending_velocity:
        return logits
    should_bias = False
    if eos_bias_after_seconds is not None and state.current_time >= eos_bias_after_seconds:
        should_bias = True
    if max_time_seconds is not None and state.current_time >= 0.85 * max_time_seconds:
        should_bias = True
    if eos_bias_after_token_ratio is not None and position >= int(max_tokens * eos_bias_after_token_ratio):
        should_bias = True
    if not should_bias:
        return logits
    out = logits.clone()
    out[codec.eos_id] += eos_logit_bias
    return out


@torch.no_grad()
def beam_decode(
    model: MiniMT3,
    features: torch.Tensor,
    codec: EventCodec,
    beam_size: int = 4,
    max_tokens: int = 4096,
    constrained: bool = True,
    repetition_penalty: float = 1.1,
    return_stats: bool = False,
) -> list[int] | tuple[list[int], DecodeStats]:
    # Beam remains correctness-oriented; use cached greedy for the fast path.
    if beam_size <= 1:
        return greedy_decode(
            model,
            features,
            codec,
            max_tokens=max_tokens,
            constrained=constrained,
            repetition_penalty=repetition_penalty,
            max_time_seconds=None,
            return_stats=return_stats,
        )
    model.eval()
    if features.ndim == 2:
        features = features.unsqueeze(0)
    device = next(model.parameters()).device
    memory = model.encode(features.to(device))
    stats = DecodeStats()
    started = time.perf_counter()
    beams = [([codec.bos_id], 0.0, ConstraintState(codec), None, False)]
    for position in range(max_tokens):
        candidates = []
        for tokens, score, state, cache, done in beams:
            if done:
                candidates.append((tokens, score, state, cache, done))
                continue
            current = torch.tensor([tokens[-1]], dtype=torch.long, device=device)
            logits, new_cache = model.decode_step(current, memory, cache, position, max_length=max_tokens)
            logits = logits[0]
            if constrained:
                logits = apply_constraints(logits, state)
            logits = _apply_repetition_penalty(logits, tokens, repetition_penalty)
            log_probs = torch.log_softmax(logits, dim=-1)
            values, indices = torch.topk(log_probs, beam_size)
            for value, idx in zip(values.tolist(), indices.tolist()):
                new_state = state.clone()
                new_state.update(idx)
                candidates.append(
                    (tokens + [idx], score + value, new_state, _clone_cache(new_cache), idx == codec.eos_id)
                )
        candidates.sort(key=lambda x: x[1] / max(1, len(x[0])), reverse=True)
        beams = candidates[:beam_size]
        if all(done for _, _, _, _, done in beams):
            stats.stop_reason = "eos"
            stats.eos_hit = True
            break
        if _detect_loop(beams[0][0]):
            stats.stop_reason = "loop_detected"
            break
    stats.wall_time = time.perf_counter() - started
    stats.tokens_per_second = (len(beams[0][0]) - 1) / max(stats.wall_time, 1e-6)
    return (beams[0][0], stats) if return_stats else beams[0][0]


def _apply_repetition_penalty(logits: torch.Tensor, tokens: list[int], penalty: float) -> torch.Tensor:
    if penalty <= 1.0 or not tokens:
        return logits
    out = logits.clone()
    for token in set(tokens[-64:]):
        if out[token] > 0:
            out[token] /= penalty
        else:
            out[token] *= penalty
    return out


def _detect_loop(tokens: list[int], loop_window: int = 16, repeats: int = 4) -> bool:
    span = loop_window * repeats
    if len(tokens) < span:
        return False
    tail = tokens[-span:]
    chunk = tail[:loop_window]
    return all(tail[i : i + loop_window] == chunk for i in range(0, span, loop_window))


def _clone_cache(cache):
    if cache is None:
        return None
    return [{k: v.clone() for k, v in layer.items()} for layer in cache]
