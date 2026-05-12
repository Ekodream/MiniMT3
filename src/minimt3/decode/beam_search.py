from __future__ import annotations

import torch

from minimt3.decode.constraints import ConstraintState, apply_constraints
from minimt3.model.seq2seq import MiniMT3
from minimt3.symbolic.events import EventCodec


@torch.no_grad()
def greedy_decode(
    model: MiniMT3,
    features: torch.Tensor,
    codec: EventCodec,
    max_tokens: int = 4096,
    constrained: bool = True,
) -> list[int]:
    model.eval()
    if features.ndim == 2:
        features = features.unsqueeze(0)
    device = next(model.parameters()).device
    features = features.to(device)
    memory = model.encode(features)
    tokens = [codec.bos_id]
    state = ConstraintState(codec)
    for _ in range(max_tokens):
        inp = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        logits = model.decoder(inp, memory)[0, -1]
        if constrained:
            logits = apply_constraints(logits, state)
        next_id = int(torch.argmax(logits).item())
        tokens.append(next_id)
        state.update(next_id)
        if next_id == codec.eos_id:
            break
    return tokens


@torch.no_grad()
def beam_decode(
    model: MiniMT3,
    features: torch.Tensor,
    codec: EventCodec,
    beam_size: int = 4,
    max_tokens: int = 4096,
    constrained: bool = True,
) -> list[int]:
    model.eval()
    if beam_size <= 1:
        return greedy_decode(model, features, codec, max_tokens=max_tokens, constrained=constrained)
    if features.ndim == 2:
        features = features.unsqueeze(0)
    device = next(model.parameters()).device
    memory = model.encode(features.to(device))
    beams: list[tuple[list[int], float, ConstraintState, bool]] = [
        ([codec.bos_id], 0.0, ConstraintState(codec), False)
    ]
    for _ in range(max_tokens):
        candidates: list[tuple[list[int], float, ConstraintState, bool]] = []
        for tokens, score, state, done in beams:
            if done:
                candidates.append((tokens, score, state, done))
                continue
            inp = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
            logits = model.decoder(inp, memory)[0, -1]
            if constrained:
                logits = apply_constraints(logits, state)
            log_probs = torch.log_softmax(logits, dim=-1)
            values, indices = torch.topk(log_probs, beam_size)
            for value, idx in zip(values.tolist(), indices.tolist()):
                new_state = state.clone()
                new_state.update(idx)
                candidates.append((tokens + [idx], score + value, new_state, idx == codec.eos_id))
        candidates.sort(key=lambda x: x[1] / max(1, len(x[0])), reverse=True)
        beams = candidates[:beam_size]
        if all(done for _, _, _, done in beams):
            break
    return beams[0][0]
