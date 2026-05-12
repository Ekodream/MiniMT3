from __future__ import annotations

import torch

from minimt3.symbolic.events import EventCodec


class ConstraintState:
    """Dynamic token mask for piano event decoding."""

    def __init__(self, codec: EventCodec):
        self.codec = codec
        self.active_notes: set[int] = set()
        self.pedal_active = False
        self.seen_content = False

    def clone(self) -> "ConstraintState":
        other = ConstraintState(self.codec)
        other.active_notes = set(self.active_notes)
        other.pedal_active = self.pedal_active
        other.seen_content = self.seen_content
        return other

    def allowed_mask(self, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(self.codec.vocab_size, dtype=torch.bool, device=device)
        for token_id, token in self.codec.id_to_token.items():
            if token in {"<PAD>", "<BOS>", "<UNK>"}:
                continue
            if token == "<EOS>":
                mask[token_id] = self.seen_content or bool(self.active_notes)
            elif token.startswith("TIME_SHIFT_") or token.startswith("VELOCITY_"):
                mask[token_id] = True
            elif token.startswith("NOTE_ON_"):
                pitch = int(token.rsplit("_", 1)[1])
                mask[token_id] = pitch not in self.active_notes
            elif token.startswith("NOTE_OFF_"):
                pitch = int(token.rsplit("_", 1)[1])
                mask[token_id] = pitch in self.active_notes
            elif token == "PEDAL_ON":
                mask[token_id] = not self.pedal_active
            elif token == "PEDAL_OFF":
                mask[token_id] = self.pedal_active
        return mask

    def update(self, token_id: int) -> None:
        token = self.codec.token(token_id)
        if token in {"<PAD>", "<BOS>", "<UNK>"}:
            return
        if token == "<EOS>":
            return
        self.seen_content = True
        if token.startswith("NOTE_ON_"):
            self.active_notes.add(int(token.rsplit("_", 1)[1]))
        elif token.startswith("NOTE_OFF_"):
            self.active_notes.discard(int(token.rsplit("_", 1)[1]))
        elif token == "PEDAL_ON":
            self.pedal_active = True
        elif token == "PEDAL_OFF":
            self.pedal_active = False


def apply_constraints(logits: torch.Tensor, state: ConstraintState) -> torch.Tensor:
    mask = state.allowed_mask(logits.device)
    constrained = logits.clone()
    constrained[~mask] = -1e9
    return constrained
