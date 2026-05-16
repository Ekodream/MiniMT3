from __future__ import annotations

from dataclasses import dataclass

import torch

from minimt3.symbolic.events import EventCodec


@dataclass
class ConstraintState:
    codec: EventCodec
    active_notes: set[int] | None = None
    pedal_active: bool = False
    current_velocity: int = 80
    pending_velocity: bool = False
    seen_content: bool = False
    current_time: float = 0.0
    last_pitch: int | None = None
    repeated_pitch_count: int = 0
    no_time_progress: int = 0
    tokens_since_shift: int = 0
    note_on_count: int = 0

    def __post_init__(self) -> None:
        if self.active_notes is None:
            self.active_notes = set()

    def clone(self) -> "ConstraintState":
        return ConstraintState(
            codec=self.codec,
            active_notes=set(self.active_notes or set()),
            pedal_active=self.pedal_active,
            current_velocity=self.current_velocity,
            pending_velocity=self.pending_velocity,
            seen_content=self.seen_content,
            current_time=self.current_time,
            last_pitch=self.last_pitch,
            repeated_pitch_count=self.repeated_pitch_count,
            no_time_progress=self.no_time_progress,
            tokens_since_shift=self.tokens_since_shift,
            note_on_count=self.note_on_count,
        )

    def allowed_mask(self, device: torch.device, min_time_for_eos: float = 0.5) -> torch.Tensor:
        mask = torch.zeros(self.codec.vocab_size, dtype=torch.bool, device=device)
        tensors = self.codec.constraint_tensors(device)
        if self.current_time >= min_time_for_eos and not self.pending_velocity:
            mask[self.codec.eos_id] = True

        if not self.pending_velocity:
            shift_ids = tensors["shift_ids"]
            if self.codec.time_mode == "absolute":
                shift_steps = tensors["shift_steps"]
                mask[shift_ids[shift_steps * self.codec.step_seconds >= self.current_time]] = True
            else:
                mask[shift_ids] = True

            velocity_ids = tensors["velocity_ids"]
            velocity_values = tensors["velocity_values"]
            if self.active_notes:
                mask[velocity_ids] = True
            else:
                mask[velocity_ids[velocity_values > 0]] = True

        pitch_ids = tensors["pitch_ids"]
        pitch_values = tensors["pitch_values"]
        active = self.active_notes or set()
        if self.pending_velocity:
            if self.current_velocity == 0:
                pitch_allowed = torch.tensor([int(p) in active for p in pitch_values.tolist()], device=device)
            else:
                pitch_allowed = torch.tensor([int(p) not in active for p in pitch_values.tolist()], device=device)
            mask[pitch_ids[pitch_allowed]] = True
        elif self.current_velocity > 0:
            pitch_allowed = torch.tensor([int(p) not in active for p in pitch_values.tolist()], device=device)
            mask[pitch_ids[pitch_allowed]] = True

        if not self.pending_velocity and not self.pedal_active:
            mask[self.codec.token_id("PEDAL_ON")] = True
        if not self.pending_velocity and self.pedal_active:
            mask[self.codec.token_id("PEDAL_OFF")] = True
        if not mask.any():
            mask[self.codec.eos_id] = True
        return mask

    def update(self, token_id: int) -> None:
        token = self.codec.token(token_id)
        if token in {"<PAD>", "<BOS>", "<UNK>", "<EOS>"}:
            return
        family = self.codec.token_family(token)
        if family == "SHIFT":
            previous_time = self.current_time
            step_time = int(token.rsplit("_", 1)[1]) * self.codec.step_seconds
            self.current_time = step_time if self.codec.time_mode == "absolute" else self.current_time + step_time
            self.no_time_progress = self.no_time_progress + 1 if self.current_time <= previous_time else 0
            self.tokens_since_shift = 0
        elif family == "VELOCITY":
            self.current_velocity = self.codec.bin_to_velocity(int(token.rsplit("_", 1)[1]))
            self.pending_velocity = True
            self.seen_content = True
            self.tokens_since_shift += 1
        elif family == "PITCH":
            pitch = int(token.rsplit("_", 1)[1])
            if self.current_velocity == 0:
                self.active_notes.discard(pitch)
            else:
                self.active_notes.add(pitch)
                self.note_on_count += 1
            self.pending_velocity = False
            if self.last_pitch == pitch:
                self.repeated_pitch_count += 1
            else:
                self.last_pitch = pitch
                self.repeated_pitch_count = 1
            self.seen_content = True
            self.tokens_since_shift += 1
        elif token == "PEDAL_ON":
            self.pedal_active = True
            self.seen_content = True
            self.tokens_since_shift += 1
        elif token == "PEDAL_OFF":
            self.pedal_active = False
            self.seen_content = True
            self.tokens_since_shift += 1


def apply_constraints(logits: torch.Tensor, state: ConstraintState, min_time_for_eos: float = 0.5) -> torch.Tensor:
    mask = state.allowed_mask(logits.device, min_time_for_eos=min_time_for_eos)
    constrained = logits.clone()
    constrained[~mask] = -1e9
    return constrained
