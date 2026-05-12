from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from minimt3.audio.features import LogMelConfig, LogMelExtractor
from minimt3.audio.preprocess import load_audio
from minimt3.decode.beam_search import beam_decode, greedy_decode
from minimt3.decode.merge_windows import merge_overlapping_notes, offset_events
from minimt3.model.seq2seq import MiniMT3, MiniMT3Config
from minimt3.symbolic.cleanup import pedal_aware_cleanup, quantize_notes
from minimt3.symbolic.events import EventCodec, NoteEvent, PedalEvent
from minimt3.symbolic.midi_io import write_midi
from minimt3.symbolic.score_render import render_score, write_musicxml
from minimt3.utils import ensure_dir, read_yaml, write_json


def build_codec(model_config: dict[str, Any]) -> EventCodec:
    event_cfg = model_config.get("events", {})
    return EventCodec(
        time_shift_ms=event_cfg.get("time_shift_ms", 10),
        max_time_shift_steps=event_cfg.get("max_time_shift_steps", 100),
        velocity_bins=event_cfg.get("velocity_bins", 32),
    )


def build_model(model_config: dict[str, Any], codec: EventCodec) -> MiniMT3:
    audio_cfg = model_config.get("audio", {})
    model_cfg = model_config.get("model", {})
    config = MiniMT3Config(
        n_mels=audio_cfg.get("n_mels", 128),
        vocab_size=codec.vocab_size,
        pad_id=codec.pad_id,
        d_model=model_cfg.get("d_model", 256),
        encoder_layers=model_cfg.get("encoder_layers", 4),
        decoder_layers=model_cfg.get("decoder_layers", 4),
        nhead=model_cfg.get("nhead", 8),
        dim_feedforward=model_cfg.get("dim_feedforward", 1024),
        dropout=model_cfg.get("dropout", 0.1),
        conv_channels=model_cfg.get("conv_channels", 256),
    )
    return MiniMT3(config)


def load_checkpoint(ckpt_path: str | Path, device: str | torch.device = "cpu") -> tuple[MiniMT3, EventCodec, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    model_config = ckpt.get("model_config")
    if model_config is None:
        raise ValueError("Checkpoint is missing model_config.")
    codec = build_codec(model_config)
    model = build_model(model_config, codec)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, codec, model_config


def feature_config_from_model(model_config: dict[str, Any]) -> LogMelConfig:
    return LogMelConfig(**model_config.get("audio", {}))


def transcribe_audio(
    audio_path: str | Path,
    model: MiniMT3,
    codec: EventCodec,
    model_config: dict[str, Any],
    infer_config: dict[str, Any],
    device: str | torch.device,
) -> tuple[list[NoteEvent], list[PedalEvent], dict]:
    audio_cfg = feature_config_from_model(model_config)
    waveform = load_audio(audio_path, sample_rate=audio_cfg.sample_rate)
    extractor = LogMelExtractor(audio_cfg).to(device)
    sr = audio_cfg.sample_rate
    window = float(infer_config.get("window_seconds", 30.0))
    overlap = float(infer_config.get("overlap_seconds", 2.0))
    max_tokens = int(infer_config.get("max_tokens", 4096))
    decode_cfg = infer_config.get("decode", {})
    mode = decode_cfg.get("mode", "constrained_greedy")
    beam_size = int(decode_cfg.get("beam_size", 4))
    constrained = "constrained" in mode

    total_seconds = waveform.shape[-1] / sr
    starts = [0.0]
    if total_seconds > window:
        step = max(1.0, window - overlap)
        starts = []
        cur = 0.0
        while cur < total_seconds:
            starts.append(cur)
            cur += step

    all_notes: list[NoteEvent] = []
    all_pedals: list[PedalEvent] = []
    debug: dict[str, Any] = {"windows": []}
    with torch.no_grad():
        for start in starts:
            start_sample = int(start * sr)
            end_sample = int(min(total_seconds, start + window) * sr)
            chunk = waveform[:, start_sample:end_sample]
            if chunk.numel() == 0:
                continue
            features = extractor(chunk.to(device)).squeeze(0)
            if mode == "beam":
                token_ids = beam_decode(
                    model,
                    features,
                    codec,
                    beam_size=beam_size,
                    max_tokens=max_tokens,
                    constrained=constrained,
                )
            else:
                token_ids = greedy_decode(
                    model,
                    features,
                    codec,
                    max_tokens=max_tokens,
                    constrained=constrained,
                )
            decoded = codec.decode(token_ids)
            notes, pedals = offset_events(decoded.notes, decoded.pedals, start)
            all_notes.extend(notes)
            all_pedals.extend(pedals)
            debug["windows"].append(
                {
                    "start": start,
                    "tokens": token_ids,
                    "invalid_event_rate": decoded.invalid_events / max(1, decoded.total_events),
                }
            )

    notes = merge_overlapping_notes(all_notes)
    pedals = sorted(all_pedals, key=lambda p: (p.start, p.end))
    cleanup_cfg = infer_config.get("cleanup", {})
    if cleanup_cfg.get("pedal_aware", True):
        notes = pedal_aware_cleanup(notes, pedals)
    if cleanup_cfg.get("quantize", False):
        notes = quantize_notes(notes, step=float(cleanup_cfg.get("quantize_step", 0.125)))
    return notes, pedals, debug


def export_transcription(
    notes: list[NoteEvent],
    pedals: list[PedalEvent],
    out_dir: str | Path,
    stem: str,
    infer_config: dict[str, Any],
    debug: dict | None = None,
) -> dict[str, str]:
    out_dir = ensure_dir(out_dir)
    paths: dict[str, str] = {}
    midi_path = write_midi(out_dir / f"{stem}.mid", notes, pedals)
    paths["midi"] = str(midi_path)
    render_cfg = infer_config.get("render", {})
    if render_cfg.get("write_musicxml", True):
        xml_path = write_musicxml(out_dir / f"{stem}.musicxml", notes, title=stem)
        paths["musicxml"] = str(xml_path)
        rendered = render_score(
            xml_path,
            png_path=out_dir / f"{stem}.png" if render_cfg.get("write_png", True) else None,
            pdf_path=out_dir / f"{stem}.pdf" if render_cfg.get("write_pdf", True) else None,
            svg_path=out_dir / f"{stem}.svg" if render_cfg.get("write_svg", True) else None,
        )
        paths.update({k: v for k, v in rendered.items() if not k.endswith("_missing")})
        if any(k.endswith("_missing") for k in rendered):
            paths["render_warnings"] = "; ".join(v for k, v in rendered.items() if k.endswith("_missing"))
    if debug is not None:
        debug_path = out_dir / f"{stem}.json"
        debug["notes"] = [n.__dict__ for n in notes]
        debug["pedals"] = [p.__dict__ for p in pedals]
        write_json(debug_path, debug)
        paths["debug_json"] = str(debug_path)
    return paths


def load_infer_config(path: str | Path) -> dict[str, Any]:
    return read_yaml(path)
