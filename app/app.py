#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gradio as gr
import torch

from minimt3.pipeline import export_transcription, load_checkpoint, load_infer_config, transcribe_audio
from minimt3.utils import ensure_dir, read_yaml


def create_app(config: dict[str, Any]) -> gr.Blocks:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    infer_cfg = load_infer_config(config.get("infer_config", "configs/infer.yaml"))
    checkpoint = config.get("checkpoint", "outputs/ckpt/best.pt")
    output_dir = ensure_dir(config.get("output_dir", "outputs/ui"))
    state: dict[str, Any] = {"model": None, "codec": None, "model_cfg": None}

    def transcribe(audio_path: str, decode_mode: str, pedal_cleanup: bool, quantize: bool):
        if not audio_path:
            raise gr.Error("Please upload an audio file.")
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise gr.Error(f"Checkpoint not found: {ckpt_path}")
        if state["model"] is None:
            model, codec, model_cfg = load_checkpoint(ckpt_path, device)
            state.update({"model": model, "codec": codec, "model_cfg": model_cfg})
        run_cfg = dict(infer_cfg)
        run_cfg["decode"] = dict(run_cfg.get("decode", {}), mode=decode_mode)
        run_cfg["cleanup"] = dict(
            run_cfg.get("cleanup", {}),
            pedal_aware=pedal_cleanup,
            quantize=quantize,
        )
        notes, pedals, debug = transcribe_audio(
            audio_path,
            state["model"],
            state["codec"],
            state["model_cfg"],
            run_cfg,
            device,
        )
        stem = Path(audio_path).stem
        paths = export_transcription(notes, pedals, output_dir, stem, run_cfg, debug)
        preview = paths.get("png") or paths.get("svg")
        message = f"Generated {len(notes)} notes, {len(pedals)} pedal regions."
        if paths.get("render_warnings"):
            message += "\n" + paths["render_warnings"]
        return (
            message,
            paths.get("midi"),
            paths.get("musicxml"),
            paths.get("debug_json"),
            preview,
        )

    with gr.Blocks(title="MiniMT3-Piano") as demo:
        gr.Markdown("# MiniMT3-Piano")
        with gr.Row():
            audio = gr.Audio(type="filepath", label="Audio")
            with gr.Column():
                decode_mode = gr.Dropdown(
                    ["constrained_greedy", "greedy", "beam"],
                    value="constrained_greedy",
                    label="Decode",
                )
                pedal_cleanup = gr.Checkbox(value=True, label="Pedal-aware cleanup")
                quantize = gr.Checkbox(value=True, label="Score quantization")
                run = gr.Button("Transcribe", variant="primary")
        status = gr.Textbox(label="Status")
        preview = gr.Image(label="Score Preview")
        with gr.Row():
            midi = gr.File(label="MIDI")
            musicxml = gr.File(label="MusicXML")
            debug = gr.File(label="Debug JSON")
        run.click(
            transcribe,
            inputs=[audio, decode_mode, pedal_cleanup, quantize],
            outputs=[status, midi, musicxml, debug, preview],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MiniMT3-Piano local UI.")
    parser.add_argument("--config", default="configs/ui.yaml")
    args = parser.parse_args()
    config = read_yaml(args.config)
    app = create_app(config)
    app.launch(
        server_name=config.get("server_name", "127.0.0.1"),
        server_port=int(config.get("server_port", 7860)),
        share=bool(config.get("share", False)),
    )


if __name__ == "__main__":
    main()
