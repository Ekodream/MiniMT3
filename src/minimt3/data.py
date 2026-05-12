from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from minimt3.audio.features import LogMelConfig, LogMelExtractor
from minimt3.audio.preprocess import load_audio
from minimt3.symbolic.events import EventCodec
from minimt3.utils import read_json


def index_maestro(data_dir: str | Path) -> list[dict[str, Any]]:
    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("maestro-v*.csv")) + sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No MAESTRO metadata CSV found in {data_dir}")
    csv_path = csv_files[0]
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio = data_dir / row.get("audio_filename", "")
            midi = data_dir / row.get("midi_filename", "")
            split = row.get("split", row.get("canonical_split", "train"))
            duration = float(row.get("duration", 0.0) or 0.0)
            rows.append(
                {
                    "split": "validation" if split == "valid" else split,
                    "audio": str(audio),
                    "midi": str(midi),
                    "duration": duration,
                    "year": row.get("year"),
                    "composer": row.get("canonical_composer", ""),
                    "title": row.get("canonical_title", row.get("title", "")),
                    "audio_exists": audio.exists(),
                    "midi_exists": midi.exists(),
                }
            )
    return rows


@dataclass
class Collator:
    pad_id: int

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_frames = max(item["features"].shape[-1] for item in batch)
        max_tokens = max(item["tokens"].numel() for item in batch)
        n_mels = batch[0]["features"].shape[-2]
        features = torch.zeros(len(batch), n_mels, max_frames)
        tokens = torch.full((len(batch), max_tokens), self.pad_id, dtype=torch.long)
        for i, item in enumerate(batch):
            feat = item["features"].squeeze(0)
            features[i, :, : feat.shape[-1]] = feat
            tokens[i, : item["tokens"].numel()] = item["tokens"]
        return {"features": features, "tokens": tokens}


class MaestroDataset(Dataset):
    def __init__(
        self,
        metadata_path: str | Path,
        split: str,
        codec: EventCodec,
        feature_config: LogMelConfig,
        train_seconds: float = 20.0,
        max_items: int | None = None,
    ):
        rows = read_json(metadata_path)
        self.rows = [
            r
            for r in rows
            if r.get("split") == split and r.get("audio_exists", True) and r.get("midi_exists", True)
        ]
        if max_items:
            self.rows = self.rows[:max_items]
        if not self.rows:
            raise ValueError(f"No usable MAESTRO rows for split={split!r} in {metadata_path}")
        self.codec = codec
        self.feature_config = feature_config
        self.feature_extractor = LogMelExtractor(feature_config)
        self.train_seconds = train_seconds

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        duration = float(row.get("duration") or 0.0)
        start = 0.0
        if duration > self.train_seconds:
            start = random.uniform(0.0, max(0.0, duration - self.train_seconds))
        end = start + self.train_seconds
        waveform = load_audio(row["audio"], self.feature_config.sample_rate)
        start_sample = int(start * self.feature_config.sample_rate)
        end_sample = int(end * self.feature_config.sample_rate)
        waveform = waveform[:, start_sample:end_sample]
        target_samples = int(self.train_seconds * self.feature_config.sample_rate)
        if waveform.shape[-1] < target_samples:
            waveform = torch.nn.functional.pad(waveform, (0, target_samples - waveform.shape[-1]))
        with torch.no_grad():
            features = self.feature_extractor(waveform)
        tokens = torch.tensor(
            self.codec.encode_midi_file(row["midi"], start=start, end=end, add_special=True),
            dtype=torch.long,
        )
        return {"features": features, "tokens": tokens}
