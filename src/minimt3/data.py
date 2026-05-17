from __future__ import annotations

import csv
import random
from collections import Counter
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


def build_fixed_clip_manifest(
    rows: list[dict[str, Any]],
    split: str,
    clip_seconds: float,
    count: int,
    starts_per_piece: int = 1,
) -> list[dict[str, Any]]:
    usable = [
        r
        for r in rows
        if r.get("split") == split
        and r.get("audio_exists", True)
        and r.get("midi_exists", True)
        and float(r.get("duration") or 0.0) > 0
    ]
    if not usable:
        raise ValueError(f"No usable rows for fixed manifest split={split!r}")
    # Deterministic spread over the split without reading audio/MIDI.
    step = max(1, len(usable) // max(1, count))
    selected = usable[::step][:count]
    clips: list[dict[str, Any]] = []
    for row_idx, row in enumerate(selected):
        duration = float(row.get("duration") or clip_seconds)
        max_start = max(0.0, duration - clip_seconds)
        if starts_per_piece <= 1:
            starts = [0.0 if max_start == 0 else max_start * 0.33]
        else:
            starts = [max_start * i / (starts_per_piece - 1) for i in range(starts_per_piece)]
        for clip_idx, start in enumerate(starts):
            end = min(duration, start + clip_seconds)
            clips.append(
                {
                    "clip_id": f"{split}_{row_idx:04d}_{clip_idx:02d}",
                    "split": split,
                    "audio": row["audio"],
                    "midi": row["midi"],
                    "start_sec": round(start, 3),
                    "end_sec": round(end, 3),
                    "duration": round(end - start, 3),
                    "composer": row.get("composer", ""),
                    "title": row.get("title", ""),
                    "audio_exists": row.get("audio_exists", True),
                    "midi_exists": row.get("midi_exists", True),
                }
            )
    return clips[:count]


@dataclass
class Collator:
    pad_id: int

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_frames = max(item["features"].shape[-1] for item in batch)
        max_tokens = max(item["tokens"].numel() for item in batch)
        n_mels = batch[0]["features"].shape[-2]
        features = torch.zeros(len(batch), n_mels, max_frames)
        tokens = torch.full((len(batch), max_tokens), self.pad_id, dtype=torch.long)
        lengths = torch.zeros(len(batch), dtype=torch.long)
        for i, item in enumerate(batch):
            feat = item["features"].squeeze(0)
            features[i, :, : feat.shape[-1]] = feat
            tokens[i, : item["tokens"].numel()] = item["tokens"]
            lengths[i] = item["tokens"].numel()
        return {"features": features, "tokens": tokens, "lengths": lengths}


class MaestroDataset(Dataset):
    def __init__(
        self,
        metadata_path: str | Path,
        split: str,
        codec: EventCodec,
        feature_config: LogMelConfig,
        train_seconds: float = 20.0,
        max_items: int | None = None,
        sampling: str = "random",
        seed: int = 42,
        include_ties: bool = False,
    ):
        if sampling not in {"random", "fixed"}:
            raise ValueError("sampling must be 'random' or 'fixed'")
        rows = read_json(metadata_path)
        self.sampling = sampling
        self.seed = seed
        self.epoch = 0
        self.codec = codec
        self.feature_config = feature_config
        self.feature_extractor = LogMelExtractor(feature_config)
        self.train_seconds = train_seconds
        self.include_ties = include_ties

        if _looks_like_clip_manifest(rows):
            self.rows = [
                r
                for r in rows
                if r.get("audio_exists", True)
                and r.get("midi_exists", True)
                and (not split or r.get("split", split) == split)
            ]
            if sampling != "fixed":
                raise ValueError("Clip manifests must be used with sampling='fixed'")
        else:
            self.rows = [
                r
                for r in rows
                if r.get("split") == split
                and r.get("audio_exists", True)
                and r.get("midi_exists", True)
            ]
            if sampling == "fixed":
                self.rows = _rows_to_default_fixed_clips(self.rows, train_seconds)

        if max_items:
            self.rows = self.rows[:max_items]
        if not self.rows:
            raise ValueError(f"No usable MAESTRO rows for split={split!r} in {metadata_path}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        start, end = self._clip_bounds(row, index)
        waveform = load_audio(row["audio"], self.feature_config.sample_rate)
        start_sample = int(start * self.feature_config.sample_rate)
        end_sample = int(end * self.feature_config.sample_rate)
        waveform = waveform[:, start_sample:end_sample]
        target_samples = int((end - start) * self.feature_config.sample_rate)
        if waveform.shape[-1] < target_samples:
            waveform = torch.nn.functional.pad(waveform, (0, target_samples - waveform.shape[-1]))
        with torch.no_grad():
            features = self.feature_extractor(waveform)
        tokens = torch.tensor(
            self.codec.encode_midi_file(
                row["midi"],
                start=start,
                end=end,
                add_special=True,
                include_ties=self.include_ties,
            ),
            dtype=torch.long,
        )
        return {"features": features, "tokens": tokens}

    def get_tokens(self, index: int) -> torch.Tensor:
        row = self.rows[index]
        start, end = self._clip_bounds(row, index)
        return torch.tensor(
            self.codec.encode_midi_file(
                row["midi"],
                start=start,
                end=end,
                add_special=True,
                include_ties=self.include_ties,
            ),
            dtype=torch.long,
        )

    def _clip_bounds(self, row: dict[str, Any], index: int) -> tuple[float, float]:
        if self.sampling == "fixed":
            start = float(row.get("start_sec", 0.0))
            end = float(row.get("end_sec", start + float(row.get("duration", self.train_seconds))))
            return start, max(start + 0.01, end)

        duration = float(row.get("duration") or 0.0)
        rng = random.Random(self.seed + self.epoch * 1_000_003 + index)
        start = 0.0
        if duration > self.train_seconds:
            start = rng.uniform(0.0, max(0.0, duration - self.train_seconds))
        return start, start + self.train_seconds


def summarize_token_targets(dataset: Dataset, codec: EventCodec, max_items: int = 128) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    lengths: list[int] = []
    seconds: list[float] = []
    eos = 0
    for i in range(min(max_items, len(dataset))):
        if hasattr(dataset, "get_tokens"):
            tokens = dataset.get_tokens(i).tolist()
        else:
            tokens = dataset[i]["tokens"].tolist()
        lengths.append(len(tokens))
        rows = getattr(dataset, "rows", None)
        if rows is not None and i < len(rows):
            row = rows[i]
            if "start_sec" in row and "end_sec" in row:
                seconds.append(max(0.01, float(row["end_sec"]) - float(row["start_sec"])))
            elif hasattr(dataset, "train_seconds"):
                seconds.append(max(0.01, float(getattr(dataset, "train_seconds"))))
        eos += int(codec.eos_id in tokens)
        family_counts.update(codec.token_family(t) for t in tokens)
    total = sum(family_counts.values())
    avg_length = sum(lengths) / max(1, len(lengths))
    avg_seconds = sum(seconds) / max(1, len(seconds)) if seconds else None
    return {
        "items": len(lengths),
        "avg_target_length": avg_length,
        "max_target_length": max(lengths) if lengths else 0,
        "eos_rate": eos / max(1, len(lengths)),
        "family_counts": dict(family_counts),
        "family_ratio": {k: v / max(1, total) for k, v in family_counts.items()},
        "shift_token_density": family_counts.get("SHIFT", 0) / max(1, total),
        "avg_tokens_per_second": (avg_length / avg_seconds) if avg_seconds else None,
    }


def _looks_like_clip_manifest(rows: Any) -> bool:
    return bool(rows) and isinstance(rows, list) and "start_sec" in rows[0] and "end_sec" in rows[0]


def _rows_to_default_fixed_clips(rows: list[dict[str, Any]], clip_seconds: float) -> list[dict[str, Any]]:
    clips = []
    for idx, row in enumerate(rows):
        duration = float(row.get("duration") or clip_seconds)
        start = 0.0 if duration <= clip_seconds else max(0.0, (duration - clip_seconds) * 0.33)
        end = min(duration, start + clip_seconds)
        clips.append(
            {
                **row,
                "clip_id": f"{row.get('split', 'fixed')}_{idx:04d}",
                "start_sec": start,
                "end_sec": end,
                "duration": end - start,
            }
        )
    return clips
