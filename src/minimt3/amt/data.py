from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from minimt3.amt.targets import DenseTargetConfig, build_dense_targets, encoder_frame_count
from minimt3.audio.features import LogMelConfig, LogMelExtractor
from minimt3.audio.preprocess import load_audio
from minimt3.utils import ensure_dir, read_json


class DenseAMTDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        feature_config: LogMelConfig,
        split: str = "train",
        max_items: int | None = None,
        cache_dir: str | Path | None = None,
        target_config: DenseTargetConfig | None = None,
    ):
        rows = read_json(manifest)
        self.rows = [
            row
            for row in rows
            if (not split or row.get("split", split) == split)
            and row.get("audio_exists", True)
            and row.get("midi_exists", True)
        ]
        if max_items:
            self.rows = self.rows[:max_items]
        if not self.rows:
            raise ValueError(f"No usable AMT clips for split={split!r} in {manifest}")
        self.feature_config = feature_config
        self.extractor = LogMelExtractor(feature_config)
        self.cache_dir = ensure_dir(cache_dir) if cache_dir else None
        self.target_config = target_config or DenseTargetConfig()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | dict[str, Any]]:
        row = self.rows[index]
        cache_path = self._cache_path(row) if self.cache_dir else None
        if cache_path and cache_path.exists():
            try:
                item = torch.load(cache_path, map_location="cpu", weights_only=False)
                item["meta"] = row
                return item
            except Exception:
                pass
        item = self._build_item(row)
        if cache_path:
            tmp = cache_path.with_suffix(".tmp")
            torch.save({k: v for k, v in item.items() if k != "meta"}, tmp)
            tmp.replace(cache_path)
        return item

    def _build_item(self, row: dict[str, Any]) -> dict[str, torch.Tensor | dict[str, Any]]:
        start = float(row.get("start_sec", 0.0))
        end = float(row.get("end_sec", start + float(row.get("duration", 0.0) or 0.0)))
        clip_duration = max(0.01, end - start)
        waveform = load_audio(
            row["audio"],
            self.feature_config.sample_rate,
            offset_seconds=start,
            duration_seconds=clip_duration,
        )
        target_samples = max(1, int(clip_duration * self.feature_config.sample_rate))
        if waveform.shape[-1] < target_samples:
            waveform = torch.nn.functional.pad(waveform, (0, target_samples - waveform.shape[-1]))
        with torch.no_grad():
            features = self.extractor(waveform).squeeze(0)
        frames = encoder_frame_count(features.shape[-1])
        targets = build_dense_targets(row["midi"], start=start, end=end, frames=frames, cfg=self.target_config)
        valid_mask = torch.ones(frames, dtype=torch.bool)
        return {"features": features, "valid_mask": valid_mask, "meta": row, **targets}

    def _cache_path(self, row: dict[str, Any]) -> Path:
        key = "|".join(
            [
                str(row.get("clip_id", "")),
                str(row.get("audio", "")),
                str(row.get("midi", "")),
                str(row.get("start_sec", "")),
                str(row.get("end_sec", "")),
                str(self.feature_config.sample_rate),
                str(self.feature_config.n_mels),
                str(self.feature_config.n_fft),
                str(self.feature_config.hop_length),
                str(self.target_config.onset_width_frames),
                str(self.target_config.offset_width_frames),
                str(self.target_config.onset_soft_radius_frames),
                str(self.target_config.offset_soft_radius_frames),
                str(self.target_config.min_note_seconds),
                str(self.target_config.include_pedal),
                str(self.target_config.include_duration),
                str(self.target_config.max_duration_seconds),
            ]
        )
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
        return Path(self.cache_dir) / f"{digest}.pt"


class DenseAMTCollator:
    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor | list[dict[str, Any]]]:
        max_feature_frames = max(item["features"].shape[-1] for item in batch)
        max_target_frames = max(item["onset"].shape[0] for item in batch)
        n_mels = batch[0]["features"].shape[0]
        features = torch.zeros(len(batch), n_mels, max_feature_frames)
        out: dict[str, torch.Tensor | list[dict[str, Any]]] = {
            "features": features,
            "valid_mask": torch.zeros(len(batch), max_target_frames, dtype=torch.bool),
            "onset": torch.zeros(len(batch), max_target_frames, 88),
            "frame": torch.zeros(len(batch), max_target_frames, 88),
            "offset": torch.zeros(len(batch), max_target_frames, 88),
            "velocity": torch.zeros(len(batch), max_target_frames, 88),
            "onset_mask": torch.zeros(len(batch), max_target_frames, 88),
            "meta": [item["meta"] for item in batch],
        }
        if any("pedal" in item for item in batch):
            out["pedal"] = torch.zeros(len(batch), max_target_frames, 1)
        if any("duration" in item for item in batch):
            out["duration"] = torch.zeros(len(batch), max_target_frames, 88)
            out["duration_mask"] = torch.zeros(len(batch), max_target_frames, 88)
        for i, item in enumerate(batch):
            feat = item["features"]
            target_len = item["onset"].shape[0]
            features[i, :, : feat.shape[-1]] = feat
            out["valid_mask"][i, :target_len] = item["valid_mask"]
            for key in ("onset", "frame", "offset", "velocity", "onset_mask"):
                out[key][i, :target_len] = item[key]
            if "pedal" in out and "pedal" in item:
                out["pedal"][i, :target_len] = item["pedal"]
            if "duration" in out and "duration" in item:
                out["duration"][i, :target_len] = item["duration"]
                out["duration_mask"][i, :target_len] = item["duration_mask"]
        return out
