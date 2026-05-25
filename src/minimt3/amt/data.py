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
        teacher_midi_dir: str | Path | None = None,
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
        self.teacher_midi_dir = Path(teacher_midi_dir) if teacher_midi_dir else None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | dict[str, Any]]:
        row = self.rows[index]
        cache_path = self._cache_path(row) if self.cache_dir else None
        if cache_path and cache_path.exists():
            try:
                item = torch.load(cache_path, map_location="cpu", weights_only=False)
                if self._cache_missing_teacher_targets(item, row):
                    raise ValueError("cached AMT item predates teacher targets")
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
        frames = encoder_frame_count(features.shape[-1], self.target_config.frame_strides)
        targets = build_dense_targets(row["midi"], start=start, end=end, frames=frames, cfg=self.target_config)
        teacher_targets = self._teacher_targets(row, clip_duration, frames)
        valid_mask = torch.ones(frames, dtype=torch.bool)
        margin = float(getattr(self.target_config, "supervision_margin_seconds", 0.0) or 0.0)
        if margin > 0.0 and clip_duration > margin * 2.0 and frames > 2:
            frame_seconds = clip_duration / max(1, frames)
            left = min(frames, max(0, int(round(margin / max(1e-6, frame_seconds)))))
            right = max(left, frames - left)
            valid_mask[:left] = False
            valid_mask[right:] = False
        return {"features": features, "valid_mask": valid_mask, "meta": row, **targets, **teacher_targets}

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
                str(self.target_config.include_duration_bucket),
                str(self.target_config.include_time_shifts),
                str(self.target_config.max_duration_seconds),
                ",".join(str(float(x)) for x in self.target_config.duration_bucket_bounds),
                ",".join(str(int(s)) for s in self.target_config.frame_strides),
                str(self.target_config.time_shift_clip_frames),
                str(getattr(self.target_config, "supervision_margin_seconds", 0.0)),
                str(self.teacher_midi_dir or ""),
            ]
        )
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
        return Path(self.cache_dir) / f"{digest}.pt"

    def _cache_missing_teacher_targets(self, item: dict[str, Any], row: dict[str, Any]) -> bool:
        return self._teacher_midi_path(row) is not None and "teacher_onset" not in item

    def _teacher_targets(self, row: dict[str, Any], clip_duration: float, frames: int) -> dict[str, torch.Tensor]:
        teacher_midi = self._teacher_midi_path(row)
        if teacher_midi is None:
            return {}
        try:
            targets = build_dense_targets(
                teacher_midi,
                start=0.0,
                end=clip_duration,
                frames=frames,
                cfg=self.target_config,
            )
        except Exception:
            return {}
        out: dict[str, torch.Tensor] = {}
        for key in ("onset", "frame", "offset"):
            if key in targets:
                out[f"teacher_{key}"] = targets[key]
        return out

    def _teacher_midi_path(self, row: dict[str, Any]) -> Path | None:
        if self.teacher_midi_dir is None:
            return None
        candidates = [
            self.teacher_midi_dir / f"{row.get('clip_id')}.mid",
            self.teacher_midi_dir / f"{Path(str(row.get('audio', ''))).stem}.mid",
            self.teacher_midi_dir / f"{Path(str(row.get('midi', ''))).stem}.mid",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None


class DenseAMTCollator:
    def __init__(self, sample_weight_by_category: dict[str, float] | None = None):
        self.sample_weight_by_category = sample_weight_by_category or {}

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
            "sample_weight": torch.ones(len(batch), dtype=torch.float32),
            "meta": [item["meta"] for item in batch],
        }
        if any("pedal" in item for item in batch):
            out["pedal"] = torch.zeros(len(batch), max_target_frames, 1)
        if any("duration" in item for item in batch):
            out["duration"] = torch.zeros(len(batch), max_target_frames, 88)
            out["duration_frame"] = torch.zeros(len(batch), max_target_frames, 88)
            out["duration_mask"] = torch.zeros(len(batch), max_target_frames, 88)
        if any("duration_bucket" in item for item in batch):
            out["duration_bucket"] = torch.zeros(len(batch), max_target_frames, 88, dtype=torch.long)
            out["duration_bucket_mask"] = torch.zeros(len(batch), max_target_frames, 88)
        if any("onset_shift" in item for item in batch):
            out["onset_shift"] = torch.zeros(len(batch), max_target_frames, 88)
            out["onset_shift_mask"] = torch.zeros(len(batch), max_target_frames, 88)
            out["offset_shift"] = torch.zeros(len(batch), max_target_frames, 88)
            out["offset_shift_mask"] = torch.zeros(len(batch), max_target_frames, 88)
        if any("teacher_onset" in item for item in batch):
            out["teacher_onset"] = torch.zeros(len(batch), max_target_frames, 88)
            out["teacher_frame"] = torch.zeros(len(batch), max_target_frames, 88)
            out["teacher_offset"] = torch.zeros(len(batch), max_target_frames, 88)
            out["teacher_mask"] = torch.zeros(len(batch), max_target_frames, 88)
        for i, item in enumerate(batch):
            feat = item["features"]
            target_len = item["onset"].shape[0]
            features[i, :, : feat.shape[-1]] = feat
            out["valid_mask"][i, :target_len] = item["valid_mask"]
            meta = item.get("meta", {})
            if isinstance(meta, dict):
                category = str(meta.get("hard_category", "base"))
                out["sample_weight"][i] = float(self.sample_weight_by_category.get(category, 1.0))
            for key in ("onset", "frame", "offset", "velocity", "onset_mask"):
                out[key][i, :target_len] = item[key]
            if "pedal" in out and "pedal" in item:
                out["pedal"][i, :target_len] = item["pedal"]
            if "duration" in out and "duration" in item:
                out["duration"][i, :target_len] = item["duration"]
                if "duration_frame" in item:
                    out["duration_frame"][i, :target_len] = item["duration_frame"]
                out["duration_mask"][i, :target_len] = item["duration_mask"]
            if "duration_bucket" in out and "duration_bucket" in item:
                out["duration_bucket"][i, :target_len] = item["duration_bucket"]
                out["duration_bucket_mask"][i, :target_len] = item["duration_bucket_mask"]
            if "onset_shift" in out and "onset_shift" in item:
                out["onset_shift"][i, :target_len] = item["onset_shift"]
                out["onset_shift_mask"][i, :target_len] = item["onset_shift_mask"]
                out["offset_shift"][i, :target_len] = item["offset_shift"]
                out["offset_shift_mask"][i, :target_len] = item["offset_shift_mask"]
            if "teacher_onset" in out and "teacher_onset" in item:
                for key in ("onset", "frame", "offset"):
                    teacher_key = f"teacher_{key}"
                    if teacher_key in item:
                        out[teacher_key][i, :target_len] = item[teacher_key]
                        out["teacher_mask"][i, :target_len] = torch.maximum(
                            out["teacher_mask"][i, :target_len],
                            (item[teacher_key] > 0).float(),
                        )
        return out
