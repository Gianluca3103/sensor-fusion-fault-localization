"""Official synchronized VoD radar and clean-LiDAR pairs, with no fault dependency."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from Fault_Localization_Model.vod_dataset.vod_io import (
    align_radar_to_lidar, discover_vod_frames, load_vod_lidar,
    load_vod_radar, load_vod_radar_to_lidar,
)


class VoDStage1Dataset(Dataset):
    def __init__(self, root: str | Path, split: str, *, radar_variant: str = "radar_20frames_verified_doppler_radial", frame_ids: list[str] | None = None, include_clean: bool = True):
        self.frames = discover_vod_frames(root, split, radar_variant=radar_variant, frame_ids=frame_ids)
        self.include_clean = include_clean

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index: int) -> dict:
        frame = self.frames[index]
        radar = load_vod_radar(frame.radar_path)
        radar = align_radar_to_lidar(radar, load_vod_radar_to_lidar(frame.lidar_calibration_path, frame.radar_calibration_path))
        if not np.isfinite(radar).all():
            raise ValueError(f"Aligned VoD radar contains NaN or Inf: {frame.radar_path}")
        result = {"radar": torch.from_numpy(radar.copy()), "frame_id": frame.frame_id, "split": frame.split}
        if self.include_clean:
            result["clean_lidar"] = torch.from_numpy(load_vod_lidar(frame.lidar_path).copy())
        return result


def collate_stage1(batch: list[dict]) -> dict:
    if not batch or len({"clean_lidar" in row for row in batch}) != 1:
        raise ValueError("Batch must be nonempty and uniformly teacher-labeled")
    result = {"frame_id": [row["frame_id"] for row in batch], "split": [row["split"] for row in batch]}
    for name in ("radar", "clean_lidar"):
        if name not in batch[0]:
            continue
        maximum = max(len(row[name]) for row in batch)
        columns = batch[0][name].shape[-1]
        values = torch.zeros(len(batch), maximum, columns, dtype=torch.float32)
        mask = torch.zeros(len(batch), maximum, dtype=torch.bool)
        for i,row in enumerate(batch):
            count = len(row[name])
            values[i,:count] = row[name]
            mask[i,:count] = True
        result[name], result[name+"_valid"] = values, mask
    return result
