from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from Fault_Localization_Model.model_blocks import resize_reliability_map
from Fault_Localization_Model.sample_utils import InvalidSampleError, validate_heatmap_array, validate_radar_array, validate_rgb_array
from PFS_Radar.radar_data import radar_cache_path
from .patching import crop_sample_tensors


def _resize(tensor: torch.Tensor, size, mode: str) -> torch.Tensor:
    if size is None or tensor.shape[-2:] == tuple(size):
        return tensor
    kwargs = {"size": tuple(size), "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return F.interpolate(tensor[None], **kwargs)[0]


class Stage1RadarReconstructionDataset(Dataset):
    """Use current reliability samples for oracle-mask LiDAR reconstruction.

    Mapping:
    - ``faulty_rgb`` -> corrupted LiDAR BEV features, normalized to ``[0,1]``.
    - ``clean_rgb`` -> clean LiDAR target features.
    - ``fault_heatmap`` -> oracle binary mask after thresholding.
    - existing radar cache ``radar_bev`` -> aligned radar conditioning.
    """

    def __init__(
        self,
        paths,
        radar_root: Path,
        *,
        resize_hw=(320, 320),
        mask_threshold: float = 0.0,
        use_patches: bool = False,
        patch_size: int = 128,
        halo_radius: int = 12,
        full_frame_fallback: bool = True,
    ):
        self.paths = list(paths)
        self.radar_root = Path(radar_root)
        self.resize_hw = tuple(resize_hw) if resize_hw is not None else None
        self.mask_threshold = float(mask_threshold)
        self.use_patches = bool(use_patches)
        self.patch_size = int(patch_size)
        self.halo_radius = int(halo_radius)
        self.full_frame_fallback = bool(full_frame_fallback)
        if not self.paths:
            raise FileNotFoundError("No Stage I reconstruction samples were provided")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = Path(self.paths[index])
        try:
            with np.load(path, allow_pickle=False) as data:
                required = {"faulty_rgb", "clean_rgb", "fault_heatmap", "metadata_json"}
                missing = required - set(data.files)
                if missing:
                    raise KeyError(f"missing arrays: {', '.join(sorted(missing))}")
                faulty = validate_rgb_array(data["faulty_rgb"], name="faulty_rgb", path=path).astype(np.float32) / 255.0
                clean = validate_rgb_array(data["clean_rgb"], name="clean_rgb", path=path).astype(np.float32) / 255.0
                heatmap = validate_heatmap_array(data["fault_heatmap"], path=path).astype(np.float32)
                metadata_json = str(data["metadata_json"])
                metadata = json.loads(metadata_json)
            radar_path = radar_cache_path(self.radar_root, metadata)
            with np.load(radar_path, allow_pickle=False) as radar_data:
                radar = validate_radar_array(radar_data["radar_bev"], path=radar_path).astype(np.float32)
        except InvalidSampleError:
            raise
        except Exception as exc:
            raise InvalidSampleError(f"Cannot load Stage I sample {path}: {exc}") from exc
        lidar_corrupt = torch.from_numpy(faulty.transpose(2, 0, 1))
        lidar_clean = torch.from_numpy(clean.transpose(2, 0, 1))
        mask = torch.from_numpy((heatmap > self.mask_threshold).astype(np.float32))[None]
        radar = torch.from_numpy(radar)
        if self.resize_hw:
            lidar_corrupt = _resize(lidar_corrupt, self.resize_hw, "bilinear")
            lidar_clean = _resize(lidar_clean, self.resize_hw, "bilinear")
            mask = resize_reliability_map(mask[None], self.resize_hw)[0]
            mask = (mask > 0.5).float()
            radar = _resize(radar, self.resize_hw, "bilinear")
        if int(mask.sum().item()) == 0:
            raise InvalidSampleError(f"Stage I sample has an empty oracle mask: {path}")
        occupancy = (lidar_corrupt.abs().sum(dim=0, keepdim=True) > 1e-6).float()
        tensors = {
            "lidar_corrupt": lidar_corrupt,
            "lidar_clean": lidar_clean,
            "radar": radar,
            "mask": mask,
            "occupancy": occupancy,
        }
        patch_metadata = None
        if self.use_patches:
            tensors, patch_metadata = crop_sample_tensors(
                tensors,
                patch_size=self.patch_size,
                halo_radius=self.halo_radius,
                full_frame_fallback=self.full_frame_fallback,
            )
        return {
            **tensors,
            "metadata_json": metadata_json,
            "metadata": metadata,
            "path": str(path),
            "patch_metadata": patch_metadata,
        }


def collate_stage1_batch(batch):
    keys = ("lidar_corrupt", "lidar_clean", "radar", "mask", "occupancy")
    return {
        **{key: torch.stack([item[key] for item in batch]) for key in keys},
        "metadata": [item["metadata"] for item in batch],
        "metadata_json": [item["metadata_json"] for item in batch],
        "path": [item["path"] for item in batch],
        "patch_metadata": [item["patch_metadata"] for item in batch],
    }

