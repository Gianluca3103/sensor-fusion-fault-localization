"""Loading and dataset adapter for coarse reconstruction samples."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import Dataset

from Fault_Localization_Model.sample_utils import InvalidSampleError

from .fault_selector import FaultSelectorConfig
from .pointpillars import BEVGridGeometry
from .geometric_augmentation import (
    GeometricAugmentationConfig,
    ReconstructionGeometricAugmentation,
)


def radar_cache_path(radar_root: str | Path, metadata: dict) -> Path:
    """Resolve the aligned View-of-Delft radar cache for one sample."""

    dataset = str(metadata.get("dataset", "")).strip().lower()
    if dataset not in {"view-of-delft", "view of delft", "vod"}:
        raise ValueError(f"Unsupported dataset {metadata.get('dataset')!r}; expected VoD")
    split = str(metadata.get("split", "")).strip()
    frame_id = str(metadata.get("frame_id", metadata.get("radar_index", ""))).strip()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Invalid VoD split {split!r}")
    if not frame_id.isdigit():
        raise ValueError(f"VoD frame_id must be numeric, got {frame_id!r}")
    return Path(radar_root) / split / f"{int(frame_id):05d}.npz"
from .fault_selector_cache import (
    load_selector_cache,
    selector_cache_path,
)


def load_bev_triplet(
    sample_path: str | Path,
    radar_root: str | Path,
    *,
    include_pointpillars_inputs: bool = False,
) -> dict[str, object]:
    """Load one aligned, normalized BEV triplet from existing artifacts."""

    sample_path = Path(sample_path)
    radar_root = Path(radar_root)
    try:
        with np.load(sample_path, allow_pickle=False) as sample:
            clean = np.asarray(sample["clean_rgb"])
            faulty = np.asarray(sample["faulty_rgb"])
            metadata = json.loads(str(sample["metadata_json"]))
            if include_pointpillars_inputs and "faulty_lidar_points" not in sample.files:
                raise InvalidSampleError(
                    f"{sample_path} has no raw faulty LiDAR points; regenerate "
                    "the sample with generator version 11 or newer"
                )
            faulty_lidar_points = (
                np.asarray(sample["faulty_lidar_points"], dtype=np.float32)
                if include_pointpillars_inputs
                else None
            )
            observability = (
                np.asarray(sample["observability_confidence"], dtype=np.float32)
                if "observability_confidence" in sample.files
                else None
            )
            lidar_input_bev = (
                np.asarray(sample["faulty_lidar_input_bev"], dtype=np.float32)
                if "faulty_lidar_input_bev" in sample.files
                else None
            )
        radar_path = radar_cache_path(radar_root, metadata)
        with np.load(radar_path, allow_pickle=False) as radar_cache:
            if include_pointpillars_inputs and "radar_points" not in radar_cache.files:
                raise InvalidSampleError(
                    f"{radar_path} has no aligned raw radar points; rebuild the "
                    "View-of-Delft radar cache"
                )
            radar = np.asarray(radar_cache["radar_bev"], dtype=np.float32)
            radar_points = (
                np.asarray(radar_cache["radar_points"], dtype=np.float32)
                if include_pointpillars_inputs
                else None
            )
    except InvalidSampleError:
        raise
    except Exception as exc:
        raise InvalidSampleError(
            f"Cannot load aligned BEV triplet for {sample_path}: {exc}"
        ) from exc

    item = {
        "clean_bev": torch.from_numpy(
            clean.astype(np.float32).transpose(2, 0, 1) / 255.0
        ),
        "radar_bev": torch.from_numpy(radar),
        "faulty_bev": torch.from_numpy(
            faulty.astype(np.float32).transpose(2, 0, 1) / 255.0
        ),
        "sample_path": str(sample_path),
    }
    if lidar_input_bev is not None:
        if lidar_input_bev.ndim != 3 or lidar_input_bev.shape[1:] != clean.shape[:2]:
            raise InvalidSampleError(
                "faulty_lidar_input_bev must have shape [C,H,W] aligned with "
                f"the target BEV; got {lidar_input_bev.shape}"
            )
        item["lidar_input_bev"] = torch.from_numpy(lidar_input_bev)
    if observability is not None:
        if observability.shape != clean.shape[:2]:
            raise InvalidSampleError(
                "observability_confidence must align with the LiDAR BEV; "
                f"got {observability.shape} and {clean.shape[:2]}"
            )
        item["observability_confidence"] = torch.from_numpy(observability)[None]
    if include_pointpillars_inputs:
        if faulty_lidar_points is None or faulty_lidar_points.ndim != 2 or faulty_lidar_points.shape[1] != 4:
            raise InvalidSampleError(
                f"{sample_path} requires faulty_lidar_points with shape [N,4] "
                "([x,y,z,reflectivity]); regenerate the sample for the "
                "PointPillars experiment"
            )
        if radar_points is None or radar_points.ndim != 2 or radar_points.shape[1] != 5:
            raise InvalidSampleError(
                f"{radar_path} requires radar_points with shape [N,5] "
                "([x,y,z,power,doppler]); rebuild the View-of-Delft radar "
                "cache for the PointPillars experiment"
            )
        item["faulty_lidar_points"] = torch.from_numpy(faulty_lidar_points)
        item["radar_points"] = torch.from_numpy(radar_points)
    return item


def load_bev_grid_geometry(sample_path: str | Path) -> BEVGridGeometry:
    """Read the generated sample's BEV geometry, the experiment source of truth."""

    sample_path = Path(sample_path)
    with np.load(sample_path, allow_pickle=False) as sample:
        metadata = json.loads(str(sample["metadata_json"]))
        clean_shape = np.asarray(sample["clean_rgb"]).shape[:2]
    x_range = tuple(float(value) for value in metadata["x_range"])
    y_range = tuple(float(value) for value in metadata["y_range"])
    geometry = BEVGridGeometry(
        x_min=x_range[0],
        x_max=x_range[1],
        y_min=y_range[0],
        y_max=y_range[1],
        height=int(clean_shape[0]),
        width=int(clean_shape[1]),
    )
    geometry.validate()
    if "resolution" in metadata:
        resolution = float(metadata["resolution"])
        if not (
            np.isclose(geometry.pillar_size_x, resolution)
            and np.isclose(geometry.pillar_size_y, resolution)
        ):
            raise InvalidSampleError(
                "Sample BEV bounds, raster shape, and resolution disagree: "
                f"pillar sizes are {geometry.pillar_size_x:g} and "
                f"{geometry.pillar_size_y:g} m but metadata resolution is "
                f"{resolution:g} m"
            )
    return geometry


def coarse_reconstruction_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    """Collate dense tensors normally while retaining variable point clouds."""

    point_keys = ("faulty_lidar_points", "radar_points")
    dense_batch = [
        {key: value for key, value in item.items() if key not in point_keys}
        for item in batch
    ]
    collated = default_collate(dense_batch)
    for key in point_keys:
        if key in batch[0]:
            collated[key] = tuple(item[key] for item in batch)
    return collated


class CoarseReconstructionDataset(Dataset):
    """Load aligned BEVs and precomputed Stage-I oracle masks."""

    def __init__(
        self,
        sample_paths,
        radar_root: str | Path,
        *,
        data_root: str | Path,
        selector_config: FaultSelectorConfig | None = None,
        use_pointpillars: bool = False,
        augmentation_config: GeometricAugmentationConfig | None = None,
    ):
        self.sample_paths = tuple(Path(path) for path in sample_paths)
        if not self.sample_paths:
            raise FileNotFoundError("No coarse reconstruction samples were provided")
        self.radar_root = Path(radar_root)
        self.data_root = Path(data_root)
        self.selector_config = selector_config or FaultSelectorConfig()
        self.use_pointpillars = bool(use_pointpillars)
        self.grid_geometry = load_bev_grid_geometry(self.sample_paths[0])
        self.augmentation = (
            ReconstructionGeometricAugmentation(
                augmentation_config,
                self.grid_geometry,
            )
            if augmentation_config is not None and augmentation_config.enabled
            else None
        )

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = load_bev_triplet(
            self.sample_paths[index],
            self.radar_root,
            include_pointpillars_inputs=self.use_pointpillars,
        )
        cache_path = selector_cache_path(
            self.sample_paths[index],
            self.data_root,
        )
        cached = load_selector_cache(
            cache_path,
            self.selector_config,
        )
        item.update(
            {
                "reconstruction_mask": torch.from_numpy(
                    cached["reconstruction_mask"]
                )[None],
                "healthy_context_mask": torch.from_numpy(
                    cached["healthy_context_mask"]
                )[None],
                "halo_mask": torch.from_numpy(
                    cached["halo_mask"]
                )[None],
            }
        )
        if self.augmentation is not None:
            item = self.augmentation.apply(item)
        return item
