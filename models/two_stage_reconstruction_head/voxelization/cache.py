"""Versioned deterministic cache for standalone 3D voxel artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from Fault_Localization_Model.io_utils import atomic_savez

from .config import VoxelizationConfig
from .hard_voxelizer import VoxelizedPointCloud


class InvalidVoxelCacheError(RuntimeError):
    pass


def cache_path(
    cache_root: str | Path,
    modality: str,
    split: str,
    identifier: str,
) -> Path:
    if modality not in {"lidar", "radar"}:
        raise ValueError("modality must be lidar or radar")
    return Path(cache_root) / modality / split / f"{identifier}.npz"


def cache_metadata(
    result: VoxelizedPointCloud,
    config: VoxelizationConfig,
    *,
    modality: str,
    source_path: str,
    sample_path: str,
    split: str,
    frame_id: str,
    lidar_source: str,
) -> dict:
    return {
        "cache_version": config.cache_version,
        "config_hash": config.fingerprint,
        "config": config.to_dict(),
        "modality": modality,
        "coordinate_frame": "lidar",
        "point_coordinate_order": "xyz",
        "sparse_coordinate_order": "zyx",
        "source_path": source_path,
        "sample_path": sample_path,
        "split": split,
        "frame_id": frame_id,
        "lidar_source": lidar_source,
        "raw_feature_names": list(result.raw_feature_names),
        "feature_names": list(result.feature_names),
        "input_point_count": result.input_point_count,
        "valid_point_count": result.valid_point_count,
        "nonfinite_point_count": result.nonfinite_point_count,
        "out_of_range_point_count": result.out_of_range_point_count,
        "truncated_point_count": result.truncated_point_count,
        "occupied_voxel_count": result.occupied_voxel_count,
    }


def write_voxel_cache(
    path: str | Path,
    result: VoxelizedPointCloud,
    metadata: dict,
    *,
    compression_level: int,
) -> None:
    atomic_savez(
        path,
        compression_level=compression_level,
        voxel_coords=result.voxel_coords,
        voxel_points=result.voxel_points,
        num_points=result.num_points,
        original_num_points=result.original_num_points,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def load_voxel_cache(
    path: str | Path,
    config: VoxelizationConfig,
    *,
    modality: str,
) -> tuple[VoxelizedPointCloud, dict]:
    path = Path(path)
    if not path.is_file():
        raise InvalidVoxelCacheError(f"Voxel cache is missing: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if int(metadata["cache_version"]) != config.cache_version:
                raise InvalidVoxelCacheError(f"Voxel cache version is stale: {path}")
            if metadata["config_hash"] != config.fingerprint:
                raise InvalidVoxelCacheError(f"Voxel cache configuration is stale: {path}")
            if metadata["modality"] != modality:
                raise InvalidVoxelCacheError(f"Voxel cache modality is wrong: {path}")
            result = VoxelizedPointCloud(
                voxel_coords=np.asarray(archive["voxel_coords"], dtype=np.int32),
                voxel_points=np.asarray(archive["voxel_points"], dtype=np.float32),
                num_points=np.asarray(archive["num_points"], dtype=np.int32),
                original_num_points=np.asarray(
                    archive["original_num_points"], dtype=np.int32
                ),
                raw_feature_names=tuple(metadata["raw_feature_names"]),
                feature_names=tuple(metadata["feature_names"]),
                input_point_count=int(metadata["input_point_count"]),
                valid_point_count=int(metadata["valid_point_count"]),
                nonfinite_point_count=int(metadata["nonfinite_point_count"]),
                out_of_range_point_count=int(metadata["out_of_range_point_count"]),
                truncated_point_count=int(metadata["truncated_point_count"]),
            )
    except InvalidVoxelCacheError:
        raise
    except Exception as exc:
        raise InvalidVoxelCacheError(f"Cannot load voxel cache {path}: {exc}") from exc
    return result, metadata
