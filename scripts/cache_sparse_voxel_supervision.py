"""Cache model-ready sparse 3D selector components for diffusion training."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np

from Fault_Localization_Model.io_utils import atomic_savez
from models.two_stage_reconstruction_head.diffusion_process import (
    build_sparse_voxel_example,
)
from voxelization import (
    HardVoxelizer,
    OracleFaultSelector3DConfig,
    build_voxel_fault_targets,
    load_voxelization_config,
    select_oracle_fault_regions_3d,
)
from voxelization.inputs import (
    discover_sample_paths,
    load_aligned_point_inputs,
    load_clean_lidar_from_metadata,
)
from voxelization.hard_voxelizer import VoxelizedPointCloud


CACHE_VERSION = 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--radar-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/voxelization_3d.json"))
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        parser.error("fraction must be in (0, 1]")
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.limit_samples is not None and args.limit_samples < 1:
        parser.error("limit-samples must be positive")
    return args


def _destination(cache_root: Path, split: str, sample_path: Path) -> Path:
    return cache_root / split / f"{sample_path.stem}.npz"


def _valid(path: Path, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            return (
                int(archive["cache_version"].item()) == CACHE_VERSION
                and str(archive["config_hash"].item()) == fingerprint
            )
    except Exception:
        return False


def _write(path: Path, examples, *, config_hash: str, sample_path: Path, frame_id: str, selector_config: OracleFaultSelector3DConfig) -> None:
    offsets = [0]
    for example in examples:
        offsets.append(offsets[-1] + len(example.coords_zyx))
    if examples:
        coords = np.concatenate([item.coords_zyx.numpy() for item in examples], axis=0)
        xyz = np.concatenate([item.coords_xyz_m.numpy() for item in examples], axis=0)
        condition = np.concatenate([item.condition_features.numpy() for item in examples], axis=0)
        target = np.concatenate([item.target_occupancy.numpy() for item in examples], axis=0)
        faulty = np.concatenate([item.faulty_occupancy.numpy() for item in examples], axis=0)
        editable = np.concatenate([item.editable_mask.numpy() for item in examples], axis=0)
    else:
        coords = np.empty((0, 3), dtype=np.int64)
        xyz = np.empty((0, 3), dtype=np.float32)
        condition = np.empty((0, 6), dtype=np.float32)
        target = faulty = editable = np.empty((0, 1), dtype=np.float32)
    metadata = {
        "sample_path": str(sample_path),
        "frame_id": frame_id,
        "selector_config": asdict(selector_config),
        "component_count": len(examples),
    }
    atomic_savez(
        path,
        compression_level=6,
        cache_version=np.asarray(CACHE_VERSION, dtype=np.int64),
        config_hash=np.asarray(config_hash),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        offsets=np.asarray(offsets, dtype=np.int64),
        coords_zyx=coords,
        coords_xyz_m=xyz,
        condition_features=condition,
        target_occupancy=target,
        faulty_occupancy=faulty,
        editable_mask=editable,
    )


def _count_voxelize(points: np.ndarray, names: tuple[str, ...], voxelizer: HardVoxelizer) -> VoxelizedPointCloud:
    """Return only the sparse evidence consumed by the baseline.

    The model conditions on occupied coordinates and original point counts, not
    per-point embeddings.  Avoiding ``HardVoxelizer.voxelize``'s per-voxel
    Python loop and padded point tensor substantially accelerates cache builds.
    """

    coordinates, valid = voxelizer.point_indices(points)
    valid_coords = coordinates[valid]
    finite = np.isfinite(points).all(axis=1)
    if not len(valid_coords):
        empty = np.empty((0,), dtype=np.int32)
        return VoxelizedPointCloud(
            voxel_coords=np.empty((0, 3), dtype=np.int32),
            voxel_points=np.empty((0, 0, len(names) + 6), dtype=np.float32),
            num_points=empty,
            original_num_points=empty,
            raw_feature_names=names,
            feature_names=names,
            input_point_count=len(points), valid_point_count=0,
            nonfinite_point_count=int((~finite).sum()),
            out_of_range_point_count=int((finite & ~valid).sum()),
            truncated_point_count=0,
        )
    nx, ny, _ = voxelizer.grid.dimensions_xyz
    flat = (valid_coords[:, 0].astype(np.int64) * ny + valid_coords[:, 1]) * nx + valid_coords[:, 2]
    unique, counts = np.unique(flat, return_counts=True)
    iz = unique // (ny * nx)
    remainder = unique % (ny * nx)
    coordinates = np.stack((iz, remainder // nx, remainder % nx), axis=1).astype(np.int32)
    counts = counts.astype(np.int32)
    return VoxelizedPointCloud(
        voxel_coords=coordinates,
        voxel_points=np.empty((len(coordinates), 0, len(names) + 6), dtype=np.float32),
        num_points=counts,
        original_num_points=counts,
        raw_feature_names=names,
        feature_names=names,
        input_point_count=len(points), valid_point_count=int(valid.sum()),
        nonfinite_point_count=int((~finite).sum()),
        out_of_range_point_count=int((finite & ~valid).sum()),
        truncated_point_count=0,
    )


def _cache_one(
    sample_path: str,
    *,
    radar_root: str,
    cache_root: str,
    config_path: str,
    split: str,
) -> str:
    """Build one independent entry; safe to execute in a worker process."""

    sample_path = Path(sample_path)
    config = load_voxelization_config(config_path)
    selector_config = OracleFaultSelector3DConfig()
    lidar_voxelizer = HardVoxelizer(config.grid, max_points_per_voxel=config.lidar.max_points_per_voxel)
    radar_voxelizer = HardVoxelizer(config.grid, max_points_per_voxel=config.radar.max_points_per_voxel)
    destination = _destination(Path(cache_root), split, sample_path)
    if _valid(destination, config.fingerprint):
        return "cached"
    inputs = load_aligned_point_inputs(sample_path, radar_root, lidar_source="faulty")
    with np.load(sample_path, allow_pickle=False) as archive:
        if "faulty_source_ids" not in archive.files:
            raise ValueError(
                f"{sample_path} lacks faulty_source_ids and cannot produce exact "
                "3D oracle supervision. Regenerate it with generator version 3 "
                "or later into a new output root."
            )
        source_ids = np.asarray(archive["faulty_source_ids"], dtype=np.int64)
    # The metadata-aware loader selects the correct binary decoder for VoD or
    # HeRCULES.  This is deliberately the only dataset-specific operation;
    # voxel fault targets and oracle 3D selection remain identical.
    clean_points = load_clean_lidar_from_metadata(inputs.metadata)
    targets = build_voxel_fault_targets(
        clean_points, inputs.lidar_points, source_ids, config.grid
    )
    selection = select_oracle_fault_regions_3d(
        targets.repair_mask, targets.remove_mask, config.grid, selector_config
    )
    faulty_voxels = _count_voxelize(
        inputs.lidar_points, inputs.lidar_feature_names, lidar_voxelizer
    )
    radar_voxels = _count_voxelize(
        inputs.radar_points, inputs.radar_feature_names, radar_voxelizer
    )
    examples = tuple(
        build_sparse_voxel_example(
            faulty_lidar=faulty_voxels,
            radar=radar_voxels,
            targets=targets,
            selection=selection,
            component=component,
            grid=config.grid,
        )
        for component in selection.components
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write(
        destination, examples, config_hash=config.fingerprint,
        sample_path=sample_path, frame_id=str(inputs.metadata["frame_id"]),
        selector_config=selector_config,
    )
    return "created"


def main() -> None:
    args = _parse_args()
    paths = discover_sample_paths(args.data_root, args.split, unique_frames=False)
    if args.fraction < 1.0:
        count = max(1, round(len(paths) * args.fraction))
        paths = sorted(random.Random(args.seed).sample(paths, count))
    if args.limit_samples is not None:
        paths = paths[:args.limit_samples]
    common = {
        "radar_root": str(args.radar_root),
        "cache_root": str(args.cache_root),
        "config_path": str(args.config),
        "split": args.split,
    }
    created = cached = 0
    if args.workers == 1:
        results = (_cache_one(str(path), **common) for path in paths)
        iterator = enumerate(results, start=1)
        for index, result in iterator:
            created += result == "created"
            cached += result == "cached"
            if index == 1 or index % 25 == 0 or index == len(paths):
                print(f"Processed {index}/{len(paths)}; created={created} cached={cached}", flush=True)
        return
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_cache_one, str(path), **common) for path in paths]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            created += result == "created"
            cached += result == "cached"
            if index == 1 or index % 25 == 0 or index == len(paths):
                print(f"Processed {index}/{len(paths)}; created={created} cached={cached}", flush=True)


if __name__ == "__main__":
    main()
