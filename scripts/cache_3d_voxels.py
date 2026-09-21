"""Build separate deterministic LiDAR and radar 3D voxel caches."""

from __future__ import annotations

import argparse
from pathlib import Path

from voxelization import (
    HardVoxelizer,
    load_voxelization_config,
)
from voxelization.cache import (
    cache_metadata,
    cache_path,
    load_voxel_cache,
    write_voxel_cache,
    InvalidVoxelCacheError,
)
from voxelization.inputs import (
    discover_sample_paths,
    load_aligned_point_inputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--radar-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/voxelization_3d.json"))
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lidar-source", choices=("clean", "faulty"), default="clean")
    parser.add_argument("--modalities", nargs="+", choices=("lidar", "radar"), default=("lidar", "radar"))
    args = parser.parse_args()

    config = load_voxelization_config(args.config)
    paths = discover_sample_paths(
        args.data_root, args.split, limit=args.limit_samples, seed=args.seed,
        unique_frames=args.lidar_source == "clean",
    )
    voxelizers = {
        "lidar": HardVoxelizer(
            config.grid, max_points_per_voxel=config.lidar.max_points_per_voxel
        ),
        "radar": HardVoxelizer(
            config.grid, max_points_per_voxel=config.radar.max_points_per_voxel
        ),
    }
    created = cached = 0
    for index, sample_path in enumerate(paths, start=1):
        inputs = load_aligned_point_inputs(
            sample_path, args.radar_root, lidar_source=args.lidar_source
        )
        frame_id = str(inputs.metadata["frame_id"])
        for modality in args.modalities:
            # Radar is a property of the physical frame and is shared by all
            # fault variants. Faulty LiDAR is sample-specific; clean LiDAR is
            # likewise shared by frame.
            identifier = (
                frame_id
                if modality == "radar" or args.lidar_source == "clean"
                else sample_path.stem
            )
            destination = cache_path(args.cache_root, modality, args.split, identifier)
            try:
                load_voxel_cache(destination, config, modality=modality)
                cached += 1
                continue
            except InvalidVoxelCacheError:
                pass
            points = getattr(inputs, f"{modality}_points")
            names = getattr(inputs, f"{modality}_feature_names")
            result = voxelizers[modality].voxelize(points, names)
            source_path = (
                str(inputs.metadata["source_relative_path"])
                if modality == "lidar" and args.lidar_source == "clean"
                else str(inputs.radar_path if modality == "radar" else sample_path)
            )
            metadata = cache_metadata(
                result,
                config,
                modality=modality,
                source_path=source_path,
                sample_path=str(sample_path),
                split=args.split,
                frame_id=frame_id,
                lidar_source=args.lidar_source,
            )
            write_voxel_cache(
                destination, result, metadata,
                compression_level=config.compression_level,
            )
            created += 1
        if index == 1 or index % 100 == 0 or index == len(paths):
            print(
                f"Processed {index}/{len(paths)} frames; created={created} cached={cached}",
                flush=True,
            )


if __name__ == "__main__":
    main()
