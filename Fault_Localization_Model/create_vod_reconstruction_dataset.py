"""Create View-of-Delft inputs for the existing PointPillars + HRNet model.

The generated sample and radar-cache contracts intentionally match
``CoarseReconstructionDataset``. Both sensors remain raw point clouds until
the model's two PointPillars encoders create aligned 320x320 pseudo-BEVs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import logging
from pathlib import Path
import random

import numpy as np

from Fault_Localization_Model.bev_utils import (
    HEIGHT_RANGE_M,
    LIDAR_CHANNELS,
    make_rgb_preview,
    metric_to_grid,
    normalize_occupied,
    project_lidar_bev,
)
from Fault_Localization_Model.config.defaults import (
    DEFAULT_FOG_ROOT,
    DEFAULT_INJECTOR_ROOT,
)
from Fault_Localization_Model.data_injection_utils import filter_pointcloud
from Fault_Localization_Model.fault_injector import (
    build_fault_plan,
    inject_fault,
    load_fault_injector,
)
from Fault_Localization_Model.io_utils import atomic_savez_compressed
from Fault_Localization_Model.lidar_observability import (
    LIDAR_SENSOR_ORIGIN,
    create_observability_map,
)
from Fault_Localization_Model.reliability_maps import (
    canonical_maps_for_storage,
    make_reliability_maps,
)
from Fault_Localization_Model.vod_dataset import (
    VODFrame,
    align_radar_to_lidar,
    discover_vod_frames,
    load_vod_lidar,
    load_vod_radar,
    load_vod_radar_to_lidar,
)


LOGGER = logging.getLogger("create_vod_reconstruction_dataset")
GENERATOR_VERSION = 1
DEFAULT_FAULT_PLAN = (("fog_sim", 4), ("fog_sim", 5))
WORKER_CONFIG: dict | None = None
LIDAR_CORRUPTIONS = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create aligned View-of-Delft LiDAR/radar PointPillars inputs "
            "for the existing coarse HRNet reconstruction pipeline."
        )
    )
    parser.add_argument("--vod-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--radar-cache-root", required=True, type=Path)
    parser.add_argument(
        "--split",
        required=True,
        choices=("train", "val", "test", "train_val", "full"),
    )
    parser.add_argument(
        "--radar-variant",
        default="radar_3frames",
        choices=(
            "radar",
            "radar_3frames",
            "radar_5frames",
            "radar_10frames",
            "radar_20frames",
        ),
        help="Use the official accumulated three-scan radar release by default.",
    )
    parser.add_argument("--num-samples", type=int)
    parser.add_argument(
        "--fault-plan",
        nargs="*",
        default=["fog_sim:4", "fog_sim:5"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=64.0)
    parser.add_argument("--y-min", type=float, default=-32.0)
    parser.add_argument("--y-max", type=float, default=32.0)
    parser.add_argument("--resolution", type=float, default=0.2)
    parser.add_argument("--min-range", type=float, default=1.0)
    parser.add_argument("--max-range", type=float, default=80.0)
    parser.add_argument("--movement-tolerance-m", type=float, default=0.05)
    parser.add_argument("--observability-num-z-bins", type=int, default=32)
    parser.add_argument("--observability-ray-support-tau", type=float, default=3.0)
    parser.add_argument(
        "--remove-added-points",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_samples is not None and args.num_samples < 1:
        raise ValueError("num-samples must be positive")
    if args.num_workers < 1:
        raise ValueError("num-workers must be at least one")
    if args.x_max <= args.x_min or args.y_max <= args.y_min:
        raise ValueError("BEV maxima must exceed minima")
    if args.resolution <= 0.0:
        raise ValueError("resolution must be positive")
    shape = (
        int(np.ceil((args.x_max - args.x_min) / args.resolution)),
        int(np.ceil((args.y_max - args.y_min) / args.resolution)),
    )
    if shape != (320, 320):
        raise ValueError(
            "The current reconstruction model requires a 320x320 BEV; "
            f"the requested geometry produces {shape}"
        )


def _worker_init(config: dict) -> None:
    global WORKER_CONFIG, LIDAR_CORRUPTIONS
    WORKER_CONFIG = config
    LIDAR_CORRUPTIONS = load_fault_injector(DEFAULT_INJECTOR_ROOT)


def _within_bev(points: np.ndarray, config: dict) -> np.ndarray:
    return (
        (points[:, 0] >= config["x_min"])
        & (points[:, 0] < config["x_max"])
        & (points[:, 1] >= config["y_min"])
        & (points[:, 1] < config["y_max"])
    )


def _radar_bev(aligned: np.ndarray, config: dict) -> np.ndarray:
    """Create the compatibility/debug raster; PointPillars uses raw points."""

    xyz, rows, cols, valid, height, width = metric_to_grid(
        aligned[:, :3],
        (config["x_min"], config["x_max"]),
        (config["y_min"], config["y_max"]),
        config["resolution"],
    )
    output = np.zeros((4, height, width), dtype=np.float32)
    if not len(xyz):
        return output
    source = aligned[valid]
    density = np.zeros((height, width), dtype=np.float32)
    np.add.at(density, (rows, cols), 1.0)
    occupied = density > 0
    output[0] = occupied
    logged = np.log1p(density)
    if logged.max(initial=0.0) > 0.0:
        output[1] = logged / logged.max()
    np.maximum.at(
        output[2],
        (rows, cols),
        np.clip(np.abs(source[:, 5]) / 30.0, 0.0, 1.0),
    )
    rcs = np.full((height, width), -np.inf, dtype=np.float32)
    np.maximum.at(rcs, (rows, cols), source[:, 3])
    output[3] = normalize_occupied(rcs, occupied)
    return output


def _write_radar_cache(frame: VODFrame, aligned: np.ndarray, config: dict) -> Path:
    destination = (
        Path(config["radar_cache_root"])
        / frame.split
        / f"{int(frame.frame_id):05d}.npz"
    )
    if destination.is_file():
        try:
            with np.load(destination, allow_pickle=False) as cached:
                if (
                    cached["radar_bev"].shape == (4, 320, 320)
                    and cached["radar_points"].ndim == 2
                    and cached["radar_points"].shape[1] == 5
                ):
                    return destination
        except Exception:
            pass

    # Existing model field names are [x,y,z,power,doppler]. For VoD, RCS is
    # the radar-strength feature and compensated radial velocity is Doppler.
    pointpillars_points = np.column_stack(
        (aligned[:, :3], aligned[:, 3], aligned[:, 5])
    ).astype(np.float32, copy=False)
    metadata = {
        "cache_format_version": 1,
        "dataset": "View-of-Delft",
        "frame_id": frame.frame_id,
        "split": frame.split,
        "radar_variant": frame.radar_variant,
        "radar_source": str(frame.radar_path),
        "source_fields": [
            "x",
            "y",
            "z",
            "rcs",
            "radial_velocity",
            "compensated_radial_velocity",
            "time_index",
        ],
        "pointpillars_radar_fields": [
            "x_lidar",
            "y_lidar",
            "z_lidar",
            "rcs",
            "compensated_radial_velocity",
        ],
        "x_range": [config["x_min"], config["x_max"]],
        "y_range": [config["y_min"], config["y_max"]],
        "resolution": config["resolution"],
    }
    atomic_savez_compressed(
        destination,
        radar_bev=_radar_bev(aligned, config).astype(np.float16),
        radar_points=pointpillars_points,
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    return destination


def _task_output_path(task: dict, config: dict) -> Path:
    return (
        Path(config["output_root"])
        / task["frame"]["split"]
        / (
            f"{int(task['frame']['frame_id']):05d}_"
            f"{task['fault']}_s{task['severity']}.npz"
        )
    )


def _create_sample(task: dict) -> dict:
    if WORKER_CONFIG is None or LIDAR_CORRUPTIONS is None:
        raise RuntimeError("VoD worker was not initialized")
    config = WORKER_CONFIG
    frame = VODFrame(
        **{
            key: Path(value) if key.endswith("_path") else value
            for key, value in task["frame"].items()
        }
    )
    destination = _task_output_path(task, config)
    if destination.is_file():
        return {"path": str(destination), "cached": True}

    lidar = load_vod_lidar(frame.lidar_path)
    _, range_mask = filter_pointcloud(
        lidar, config["min_range"], config["max_range"], return_mask=True
    )
    clean = lidar[range_mask & _within_bev(lidar, config)]
    if not len(clean):
        raise ValueError(f"No VoD LiDAR points remain in the BEV for {frame.frame_id}")

    clean_ids = np.arange(len(clean), dtype=np.int64)
    injection, injection_metadata = inject_fault(
        task["fault"],
        clean.copy(),
        clean_ids,
        task["severity"],
        DEFAULT_INJECTOR_ROOT,
        DEFAULT_FOG_ROOT,
        lidar_corruptions=LIDAR_CORRUPTIONS,
        rng_seed=task["injection_seed"],
    )
    _, faulty_range = filter_pointcloud(
        injection.points,
        config["min_range"],
        config["max_range"],
        return_mask=True,
    )
    keep = faulty_range & _within_bev(injection.points, config)
    if config["remove_added_points"]:
        keep &= injection.source_ids >= 0
    faulty = injection.points[keep]
    faulty_ids = injection.point_ids[keep]
    faulty_source_ids = injection.source_ids[keep]
    faulty_labels = injection.injector_labels[keep]

    maps = make_reliability_maps(
        clean,
        clean_ids,
        faulty,
        faulty_ids,
        faulty_source_ids,
        config["movement_tolerance_m"],
        config["x_min"],
        config["x_max"],
        config["y_min"],
        config["y_max"],
        320,
        320,
    )
    geometry = {
        "x_range": (config["x_min"], config["x_max"]),
        "y_range": (config["y_min"], config["y_max"]),
        "resolution": config["resolution"],
    }
    clean_layers = project_lidar_bev(clean, **geometry)
    faulty_layers = project_lidar_bev(faulty, **geometry)
    observability = create_observability_map(
        clean,
        LIDAR_SENSOR_ORIGIN,
        z_range=HEIGHT_RANGE_M,
        num_z_bins=config["observability_num_z_bins"],
        ray_support_tau=config["observability_ray_support_tau"],
        **geometry,
    )

    radar = load_vod_radar(frame.radar_path)
    lidar_from_radar = load_vod_radar_to_lidar(
        frame.lidar_calibration_path,
        frame.radar_calibration_path,
    )
    aligned_radar = align_radar_to_lidar(radar, lidar_from_radar)
    _write_radar_cache(frame, aligned_radar, config)

    metadata = {
        "dataset": "View-of-Delft",
        "split": frame.split,
        "frame_id": frame.frame_id,
        "scene": "View-of-Delft",
        "session": frame.split,
        "sequence": "",
        "lidar_index": frame.frame_id,
        "radar_index": frame.frame_id,
        "timestamp": frame.frame_id,
        "timestamp_ns": int(frame.frame_id),
        "source_relative_path": str(frame.lidar_path),
        "source_lidar_dir": str(frame.lidar_path.parent),
        "label_relative_path": "",
        "radar_relative_path": str(frame.radar_path),
        "radar_variant": frame.radar_variant,
        "radar_from_lidar": np.linalg.inv(lidar_from_radar).tolist(),
        "fault": task["fault"],
        "severity": task["severity"],
        "x_range": [config["x_min"], config["x_max"]],
        "y_range": [config["y_min"], config["y_max"]],
        "resolution": config["resolution"],
        "grid_size": 320,
        "image_height": 320,
        "image_width": 320,
        "lidar_channels": list(LIDAR_CHANNELS),
        "pointpillars_lidar_fields": ["x", "y", "z", "reflectivity"],
        "pointpillars_radar_fields": [
            "x_lidar",
            "y_lidar",
            "z_lidar",
            "rcs",
            "compensated_radial_velocity",
        ],
        "spatial_support": "shared front 64m x 64m Cartesian BEV",
        "generator_version": GENERATOR_VERSION,
        "remove_added_points": config["remove_added_points"],
        "injection_seed": task["injection_seed"],
        "injection_metadata": injection_metadata,
    }
    atomic_savez_compressed(
        destination,
        **canonical_maps_for_storage(maps),
        clean_rgb=make_rgb_preview(clean_layers),
        clean_density=clean_layers["raw_density"],
        faulty_rgb=make_rgb_preview(faulty_layers),
        faulty_density=faulty_layers["raw_density"],
        clean_point_ids=clean_ids,
        faulty_point_ids=faulty_ids,
        faulty_source_ids=faulty_source_ids,
        faulty_injector_labels=faulty_labels,
        faulty_lidar_points=faulty[:, :4].astype(np.float32, copy=False),
        observability_confidence=observability[
            "observability_confidence"
        ].astype(np.float16),
        observability_ray_count=observability["ray_count"].astype(np.uint32),
        observability_vertical_coverage=observability[
            "vertical_coverage"
        ].astype(np.float16),
        observability_ray_support=observability["ray_support"].astype(np.float16),
        valid_support_mask=np.ones((320, 320), dtype=np.uint8),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    return {"path": str(destination), "cached": False}


def _serialize_frame(frame: VODFrame) -> dict:
    return {
        "frame_id": frame.frame_id,
        "split": frame.split,
        "lidar_path": str(frame.lidar_path),
        "radar_path": str(frame.radar_path),
        "lidar_calibration_path": str(frame.lidar_calibration_path),
        "radar_calibration_path": str(frame.radar_calibration_path),
        "radar_variant": frame.radar_variant,
    }


def main() -> None:
    args = parse_args()
    _validate_args(args)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    frames = discover_vod_frames(
        args.vod_root,
        args.split,
        radar_variant=args.radar_variant,
    )
    rng = random.Random(args.seed)
    rng.shuffle(frames)
    count = len(frames) if args.num_samples is None else args.num_samples
    if count > len(frames):
        raise ValueError(
            f"Requested {count} unique samples from only {len(frames)} {args.split} frames"
        )
    frames = frames[:count]
    plan = build_fault_plan(args.fault_plan, None, None, DEFAULT_FAULT_PLAN)

    config = {
        "output_root": str(args.output_root),
        "radar_cache_root": str(args.radar_cache_root),
        "x_min": args.x_min,
        "x_max": args.x_max,
        "y_min": args.y_min,
        "y_max": args.y_max,
        "resolution": args.resolution,
        "min_range": args.min_range,
        "max_range": args.max_range,
        "movement_tolerance_m": args.movement_tolerance_m,
        "observability_num_z_bins": args.observability_num_z_bins,
        "observability_ray_support_tau": args.observability_ray_support_tau,
        "remove_added_points": args.remove_added_points,
    }
    tasks = []
    for index, frame in enumerate(frames):
        fault, severity = plan[index % len(plan)]
        injection_seed = int(
            np.random.SeedSequence([args.seed, index]).generate_state(1)[0]
        )
        tasks.append(
            {
                "frame": _serialize_frame(frame),
                "fault": fault,
                "severity": severity,
                "injection_seed": injection_seed,
            }
        )
    LOGGER.info(
        "Generating %d %s samples using %s and %d workers",
        len(tasks),
        args.split,
        args.radar_variant,
        args.num_workers,
    )
    created = cached = 0
    if args.num_workers == 1:
        _worker_init(config)
        results = map(_create_sample, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=_worker_init,
            initargs=(config,),
        )
        results = executor.map(_create_sample, tasks, chunksize=1)
    try:
        for completed, result in enumerate(results, 1):
            cached += int(result["cached"])
            created += int(not result["cached"])
            if completed % 100 == 0 or completed == len(tasks):
                LOGGER.info(
                    "Processed %d/%d; created=%d cached=%d",
                    completed,
                    len(tasks),
                    created,
                    cached,
                )
    finally:
        if executor is not None:
            executor.shutdown()
    LOGGER.info("Samples: %s", Path(args.output_root) / args.split)
    LOGGER.info("Radar cache: %s", Path(args.radar_cache_root) / args.split)


if __name__ == "__main__":
    main()
