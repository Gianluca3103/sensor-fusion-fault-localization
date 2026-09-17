"""Create View-of-Delft inputs for the existing PointPillars + HRNet model.

The generated sample and radar-cache contracts intentionally match
``CoarseReconstructionDataset``. Both sensors remain raw point clouds until
the model's two PointPillars encoders create aligned 320x320 pseudo-BEVs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing
import os
import json
import logging
import hashlib
from pathlib import Path
import random
import time

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
from Fault_Localization_Model.io_utils import atomic_savez, atomic_write_json
from Fault_Localization_Model.lidar_observability import (
    LIDAR_SENSOR_ORIGIN,
    create_observability_map,
    warm_observability_backend,
)
from Fault_Localization_Model.reliability_maps import (
    canonical_maps_for_storage,
    make_reliability_maps,
    training_maps_for_storage,
)
from Fault_Localization_Model.vod_dataset import (
    BEVGeometry,
    ENGINEERED_LIDAR_CHANNELS,
    ENGINEERED_RADAR_CHANNELS,
    SUPPORTED_RADAR_VARIANTS,
    VODFrame,
    align_radar_to_lidar,
    discover_vod_frames,
    load_vod_lidar,
    load_vod_radar,
    load_vod_radar_to_lidar,
    lidar_model_channels,
    radar_model_channels,
)


LOGGER = logging.getLogger("create_vod_reconstruction_dataset")
GENERATOR_VERSION = 2
DEFAULT_FAULT_PLAN = (
    ("fog_sim", 4),
    ("fog_sim", 5),
    ("fov_filter", 1),
    ("total_loss", 1),
)
WORKER_CONFIG: dict | None = None
LIDAR_CORRUPTIONS = None


def _bounded_results(executor, function, tasks, max_pending):
    """Report completed work without queuing the entire dataset or blocking
    behind a slow earlier frame. Per-task seeds/paths are already fixed."""
    iterator = iter(tasks)
    pending = set()
    for _ in range(max_pending):
        task = next(iterator, None)
        if task is None:
            break
        pending.add(executor.submit(function, task))
    while pending:
        ready, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in ready:
            yield future.result()
            task = next(iterator, None)
            if task is not None:
                pending.add(executor.submit(function, task))


def _chronological_tasks(tasks):
    return sorted(tasks, key=lambda task: (task['frame']['radar_path'],
                                           int(Path(task['frame']['lidar_path']).stem)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create aligned View-of-Delft LiDAR/radar PointPillars inputs "
            "for the existing coarse HRNet reconstruction pipeline."
        )
    )
    parser.add_argument('--allow-slow-observability', action='store_true',
                        help='Explicitly permit the Python reference backend when Numba is missing.')
    parser.add_argument('--npz-compression-level', type=int, choices=range(10), default=6,
                        help='Lossless ZIP effort: 0 uncompressed, 1 fast, 6 standard (default).')
    parser.add_argument(
        '--artifact-profile', choices=('training', 'full'), default='training',
        help=(
            'training (default) omits unused per-point provenance and dense '
            'diagnostic arrays; full preserves every generation diagnostic.'
        ),
    )
    parser.add_argument('--hercules-generation-order', choices=('chronological', 'random'),
                        default='chronological', help='Scheduling only; faults/seeds are assigned before ordering.')
    parser.add_argument('--skip-invalid-synchronization', action='store_true',
                        help='HeRCULES only: log/skip frames outside measured radar/pose coverage.')
    roots = parser.add_mutually_exclusive_group(required=True)
    roots.add_argument("--vod-root", type=Path)
    roots.add_argument("--hercules-root", type=Path)
    parser.add_argument('--hercules-split-manifest', type=Path,
                        help='Optional scene-held-out HeRCULES split manifest')
    parser.add_argument("--hercules-radar-frames", type=int, default=0,
                        help="V2 optional frame cap; 0 uses adaptive pose/history gates only")
    parser.add_argument("--hercules-temporal-radius", type=float, default=0.75)
    parser.add_argument('--hercules-max-radar-age-ms', type=float, default=30.0,
                        help='Maximum age of newest past radar scan (no future scans).')
    parser.add_argument('--hercules-max-pose-gap-ms', type=float, default=200.0,
                        help='Maximum measured pose interval used for interpolation/velocity.')
    parser.add_argument("--hercules-max-history-s", type=float, default=1.0)
    parser.add_argument("--hercules-max-translation-m", type=float, default=4.0)
    parser.add_argument("--hercules-max-rotation-deg", type=float, default=5.0)
    parser.add_argument("--hercules-doppler-sign", choices=('auto', '1', '-1'), default='auto')
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--radar-cache-root", required=True, type=Path)
    parser.add_argument(
        "--radar-cache-only",
        action="store_true",
        help="Build aligned Radar BEV caches without creating LiDAR fault samples.",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=("train", "val", "test", "train_val", "full"),
    )
    parser.add_argument(
        "--radar-variant",
        default="radar_3frames",
        choices=SUPPORTED_RADAR_VARIANTS,
        help="Use the official accumulated three-scan radar release by default.",
    )
    parser.add_argument(
        "--bev-channel-profile",
        choices=("baseline", "engineered"),
        default="baseline",
        help=(
            "baseline keeps the existing 3-channel LiDAR/4-channel Radar "
            "rasters; engineered writes the 6-channel LiDAR and 7-channel "
            "Radar direct-BEV ablation inputs"
        ),
    )
    parser.add_argument("--num-samples", type=int)
    parser.add_argument(
        "--fault-plan",
        nargs="*",
        default=["fog_sim:4", "fog_sim:5", "fov_filter:1", "total_loss:1"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int,
                        help='Default: up to 8 CPU workers for HeRCULES, 4 for VoD.')
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
    if args.hercules_radar_frames < 0 or args.hercules_temporal_radius <= 0:
        raise ValueError("HeRCULES frame cap must be nonnegative and temporal radius positive")
    if args.hercules_root:
        from Fault_Localization_Model.hercules_radar_types import AdaptiveStackConfig
        AdaptiveStackConfig(max_age_s=args.hercules_max_history_s,
            max_translation_m=args.hercules_max_translation_m,
            max_rotation_deg=args.hercules_max_rotation_deg).validate()
    if not args.radar_cache_only and args.output_root is None:
        raise ValueError("output-root is required unless --radar-cache-only is used")
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


def _radar_cache_worker_init(config: dict) -> None:
    global WORKER_CONFIG
    WORKER_CONFIG = config


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
    weights = config.get('_hercules_point_weights')
    np.add.at(density, (rows, cols), weights[valid] if weights is not None else 1.0)
    if weights is not None:
        density /= max(config['_hercules_alignment']['effective_frame_support'], 1e-6)
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


def _load_frame_radar(frame, config):
    if config.get("dataset") == "HeRCULES":
        from Fault_Localization_Model.hercules_dataset import load_frame_radar
        return load_frame_radar(frame, config)
    radar = load_vod_radar(frame.radar_path)
    transform = load_vod_radar_to_lidar(
        frame.lidar_calibration_path, frame.radar_calibration_path,
    )
    return radar, align_radar_to_lidar(radar, transform), transform


def _write_radar_cache(
    frame: VODFrame,
    raw: np.ndarray,
    aligned: np.ndarray,
    config: dict,
) -> Path:
    destination = (
        Path(config["radar_cache_root"])
        / frame.split
        / f"{int(frame.frame_id):05d}.npz"
    )
    if destination.is_file():
        try:
            with np.load(destination, allow_pickle=False) as cached:
                expected_channels = (
                    7 if config["bev_channel_profile"] == "engineered" else 4
                )
                metadata = json.loads(str(cached["metadata_json"].item()))
                if (
                    cached["radar_bev"].shape == (expected_channels, 320, 320)
                    and cached["radar_points"].ndim == 2
                    and cached["radar_points"].shape[1] == 5
                    and metadata.get("radar_variant") == frame.radar_variant
                    and metadata.get("radar_source") == str(frame.radar_path)
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
        "artifact_profile": "training",
        "artifact_compression_level": config.get("npz_compression_level", 6),
        "dataset": config.get("dataset", "View-of-Delft"),
        "native_sensor": "Continental" if config.get("dataset") == "HeRCULES" else "VoD radar",
        "hercules_alignment": config.get('_hercules_alignment', {}),
        "preprocessing": {
            "stack_frames": config.get("hercules_radar_frames"),
            "temporal_radius_m": config.get("hercules_temporal_radius"),
            "alignment": "V2 adaptive pose interpolation and tracked dynamic points, current Aeva axes",
            "doppler_convention": config.get('hercules_tracking', {}).get('doppler_sign', 'auto'),
        } if config.get("dataset") == "HeRCULES" else {},
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
        "bev_channel_profile": config["bev_channel_profile"],
        "radar_bev_channels": (
            list(ENGINEERED_RADAR_CHANNELS)
            if config["bev_channel_profile"] == "engineered"
            else [
                "occupancy",
                "log_density",
                "absolute_compensated_radial_velocity",
                "normalized_rcs",
            ]
        ),
    }
    geometry = BEVGeometry(
        x_range=(config["x_min"], config["x_max"]),
        y_range=(config["y_min"], config["y_max"]),
        resolution=config["resolution"],
    )
    radar_bev = (
        radar_model_channels(raw, aligned, geometry)
        if config["bev_channel_profile"] == "engineered"
        else _radar_bev(aligned, config)
    )
    atomic_savez(
        destination,
        compression_level=config.get('npz_compression_level', 6),
        radar_bev=radar_bev.astype(np.float16),
        radar_points=pointpillars_points,
        **({"radar_point_weights": config['_hercules_point_weights']} if config.get('dataset') == 'HeRCULES' else {}),
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


def _create_sample_impl(task: dict) -> dict:
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
        if config.get('dataset') != 'HeRCULES':
            return {"path": str(destination), "cached": True}
        with np.load(destination, allow_pickle=False) as previous:
            previous_metadata = json.loads(str(previous['metadata_json'].item()))
        if (previous_metadata.get('radar_variant') == frame.radar_variant
                and previous_metadata.get('source_relative_path') == str(frame.lidar_path)):
            return {"path": str(destination), "cached": True}

    if config.get("dataset") == "HeRCULES":
        from Fault_Localization_Model.hercules_dataset import load_hercules_lidar
        lidar = load_hercules_lidar(frame.lidar_path)
    else:
        lidar = load_vod_lidar(frame.lidar_path)
    _, range_mask = filter_pointcloud(
        lidar, config["min_range"], config["max_range"], return_mask=True
    )
    clean = lidar[range_mask & _within_bev(lidar, config)]
    if not len(clean):
        raise ValueError(f"No VoD LiDAR points remain in the BEV for {frame.frame_id}")

    # Validate/load alignment before expensive injection and ray tracing. This
    # changes no arrays; it only fails unsupported frames earlier.
    radar, aligned_radar, lidar_from_radar = _load_frame_radar(frame, config)

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
    engineered_lidar = None
    if config["bev_channel_profile"] == "engineered":
        engineered_lidar = lidar_model_channels(
            faulty[:, :4],
            BEVGeometry(**geometry),
        )
    observability = create_observability_map(
        clean,
        LIDAR_SENSOR_ORIGIN,
        z_range=HEIGHT_RANGE_M,
        num_z_bins=config["observability_num_z_bins"],
        ray_support_tau=config["observability_ray_support_tau"],
        **geometry,
    )

    _write_radar_cache(frame, radar, aligned_radar, config)

    metadata = {
        "dataset": config.get("dataset", "View-of-Delft"),
        "split": frame.split,
        "frame_id": frame.frame_id,
        "scene": frame.radar_path.parent.name if config.get("dataset") == "HeRCULES" else "View-of-Delft",
        "session": frame.radar_path.name if config.get("dataset") == "HeRCULES" else frame.split,
        "sequence": "",
        "lidar_index": frame.frame_id,
        "radar_index": frame.frame_id,
        "timestamp": frame.lidar_path.stem if config.get("dataset") == "HeRCULES" else frame.frame_id,
        "timestamp_ns": int(frame.lidar_path.stem) if config.get("dataset") == "HeRCULES" else int(frame.frame_id),
        "sensor_timestamp_ns": int(frame.lidar_path.stem),
        "source_session": str(frame.radar_path) if config.get("dataset") == "HeRCULES" else frame.split,
        "radar_preprocessing": {
            "alignment": config.get('_hercules_alignment', {}),
            "stack_frames": config.get("hercules_radar_frames"),
            "temporal_radius_m": config.get("hercules_temporal_radius"),
        } if config.get("dataset") == "HeRCULES" else {},
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
        "bev_channel_profile": config["bev_channel_profile"],
        "lidar_input_channels": (
            list(ENGINEERED_LIDAR_CHANNELS)
            if engineered_lidar is not None
            else list(LIDAR_CHANNELS)
        ),
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
        "artifact_profile": config.get("artifact_profile", "training"),
        "artifact_compression_level": config.get("npz_compression_level", 6),
        "remove_added_points": config["remove_added_points"],
        "injection_seed": task["injection_seed"],
        "injection_metadata": injection_metadata,
    }
    sample_arrays = {
        "faulty_lidar_input_bev": engineered_lidar.astype(np.float16)
    } if engineered_lidar is not None else {}
    compact = config.get("artifact_profile", "training") == "training"
    reliability_arrays = (
        training_maps_for_storage(maps)
        if compact
        else canonical_maps_for_storage(maps)
    )
    diagnostic_arrays = {} if compact else {
        "clean_density": clean_layers["raw_density"],
        "faulty_density": faulty_layers["raw_density"],
        "clean_point_ids": clean_ids,
        "faulty_point_ids": faulty_ids,
        "faulty_source_ids": faulty_source_ids,
        "faulty_injector_labels": faulty_labels,
        "observability_ray_count": observability["ray_count"].astype(np.uint32),
        "observability_vertical_coverage": observability[
            "vertical_coverage"
        ].astype(np.float16),
        "observability_ray_support": observability["ray_support"].astype(np.float16),
        "valid_support_mask": np.ones((320, 320), dtype=np.uint8),
    }
    atomic_savez(
        destination,
        compression_level=config.get('npz_compression_level', 6),
        **reliability_arrays,
        clean_rgb=make_rgb_preview(clean_layers),
        faulty_rgb=make_rgb_preview(faulty_layers),
        faulty_lidar_points=faulty[:, :4].astype(np.float32, copy=False),
        observability_confidence=observability[
            "observability_confidence"
        ].astype(np.float16),
        metadata_json=np.asarray(json.dumps(metadata)),
        **diagnostic_arrays,
        **sample_arrays,
    )
    return {"path": str(destination), "cached": False}


def _create_sample(task: dict) -> dict:
    try:
        return _create_sample_impl(task)
    except Exception as error:
        if WORKER_CONFIG is None or not WORKER_CONFIG.get('skip_invalid_synchronization'):
            raise
        from Fault_Localization_Model.hercules_dataset import HerculesSynchronizationError
        if not isinstance(error, HerculesSynchronizationError):
            raise
        frame = task['frame']
        return {'skipped': True, 'cached': False, 'path': '',
                'frame_id': frame['frame_id'], 'split': frame['split'],
                'lidar_path': frame['lidar_path'], 'scene': frame['radar_path'],
                'fault': task.get('fault'), 'severity': task.get('severity'),
                'reason_type': type(error).__name__, 'reason': str(error)}


def _create_radar_cache(task: dict) -> dict:
    if WORKER_CONFIG is None:
        raise RuntimeError("VoD Radar-cache worker was not initialized")
    frame = VODFrame(
        **{
            key: Path(value) if key.endswith("_path") else value
            for key, value in task["frame"].items()
        }
    )
    radar, aligned_radar, lidar_from_radar = _load_frame_radar(frame, WORKER_CONFIG)
    destination = _write_radar_cache(
        frame,
        radar,
        aligned_radar,
        WORKER_CONFIG,
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
    if args.num_workers is None:
        args.num_workers = min(8, os.cpu_count() or 1) if args.hercules_root else 4
    # Spawned interpreters read these before importing NumPy. Explicit user
    # settings win; single-worker BLAS is best controlled in the shell.
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(variable, '1')
    _validate_args(args)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.radar_cache_only:
        started = time.perf_counter()
        LOGGER.info('Loading/compiling observability backend...')
        compiled = warm_observability_backend()
        if not compiled and not args.allow_slow_observability:
            raise RuntimeError('Numba observability backend is unavailable. Install it in this '
                               'Python environment: python -m pip install "numba>=0.60". '
                               'The Python fallback is extremely slow; explicitly opt in with '
                               '--allow-slow-observability only for diagnostics.')
        LOGGER.log(logging.INFO if compiled else logging.WARNING,
                   'Observability backend: %s | startup %.2fs',
                   'Numba compiled exact DDA' if compiled else 'SLOW Python reference',
                   time.perf_counter()-started)
    if args.hercules_root:
        from Fault_Localization_Model.hercules_dataset import discover_hercules_frames, ALIGNMENT_POLICY
        policy = {
            'max_frames': args.hercules_radar_frames or None,
            'max_age_s': args.hercules_max_history_s,
            'max_translation_m': args.hercules_max_translation_m,
            'max_rotation_deg': args.hercules_max_rotation_deg,
        }
        digest = hashlib.sha256(json.dumps([ALIGNMENT_POLICY, policy, args.hercules_doppler_sign,
            args.hercules_temporal_radius, args.hercules_max_radar_age_ms,
            args.hercules_max_pose_gap_ms,
            args.hercules_split_manifest.read_text() if args.hercules_split_manifest else None], sort_keys=True).encode()).hexdigest()[:12]
        frames = discover_hercules_frames(args.hercules_root, args.split,
            radar_variant=f"hercules_v2_{digest}", split_manifest=args.hercules_split_manifest)
    else:
        frames = discover_vod_frames(args.vod_root, args.split, radar_variant=args.radar_variant)
    rng = random.Random(args.seed)
    rng.shuffle(frames)
    count = len(frames) if args.num_samples is None else args.num_samples
    if count > len(frames):
        raise ValueError(
            f"Requested {count} unique samples from only {len(frames)} {args.split} frames"
        )
    frames = frames[:count]
    config = {
        'npz_compression_level': args.npz_compression_level,
        'artifact_profile': args.artifact_profile,
        'skip_invalid_synchronization': args.skip_invalid_synchronization,
        "dataset": "HeRCULES" if args.hercules_root else "View-of-Delft",
        "hercules_radar_frames": args.hercules_radar_frames,
        "hercules_temporal_radius": args.hercules_temporal_radius,
        "hercules_max_radar_age_ms": args.hercules_max_radar_age_ms,
        "hercules_max_pose_gap_ms": args.hercules_max_pose_gap_ms,
        "hercules_split_manifest": str(args.hercules_split_manifest) if args.hercules_split_manifest else None,
        "hercules_stack": policy if args.hercules_root else {},
        "hercules_tracking": {'doppler_sign': args.hercules_doppler_sign},
        "output_root": str(args.output_root) if args.output_root else "",
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
        "bev_channel_profile": args.bev_channel_profile,
    }
    if args.radar_cache_only:
        tasks = [{"frame": _serialize_frame(frame)} for frame in frames]
        worker_initializer = _radar_cache_worker_init
        worker_function = _create_radar_cache
    else:
        plan = build_fault_plan(args.fault_plan, None, None, DEFAULT_FAULT_PLAN)
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
        worker_initializer = _worker_init
        worker_function = _create_sample
    if args.hercules_root and args.hercules_generation_order == 'chronological':
        tasks = _chronological_tasks(tasks)
    LOGGER.info('Scheduling: %s | lossless NPZ compression level: %d',
                args.hercules_generation_order if args.hercules_root else 'random',
                args.npz_compression_level)
    LOGGER.info(
        "%s %d %s frames using %s and %d workers",
        "Caching Radar for" if args.radar_cache_only else "Generating",
        len(tasks),
        args.split,
        'HeRCULES V2 adaptive stack' if args.hercules_root else args.radar_variant,
        args.num_workers,
    )
    created = cached = 0
    skipped = []
    processing_started = last_progress = time.perf_counter()
    if args.num_workers == 1:
        worker_initializer(config)
        results = map(worker_function, tasks)
        executor = None
    else:
        LOGGER.info('Starting %d workers with spawn (no inherited Numba/BLAS state)...', args.num_workers)
        executor = ProcessPoolExecutor(
            max_workers=args.num_workers,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=worker_initializer,
            initargs=(config,),
        )
        results = _bounded_results(executor, worker_function, tasks, args.num_workers * 2)
    try:
        for completed, result in enumerate(results, 1):
            if result.get('skipped'):
                skipped.append(result)
                now = time.perf_counter()
                LOGGER.warning('Skipped unsupported synchronization %s/%s: %s',
                               result['split'], result['frame_id'], result['reason'])
                continue
            cached += int(result["cached"])
            created += int(not result["cached"])
            now = time.perf_counter()
            if completed == 1 or now-last_progress >= 5 or completed == len(tasks):
                LOGGER.info(
                    "Processed %d/%d; created=%d cached=%d | %.2f samples/s | elapsed %.1fs",
                    completed,
                    len(tasks),
                    created,
                    cached,
                    completed / max(now-processing_started, 1e-9),
                    now-processing_started,
                )
                last_progress = now
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)
    if not args.radar_cache_only:
        LOGGER.info("Samples: %s", Path(args.output_root) / args.split)
    if skipped:
        report = Path(args.output_root) / f'skipped_synchronization_{args.split}.json'
        atomic_write_json(report, {'split': args.split, 'count': len(skipped),
                                   'strict_no_extrapolation': True, 'frames': skipped})
        LOGGER.warning('Skipped %d unsupported frames; report: %s', len(skipped), report)
    LOGGER.info('Finished %s: created=%d cached=%d skipped=%d eligible=%d',
                args.split, created, cached, len(skipped), created + cached)
    LOGGER.info("Radar cache: %s", Path(args.radar_cache_root) / args.split)


if __name__ == "__main__":
    main()
