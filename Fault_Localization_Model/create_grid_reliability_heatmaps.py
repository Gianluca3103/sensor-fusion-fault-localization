from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
import logging
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from Fault_Localization_Model.bev_utils import (  # noqa: E402
    HEIGHT_RANGE_M,
    LIDAR_CHANNELS,
    UPPER_HEIGHT_QUANTILE,
    make_rgb_preview,
    project_lidar_bev,
    write_image,
)
from Fault_Localization_Model.kradar_dataset import (  # noqa: E402
    K_RADAR_AZIMUTH_RANGE_RAD,
    K_RADAR_ELEVATION_RANGE_RAD,
    K_RADAR_RANGE_M,
    kradar_source_metadata,
    list_all_kradar_lidar_frames,
    load_radar_from_lidar_transform,
    radar_overlap_mask,
    read_kradar_lidar_pcd,
    select_temporal_split_frames,
)
from Fault_Localization_Model.config.cli import parse_args  # noqa: E402
from Fault_Localization_Model.config.defaults import DEFAULT_FOG_ROOT, DEFAULT_INJECTOR_ROOT  # noqa: E402
from Fault_Localization_Model.config.validation import validate_generation_args  # noqa: E402
from Fault_Localization_Model.config_utils import setup_logging  # noqa: E402
from Fault_Localization_Model.concurrency_utils import iter_bounded_futures  # noqa: E402
from Fault_Localization_Model.data_injection_utils import (  # noqa: E402
    DEFAULT_FOG_SIMULATOR_NOISE,
    DEFAULT_WEATHER_THREADS,
    dilate_mask,
    filter_pointcloud,
)
from Fault_Localization_Model.fault_injector import build_fault_plan, choose_samples, inject_fault, load_fault_injector  # noqa: E402
from Fault_Localization_Model.io_utils import atomic_savez_compressed, write_csv_rows  # noqa: E402
from Fault_Localization_Model.reliability_maps import (  # noqa: E402
    POINT_STATUS_ADDED,
    POINT_STATUS_MISSING,
    POINT_STATUS_MOVED,
    canonical_maps_for_storage,
    make_reliability_maps,
    point_counts_grid,
)
from Fault_Localization_Model.visualization_utils import add_reliability_colorbar, side_by_side  # noqa: E402
LOGGER = logging.getLogger("create_grid_reliability_heatmaps")
FAULT_PLAN = [
    ("rain_sim", 5),
    ("snow_sim", 5),
    ("fog_sim", 5),
    ("fov_filter", 1),
    ("lidar_crosstalk_noise", 1),
    ("gaussian_noise", 1),
    ("uniform_noise", 1),
    ("impulse_noise", 1),
]
GROUND_TRUTH_METHOD = "point_id_provenance_v2_literature_fov"
VISUALIZATION_METHOD = "point_status_overlay_v1"
GENERATOR_VERSION = 8
RESUME_REQUIRED_ARRAYS = (
    "fault_heatmap",
    "reliability_map",
    "clean_rgb",
    "faulty_rgb",
    "clean_point_ids",
    "faulty_point_ids",
    "faulty_source_ids",
    "faulty_injector_labels",
    "clean_point_counts",
    "faulty_point_counts",
    "missing_faulty_counts",
    "moved_faulty_counts",
    "added_faulty_counts",
    "correct_point_ids",
    "missing_point_ids",
    "moved_point_ids",
    "added_point_ids",
)
WORKER_CONTEXT = None


def colorize_fault_heatmap(values):
    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
    clipped = np.clip(values, 0.0, 1.0)
    rgb[..., 0] = np.clip(clipped * 255, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(np.maximum(0.0, 1.0 - np.abs(clipped - 0.5) * 2.0) * 200, 0, 200).astype(np.uint8)
    rgb[..., 2] = np.clip((1.0 - clipped) * 80, 0, 80).astype(np.uint8)
    return rgb


def colorize_reliability(values):
    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
    clipped = np.clip(values, 0.0, 1.0)
    rgb[..., 0] = np.clip((1.0 - clipped) * 255, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(clipped * 255, 0, 255).astype(np.uint8)
    return rgb


def add_legend_above(rgb, title, details):
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    lines = [title, *details]
    pad = len(lines) * 18 + 10
    canvas = np.zeros((rgb.shape[0] + pad, rgb.shape[1], 3), dtype=np.uint8)
    canvas[:pad] = np.array([18, 18, 18], dtype=np.uint8)
    canvas[pad:] = rgb
    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    y = 6
    for line in lines:
        draw.text((8, y), line, fill=(255, 255, 255), font=font)
        y += 18
    return np.array(image)


def overlay_heatmap_on_counts(counts, heatmap):
    base = np.zeros((*counts.shape, 3), dtype=np.uint8)
    if np.max(counts) > 0:
        density = np.log1p(counts) / np.max(np.log1p(counts))
        base[..., 2] = np.clip(density * 180, 0, 180).astype(np.uint8)
    heat_rgb = colorize_fault_heatmap(heatmap).astype(np.float32)
    alpha = np.clip(heatmap[..., None] * 0.85, 0.0, 0.85)
    return np.clip(base.astype(np.float32) * (1.0 - alpha) + heat_rgb * alpha, 0, 255).astype(np.uint8)


def mark_bev_point_statuses(
    clean_points,
    faulty_points,
    clean_point_status,
    faulty_point_status,
    faulty_rgb,
    x_min,
    x_max,
    y_min,
    y_max,
):
    """Mark only missing, moved, and added evidence used by the reliability target."""
    image_rows, image_cols = faulty_rgb.shape[:2]
    missing_points = clean_points[clean_point_status == POINT_STATUS_MISSING]
    moved_points = faulty_points[faulty_point_status == POINT_STATUS_MOVED]
    added_points = faulty_points[faulty_point_status == POINT_STATUS_ADDED]

    missing_mask = point_counts_grid(
        missing_points, x_min, x_max, y_min, y_max, image_rows, image_cols
    ) > 0
    moved_mask = point_counts_grid(
        moved_points, x_min, x_max, y_min, y_max, image_rows, image_cols
    ) > 0
    added_mask = point_counts_grid(
        added_points, x_min, x_max, y_min, y_max, image_rows, image_cols
    ) > 0
    marked_mask = missing_mask | moved_mask | added_mask

    overlay = faulty_rgb.copy()
    overlay[dilate_mask(marked_mask, 2)] = np.array([0, 0, 0], dtype=np.uint8)
    overlay[dilate_mask(missing_mask, 1)] = np.array([255, 80, 0], dtype=np.uint8)
    overlay[dilate_mask(added_mask, 1)] = np.array([255, 255, 0], dtype=np.uint8)
    overlay[dilate_mask(moved_mask, 1)] = np.array([0, 255, 255], dtype=np.uint8)

    return overlay, {
        "marked_status_cells": int(marked_mask.sum()),
        "missing_point_cells": int(missing_mask.sum()),
        "moved_point_cells": int(moved_mask.sum()),
        "added_point_cells": int(added_mask.sum()),
        "missing_points_marked": int(len(missing_points)),
        "moved_points_marked": int(len(moved_points)),
        "added_points_marked": int(len(added_points)),
    }


def resize_nearest(rgb, height, width):
    return np.array(Image.fromarray(rgb, mode="RGB").resize((width, height), Image.Resampling.NEAREST))


def clean_bev_rgb(points, x_range, y_range, resolution):
    layers = project_lidar_bev(points[:, :4], x_range=x_range, y_range=y_range, resolution=resolution)
    return make_rgb_preview(layers), layers


def worker_init(context):
    global WORKER_CONTEXT
    WORKER_CONTEXT = dict(context)
    _load_clean_artifacts.cache_clear()
    WORKER_CONTEXT["lidar_corruptions"] = load_fault_injector(DEFAULT_INJECTOR_ROOT)


@lru_cache(maxsize=2)
def _load_clean_artifacts(lidar_path_text):
    """Load and project one immutable clean source frame per worker."""

    if WORKER_CONTEXT is None:
        raise RuntimeError("Worker context was not initialized.")
    cfg = WORKER_CONTEXT
    lidar_path = Path(lidar_path_text)
    source_meta = kradar_source_metadata(lidar_path, cfg["data_root"])
    radar_from_lidar = load_radar_from_lidar_transform(
        source_meta["calibration_path"]
    )
    clean_raw = read_kradar_lidar_pcd(lidar_path)
    _, range_mask = filter_pointcloud(
        clean_raw,
        cfg["min_range"],
        cfg["max_range"],
        return_mask=True,
    )
    overlap_mask = radar_overlap_mask(clean_raw, radar_from_lidar)
    clean_points = clean_raw[range_mask & overlap_mask, :4]
    if len(clean_points) == 0:
        raise ValueError(
            "No LiDAR points remain after range and K-Radar FOV filtering "
            f"{lidar_path}"
        )
    clean_point_ids = np.arange(len(clean_points), dtype=np.int64)
    clean_rgb, clean_layers = clean_bev_rgb(
        clean_points,
        x_range=(cfg["x_min"], cfg["x_max"]),
        y_range=(cfg["y_min"], cfg["y_max"]),
        resolution=cfg["resolution"],
    )
    return (
        clean_points,
        clean_point_ids,
        clean_rgb,
        clean_layers["raw_density"],
        radar_from_lidar,
    )


def chronological_source_batches(tasks, batch_size):
    """Reorder execution only, keeping duplicate source frames in one batch."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    groups = {}
    for task in tasks:
        groups.setdefault(task["lidar_path"], []).append(task)

    batches = []
    current = []
    for lidar_path in sorted(groups, key=lambda value: value.lower()):
        group = sorted(groups[lidar_path], key=lambda task: task["index"])
        if current and len(current) + len(group) > batch_size:
            batches.append(current)
            current = []
        current.extend(group)
    if current:
        batches.append(current)
    return batches


def create_sample_batch(tasks):
    """Create a chronological batch inside one persistent worker process."""

    return [create_one_sample(task) for task in tasks]


def load_matching_existing_sample(
    npz_path,
    cfg,
    source_meta,
    timestamp,
    fault,
    severity,
    injection_seed,
):
    """Return validated metadata/arrays for a resumable sample, or None."""
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            missing = set(RESUME_REQUIRED_ARRAYS) - set(data.files)
            if missing:
                raise KeyError(
                    "missing arrays: " + ", ".join(sorted(missing))
                )
            metadata = json.loads(str(data["metadata_json"]))
            arrays = {
                key: np.asarray(data[key])
                for key in RESUME_REQUIRED_ARRAYS
            }
            if arrays["fault_heatmap"].shape != (
                cfg["grid_size"],
                cfg["grid_size"],
            ):
                raise ValueError(
                    f"fault_heatmap has shape {arrays['fault_heatmap'].shape}"
                )
            for map_name in (
                "reliability_map",
                "clean_point_counts",
                "faulty_point_counts",
                "missing_faulty_counts",
                "moved_faulty_counts",
                "added_faulty_counts",
            ):
                if arrays[map_name].shape != arrays["fault_heatmap"].shape:
                    raise ValueError(
                        f"{map_name} has shape {arrays[map_name].shape}"
                    )
                if not np.isfinite(arrays[map_name]).all():
                    raise ValueError(f"{map_name} contains non-finite values")
            if (
                not np.isfinite(arrays["fault_heatmap"]).all()
                or float(arrays["fault_heatmap"].min()) < 0.0
                or float(arrays["fault_heatmap"].max()) > 1.0
            ):
                raise ValueError("fault_heatmap must contain finite values in [0,1]")
            if (
                float(arrays["reliability_map"].min()) < 0.0
                or float(arrays["reliability_map"].max()) > 1.0
            ):
                raise ValueError("reliability_map must contain values in [0,1]")
            if arrays["clean_rgb"].shape != (
                cfg["image_height"],
                cfg["image_width"],
                3,
            ):
                raise ValueError(
                    f"clean_rgb has shape {arrays['clean_rgb'].shape}"
                )
            if arrays["faulty_rgb"].shape != arrays["clean_rgb"].shape:
                raise ValueError(
                    "faulty_rgb and clean_rgb shapes do not match"
                )
            if len(arrays["clean_point_ids"]) == 0:
                raise ValueError("clean_point_ids is empty")
            if (
                len(arrays["faulty_point_ids"])
                != len(arrays["faulty_source_ids"])
            ):
                raise ValueError(
                    "faulty point IDs and source IDs have different lengths"
                )
            if (
                len(arrays["faulty_point_ids"])
                != len(arrays["faulty_injector_labels"])
            ):
                raise ValueError(
                    "faulty point IDs and injector labels have different lengths"
                )
        if cfg["save_previews"]:
            preview_suffixes = (
                "_fault_heatmap.png",
                "_reliability_map.png",
                "_fault_overlay.png",
                "_clean_bev.png",
                "_ideal_bev_changes_marked.png",
                "_comparison.png",
            )
            for suffix in preview_suffixes:
                preview_path = npz_path.with_name(f"{npz_path.stem}{suffix}")
                with Image.open(preview_path) as preview:
                    preview.verify()
    except Exception as exc:
        LOGGER.warning("Regenerating unreadable existing sample %s: %s", npz_path.name, exc)
        return None

    expected = {
        "dataset": "K-Radar",
        "scene": source_meta["scene"],
        "session": source_meta["session"],
        "source_relative_path": source_meta["source_relative_path"],
        "sequence": source_meta["sequence"],
        "lidar_index": source_meta["lidar_index"],
        "radar_index": source_meta["radar_index"],
        "timestamp": timestamp,
        "fault": fault,
        "severity": severity,
        "grid_size": cfg["grid_size"],
        "image_height": cfg["image_height"],
        "image_width": cfg["image_width"],
        "x_range": [cfg["x_min"], cfg["x_max"]],
        "y_range": [cfg["y_min"], cfg["y_max"]],
        "resolution": cfg["resolution"],
        "min_range": cfg["min_range"],
        "max_range": cfg["max_range"],
        "fog_noise": DEFAULT_FOG_SIMULATOR_NOISE,
        "ground_truth_method": GROUND_TRUTH_METHOD,
        "visualization_method": VISUALIZATION_METHOD,
        "movement_tolerance_m": cfg["movement_tolerance_m"],
        "generator_version": GENERATOR_VERSION,
        "generation_seed": cfg["generation_seed"],
        "injection_seed": injection_seed,
        "weather_threads": DEFAULT_WEATHER_THREADS,
        "radar_azimuth_range_rad": list(K_RADAR_AZIMUTH_RANGE_RAD),
        "radar_elevation_range_rad": list(K_RADAR_ELEVATION_RANGE_RAD),
        "radar_range_m": list(K_RADAR_RANGE_M),
        "lidar_channels": list(LIDAR_CHANNELS),
        "lidar_upper_height_quantile": UPPER_HEIGHT_QUANTILE,
        "lidar_height_range_m": list(HEIGHT_RANGE_M),
    }
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            LOGGER.warning(
                "Regenerating %s because metadata %s is %r, expected %r",
                npz_path.name,
                key,
                metadata.get(key),
                expected_value,
            )
            return None
    return {"metadata": metadata, "arrays": arrays}


def build_manifest_row(
    index,
    npz_path,
    cfg,
    metadata,
    arrays,
    *,
    reused_existing,
):
    """Build one stable manifest row from saved sample contents."""
    total_clean = float(np.sum(arrays["clean_point_counts"]))
    total_faulty = float(np.sum(arrays["faulty_point_counts"]))
    total_missing = float(np.sum(arrays["missing_faulty_counts"]))
    total_moved = float(np.sum(arrays["moved_faulty_counts"]))
    total_added = float(np.sum(arrays["added_faulty_counts"]))
    denominator = max(total_clean + total_faulty, 1.0)
    labels = np.asarray(arrays["faulty_injector_labels"], dtype=np.int8)
    injection_metadata = metadata.get("injection_metadata") or {}
    if not isinstance(injection_metadata, dict):
        injection_metadata = {"injection_metadata": str(injection_metadata)}

    stem = Path(npz_path).stem
    output_root = Path(cfg["output_root"])
    preview_path = lambda suffix: (
        str(output_root / f"{stem}_{suffix}.png")
        if cfg["save_previews"]
        else ""
    )
    return {
        "index": int(index),
        "reused_existing": bool(reused_existing),
        "scene": metadata.get("scene", ""),
        "day": metadata.get("day", ""),
        "session": metadata.get("session", ""),
        "source_relative_path": metadata.get("source_relative_path", ""),
        "source_lidar_dir": metadata.get("source_lidar_dir", ""),
        "sequence": metadata.get("sequence", ""),
        "lidar_index": metadata.get("lidar_index", ""),
        "radar_index": metadata.get("radar_index", ""),
        "timestamp": metadata.get("timestamp", ""),
        "fault": metadata.get("fault", ""),
        "severity": metadata.get("severity", ""),
        "generator_version": metadata.get("generator_version", ""),
        "generation_seed": metadata.get("generation_seed", ""),
        "injection_seed": metadata.get("injection_seed", ""),
        "weather_threads": metadata.get("weather_threads", ""),
        "clean_points": int(len(arrays["clean_point_ids"])),
        "faulty_points": int(len(arrays["faulty_point_ids"])),
        "correct_points": int(len(arrays["correct_point_ids"])),
        "missing_points": int(len(arrays["missing_point_ids"])),
        "moved_points": int(len(arrays["moved_point_ids"])),
        "added_points": int(len(arrays["added_point_ids"])),
        "correct_grid_cells": int(np.count_nonzero(arrays["clean_point_counts"])),
        "missing_grid_cells": int(np.count_nonzero(arrays["missing_faulty_counts"])),
        "moved_grid_cells": int(np.count_nonzero(arrays["moved_faulty_counts"])),
        "added_grid_cells": int(np.count_nonzero(arrays["added_faulty_counts"])),
        "total_clean_reliable_points": total_clean,
        "total_faulty_unreliable_points": total_faulty,
        "total_missing_faulty_points": total_missing,
        "total_moved_faulty_points": total_moved,
        "total_added_faulty_points": total_added,
        "global_reliability": total_clean / denominator,
        "global_error_ratio": total_faulty / denominator,
        "mean_fault_heatmap": float(np.mean(arrays["fault_heatmap"])),
        "max_fault_heatmap": float(np.max(arrays["fault_heatmap"])),
        "lisa_label0_lost_points": int(np.sum(labels == 0)),
        "lisa_label1_non_scattered_points": int(np.sum(labels == 1)),
        "lisa_label2_scattered_points": int(np.sum(labels == 2)),
        "fault_heatmap_png": preview_path("fault_heatmap"),
        "reliability_png": preview_path("reliability_map"),
        "overlay_png": preview_path("fault_overlay"),
        "clean_png": preview_path("clean_bev"),
        "marked_png": preview_path("ideal_bev_changes_marked"),
        "comparison_png": preview_path("comparison"),
        "npz": str(npz_path),
        **injection_metadata,
    }


def create_one_sample(task):
    if WORKER_CONTEXT is None:
        raise RuntimeError("Worker context was not initialized.")

    cfg = WORKER_CONTEXT
    index = task["index"]
    lidar_path = Path(task["lidar_path"])
    data_root = Path(cfg["data_root"])
    output_root = Path(cfg["output_root"])
    fault = task["fault"]
    severity = task["severity"]
    injection_seed = task["injection_seed"]

    source_meta = kradar_source_metadata(lidar_path, data_root)
    timestamp = source_meta["timestamp"]
    stem = f"{index:04d}_{timestamp}_{fault}_s{severity}"

    fault_png = output_root / f"{stem}_fault_heatmap.png"
    reliability_png = output_root / f"{stem}_reliability_map.png"
    overlay_png = output_root / f"{stem}_fault_overlay.png"
    clean_png = output_root / f"{stem}_clean_bev.png"
    marked_png = output_root / f"{stem}_ideal_bev_changes_marked.png"
    comparison_png = output_root / f"{stem}_comparison.png"
    npz_path = output_root / f"{stem}.npz"

    if npz_path.exists():
        existing = load_matching_existing_sample(
            npz_path,
            cfg,
            source_meta,
            timestamp,
            fault,
            severity,
            injection_seed,
        )
        if existing is not None:
            return {
                **build_manifest_row(
                    index,
                    npz_path,
                    cfg,
                    existing["metadata"],
                    existing["arrays"],
                    reused_existing=True,
                ),
                "skipped": True,
            }

    (
        cached_clean_points,
        cached_clean_point_ids,
        clean_rgb,
        clean_density,
        radar_from_lidar,
    ) = _load_clean_artifacts(str(lidar_path))
    # Fault injectors receive per-sample copies so cached clean artifacts cannot
    # leak mutations between different fault realizations of the same frame.
    clean_points = cached_clean_points.copy()
    clean_point_ids = cached_clean_point_ids.copy()
    injection, fog_counts = inject_fault(
        fault,
        clean_points,
        clean_point_ids,
        severity,
        DEFAULT_INJECTOR_ROOT,
        DEFAULT_FOG_ROOT,
        lidar_corruptions=cfg["lidar_corruptions"],
        rng_seed=injection_seed,
    )
    _, range_mask = filter_pointcloud(
        injection.points,
        cfg["min_range"],
        cfg["max_range"],
        return_mask=True,
    )
    overlap_mask = radar_overlap_mask(injection.points, radar_from_lidar)
    active_mask = (
        range_mask
        & overlap_mask
        & (injection.injector_labels != 0)
    )
    faulty_points = injection.points[active_mask]
    faulty_point_ids = injection.point_ids[active_mask]
    faulty_source_ids = injection.source_ids[active_mask]
    faulty_injector_labels = injection.injector_labels[active_mask]
    faulty_rgb, faulty_layers = clean_bev_rgb(
        faulty_points[:, :4],
        x_range=(cfg["x_min"], cfg["x_max"]),
        y_range=(cfg["y_min"], cfg["y_max"]),
        resolution=cfg["resolution"],
    )

    maps = make_reliability_maps(
        clean_points,
        clean_point_ids,
        faulty_points,
        faulty_point_ids,
        faulty_source_ids,
        movement_tolerance_m=cfg["movement_tolerance_m"],
        x_min=cfg["x_min"],
        x_max=cfg["x_max"],
        y_min=cfg["y_min"],
        y_max=cfg["y_max"],
        grid_rows=cfg["grid_size"],
        grid_cols=cfg["grid_size"],
    )
    if cfg["save_previews"]:
        marked_rgb, change_counts = mark_bev_point_statuses(
            clean_points,
            faulty_points,
            maps["clean_point_status"],
            maps["faulty_point_status"],
            faulty_rgb,
            cfg["x_min"],
            cfg["x_max"],
            cfg["y_min"],
            cfg["y_max"],
        )
        fault_rgb = resize_nearest(
            colorize_fault_heatmap(maps["fault_heatmap"]),
            cfg["image_height"],
            cfg["image_width"],
        )
        reliability_rgb = resize_nearest(
            colorize_reliability(maps["reliability_map"]),
            cfg["image_height"],
            cfg["image_width"],
        )
        overlay_rgb = resize_nearest(
            overlay_heatmap_on_counts(maps["clean_counts"], maps["fault_heatmap"]),
            cfg["image_height"],
            cfg["image_width"],
        )
        clean_labeled = add_legend_above(
            clean_rgb,
            "ORIGINAL CLEAN BEV",
            [f"x={cfg['x_min']:g}..{cfg['x_max']:g}m, y={cfg['y_min']:g}..{cfg['y_max']:g}m"],
        )
        fault_rgb = add_legend_above(
            fault_rgb,
            "FAULT HEATMAP: 0=ok, 1=max fault",
            [
                f"{cfg['image_width']}x{cfg['image_height']} BEV split into {cfg['grid_size']}x{cfg['grid_size']} squares",
                "reliability=correct/(correct+missing+moved+added)",
                (
                    f"IDs: correct={len(maps['correct_point_ids'])}, missing={len(maps['missing_point_ids'])}, "
                    f"moved>{cfg['movement_tolerance_m']:g}m={len(maps['moved_point_ids'])}, "
                    f"added={len(maps['added_point_ids'])}"
                ),
            ],
        )
        reliability_rgb = add_legend_above(
            reliability_rgb,
            "IDEAL RELIABILITY MAP",
            [
                "blue=reliable, red=unreliable",
                f"{cfg['image_width']}x{cfg['image_height']} BEV split into {cfg['grid_size']}x{cfg['grid_size']} squares",
            ],
        )
        reliability_rgb = add_reliability_colorbar(reliability_rgb)
        overlay_rgb = add_legend_above(
            overlay_rgb,
            "FAULT HEATMAP OVER CLEAN DENSITY",
            [f"{fault} severity {severity}"],
        )
        marked_rgb = add_legend_above(
            marked_rgb,
            "IDEAL ID-BASED POINT STATUS",
            [
                (
                    f"ORANGE=missing {change_counts['missing_points_marked']} pts, "
                    f"YELLOW=added {change_counts['added_points_marked']} pts"
                ),
                (
                    f"CYAN=moved>{cfg['movement_tolerance_m']:g}m "
                    f"{change_counts['moved_points_marked']} pts"
                ),
            ],
        )
        comparison_rgb = side_by_side([clean_labeled, marked_rgb, reliability_rgb])

        write_image(clean_png, clean_labeled)
        write_image(fault_png, fault_rgb)
        write_image(reliability_png, reliability_rgb)
        write_image(overlay_png, overlay_rgb)
        write_image(marked_png, marked_rgb)
        write_image(comparison_png, comparison_rgb)
    sample_metadata = {
        "dataset": "K-Radar",
        "day": source_meta["day"],
        "scene": source_meta["scene"],
        "session": source_meta["session"],
        "sequence": source_meta["sequence"],
        "lidar_index": source_meta["lidar_index"],
        "radar_index": source_meta["radar_index"],
        "source_relative_path": source_meta["source_relative_path"],
        "source_lidar_dir": source_meta["source_lidar_dir"],
        "label_relative_path": source_meta["label_relative_path"],
        "radar_relative_path": source_meta["radar_relative_path"],
        "timestamp": timestamp,
        "timestamp_ns": source_meta["timestamp_ns"],
        "fault": fault,
        "severity": severity,
        "grid_size": cfg["grid_size"],
        "image_height": cfg["image_height"],
        "image_width": cfg["image_width"],
        "x_cell_size_m": (cfg["x_max"] - cfg["x_min"]) / cfg["grid_size"],
        "y_cell_size_m": (cfg["y_max"] - cfg["y_min"]) / cfg["grid_size"],
        "x_range": [cfg["x_min"], cfg["x_max"]],
        "y_range": [cfg["y_min"], cfg["y_max"]],
        "resolution": cfg["resolution"],
        "min_range": cfg["min_range"],
        "max_range": cfg["max_range"],
        "lidar_sensor": "os2-64",
        "radar_azimuth_range_rad": list(K_RADAR_AZIMUTH_RANGE_RAD),
        "radar_elevation_range_rad": list(K_RADAR_ELEVATION_RANGE_RAD),
        "radar_range_m": list(K_RADAR_RANGE_M),
        "lidar_channels": list(LIDAR_CHANNELS),
        "lidar_upper_height_quantile": UPPER_HEIGHT_QUANTILE,
        "lidar_height_range_m": list(HEIGHT_RANGE_M),
        "radar_from_lidar": radar_from_lidar.tolist(),
        "spatial_support": "intersection of LiDAR returns and K-Radar polar FOV",
        "fog_noise": fog_counts.get("fog_noise", ""),
        "ground_truth_method": GROUND_TRUTH_METHOD,
        "visualization_method": VISUALIZATION_METHOD,
        "movement_tolerance_m": cfg["movement_tolerance_m"],
        "generator_version": GENERATOR_VERSION,
        "generation_seed": cfg["generation_seed"],
        "injection_seed": injection_seed,
        "weather_threads": fog_counts.get("weather_threads", ""),
        "injection_metadata": fog_counts,
        "definition": (
            "reliability=correct/(correct+missing+moved+added), using exact "
            "source IDs; weather replacements have new point IDs and no source ID"
        ),
        "classification_counts": {
            "correct": len(maps["correct_point_ids"]),
            "missing": len(maps["missing_point_ids"]),
            "moved": len(maps["moved_point_ids"]),
            "added": len(maps["added_point_ids"]),
        },
        "point_status_labels": {
            "0": "correct",
            "1": "missing",
            "2": "moved",
            "3": "added",
        },
    }
    atomic_savez_compressed(
        npz_path,
        **canonical_maps_for_storage(maps),
        clean_rgb=clean_rgb,
        clean_density=clean_density,
        faulty_rgb=faulty_rgb,
        faulty_density=faulty_layers["raw_density"],
        clean_point_ids=clean_point_ids,
        faulty_point_ids=faulty_point_ids,
        faulty_source_ids=faulty_source_ids,
        faulty_injector_labels=faulty_injector_labels,
        metadata_json=json.dumps(sample_metadata, indent=2),
    )
    return {
        **build_manifest_row(
            index,
            npz_path,
            cfg,
            sample_metadata,
            {
                **maps,
                "clean_point_ids": clean_point_ids,
                "faulty_point_ids": faulty_point_ids,
                "faulty_source_ids": faulty_source_ids,
                "faulty_injector_labels": faulty_injector_labels,
            },
            reused_existing=False,
        ),
        "skipped": False,
    }


def main():
    #Read and Validate Settings
    args = parse_args() #Collects and Convert Settings
    setup_logging(args.log_level) #Configures terminal messages
    validate_generation_args(args) #Checks if settings are usable

    #Calculate BEV Dimensions
    image_height = int(np.ceil((args.x_max - args.x_min) / args.resolution))
    image_width = int(np.ceil((args.y_max - args.y_min) / args.resolution))

    #Define Important Folders
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    # Find every locally available os2-64 frame with an exact pc10p pairing.
    lidar_frames, sequence_dirs = list_all_kradar_lidar_frames(data_root)
    source_description = (
        f"{len(sequence_dirs)} K-Radar sequences under {data_root}"
    )
    #Split into Train/Val/Test given the time in which the frame is located
    if args.temporal_split:
        lidar_frames, split_counts = select_temporal_split_frames(
            lidar_frames,
            data_root,
            args.temporal_split,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        LOGGER.info(
            "Temporal split %s selected %d frames using train=%.3f val=%.3f test=%.3f",
            args.temporal_split,
            len(lidar_frames),
            args.train_ratio,
            args.val_ratio,
            1.0 - args.train_ratio - args.val_ratio,
        )
        for sequence, total_count, split_count in split_counts:
            LOGGER.debug(
                "Temporal split %s: %s -> %d/%d frames",
                args.temporal_split,
                sequence,
                split_count,
                total_count,
            )
    if not lidar_frames:
        raise FileNotFoundError("No paired K-Radar LiDAR frames selected.")
    #Choose Faults and Samples
    plan = build_fault_plan(args.fault_plan, args.faults, args.severities, FAULT_PLAN)
    LOGGER.info(
        "Selected %d paired frame candidates from %s",
        len(lidar_frames),
        source_description,
    )
    LOGGER.info("Fault plan: %s", ", ".join(f"{fault}:S{severity}" for fault, severity in plan))
    samples = choose_samples(
        lidar_frames,
        args.num_samples,
        args.seed,
        plan,
        shuffle=not args.no_shuffle,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    #Store Settings by sample processing workers
    worker_context = {
        "data_root": str(data_root),
        "output_root": str(output_root),
        "grid_size": args.grid_size,
        "x_min": args.x_min,
        "x_max": args.x_max,
        "y_min": args.y_min,
        "y_max": args.y_max,
        "resolution": args.resolution,
        "min_range": args.min_range,
        "max_range": args.max_range,
        "movement_tolerance_m": args.movement_tolerance_m,
        "save_previews": not args.no_previews,
        "image_height": image_height,
        "image_width": image_width,
        "generation_seed": args.seed,
    }
    #Create task description per generated sample
    tasks = [
        {
            "index": index,
            "lidar_path": str(lidar_path),
            "fault": fault,
            "severity": severity,
            "injection_seed": int(
                np.random.SeedSequence([args.seed, index]).generate_state(
                    1, dtype=np.uint32
                )[0]
            ),
        }
        for index, (lidar_path, fault, severity) in enumerate(samples)
    ]
    #Group Tasks into batches
    task_batches = chronological_source_batches(
        tasks,
        args.source_batch_size,
    )
    LOGGER.info(
        "Execution reordered into %d chronological source batches; "
        "sample membership, indexes, seeds, and filenames are unchanged",
        len(task_batches),
    )
    rows = []
    skipped = 0
    #Process Each sample
    if args.num_workers == 1:
        LOGGER.info("Creating samples sequentially")
        worker_init(worker_context)
        completed = 0
        for batch in task_batches:
            for result in create_sample_batch(batch):
                completed += 1
                rows.append(result)
                if result.get("skipped"):
                    skipped += 1
                    LOGGER.info(
                        "Skipping existing %04d/%04d: %s",
                        completed,
                        len(tasks),
                        Path(result["npz"]).name,
                    )
                else:
                    LOGGER.info(
                        "Created %04d/%04d: %s",
                        completed,
                        len(tasks),
                        Path(result["npz"]).name,
                    )
    else:
        LOGGER.info("Creating samples with %d worker processes", args.num_workers)
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=worker_init,
            initargs=(worker_context,),
        ) as executor:
            completed = 0
            for future, _ in iter_bounded_futures(
                executor,
                create_sample_batch,
                task_batches,
                max_pending=max(args.num_workers * 3, 1),
            ):
                for result in future.result():
                    completed += 1
                    rows.append(result)
                    if result.get("skipped"):
                        skipped += 1
                        LOGGER.info(
                            "Skipping existing %04d/%04d: %s",
                            completed,
                            len(tasks),
                            Path(result["npz"]).name,
                        )
                    else:
                        LOGGER.info(
                            "Created %04d/%04d: %s",
                            completed,
                            len(tasks),
                            Path(result["npz"]).name,
                        )

    rows = sorted(rows, key=lambda row: row["index"])
    if skipped:
        LOGGER.info("Skipped %d existing samples", skipped)
    #Save a document describing generated samples
    write_csv_rows(
        output_root / "manifest.csv",
        rows,
        fieldnames=list(rows[0]) if rows else None,
    )
    LOGGER.info("Saved grid heatmaps: %s", output_root)


if __name__ == "__main__":
    main()
