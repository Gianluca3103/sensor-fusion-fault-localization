from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import logging
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from bev_utils import make_rgb_preview, project_lidar_bev, write_image  # noqa: E402
from config_utils import (  # noqa: E402
    config_get,
    load_json_config,
    require_directory,
    require_positive,
    require_range,
    setup_logging,
)
from data_injection_utils import (  # noqa: E402
    DEFAULT_FOG_ROOT,
    DEFAULT_HERCULES_ROOT,
    DEFAULT_INJECTOR_ROOT,
    dilate_mask,
    find_aeva_dir,
    filter_pointcloud,
    list_aeva_bins,
    lisa_label_counts,
    read_hercules_aeva_bin,
)
from fault_injector import build_fault_plan, choose_samples, inject_fault, load_fault_injector  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "grid_reliability_heatmaps"
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
GROUND_TRUTH_METHOD = "point_id_provenance_v1"
VISUALIZATION_METHOD = "point_status_overlay_v1"
POINT_STATUS_CORRECT = 0
POINT_STATUS_MISSING = 1
POINT_STATUS_MOVED = 2
POINT_STATUS_ADDED = 3
WORKER_CONTEXT = None


def config_defaults(config):
    return {
        "data_root": config_get(config, "paths.data_root", str(DEFAULT_HERCULES_ROOT)),
        "injector_root": config_get(config, "paths.injector_root", str(DEFAULT_INJECTOR_ROOT)),
        "fog_root": config_get(config, "paths.fog_root", str(DEFAULT_FOG_ROOT)),
        "output_root": config_get(config, "paths.output_root", str(DEFAULT_OUTPUT_ROOT)),
        "day": config_get(config, "hercules.day", "Day_1_Parking"),
        "session": config_get(config, "hercules.session", "01_Day"),
        "all_scenes": config_get(config, "hercules.all_scenes", False),
        "include_scenes": config_get(config, "hercules.include_scenes", None),
        "exclude_scenes": config_get(config, "hercules.exclude_scenes", None),
        "keep_duplicate_frames": config_get(config, "hercules.keep_duplicate_frames", False),
        "num_samples": config_get(config, "generation.num_samples", 24),
        "seed": config_get(config, "generation.seed", 42),
        "shuffle": config_get(config, "generation.shuffle", True),
        "fault_plan": config_get(config, "faults.plan", None),
        "faults": config_get(config, "faults.names", None),
        "severities": config_get(config, "faults.severities", None),
        "fog_noise": config_get(config, "faults.fog_noise", 10),
        "num_workers": config_get(config, "generation.num_workers", 1),
        "grid_size": config_get(config, "bev.grid_size", 100),
        "x_min": config_get(config, "bev.x_min", 0.0),
        "x_max": config_get(config, "bev.x_max", 64.0),
        "y_min": config_get(config, "bev.y_min", -32.0),
        "y_max": config_get(config, "bev.y_max", 32.0),
        "resolution": config_get(config, "bev.resolution", 0.20),
        "min_range": config_get(config, "bev.min_range", 1.0),
        "max_range": config_get(config, "bev.max_range", 120.0),
        "movement_tolerance_m": config_get(config, "reliability.movement_tolerance_m", 0.05),
    }


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None, help="Optional JSON config file with dataset-generation defaults.")
    pre_args, _ = pre_parser.parse_known_args()
    defaults = config_defaults(load_json_config(pre_args.config))

    parser = argparse.ArgumentParser(
        parents=[pre_parser],
        description="Create Hercules reliability/fault heatmaps by splitting the original BEV view into reliability squares.",
    )
    parser.add_argument("--data-root", default=defaults["data_root"])
    parser.add_argument("--injector-root", default=defaults["injector_root"])
    parser.add_argument("--fog-root", default=defaults["fog_root"])
    parser.add_argument("--output-root", default=defaults["output_root"])
    parser.add_argument("--day", default=defaults["day"])
    parser.add_argument("--session", default=defaults["session"])
    parser.add_argument(
        "--all-scenes",
        action="store_true",
        default=defaults["all_scenes"],
        help="Use every Hercules LiDAR/Aeva folder found under --data-root instead of one --day/--session.",
    )
    parser.add_argument(
        "--include-scenes",
        nargs="*",
        default=defaults["include_scenes"],
        help="Only use these top-level Hercules scene folders, e.g. Bridge01_Day Mountain01_Day.",
    )
    parser.add_argument(
        "--exclude-scenes",
        nargs="*",
        default=defaults["exclude_scenes"],
        help="Exclude these top-level Hercules scene folders.",
    )
    parser.add_argument(
        "--keep-duplicate-frames",
        action="store_true",
        default=defaults["keep_duplicate_frames"],
        help="In --all-scenes mode, keep repeated scene/timestamp frames from duplicated extracted folders.",
    )
    parser.add_argument("--num-samples", type=int, default=defaults["num_samples"])
    parser.add_argument("--frames", type=int, nargs="*", default=None)
    parser.add_argument(
        "--temporal-split",
        choices=["train", "val", "test"],
        default=None,
        help="Use the first train ratio, next validation ratio, or final test ratio from every Aeva folder.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--faults", nargs="*", default=defaults["faults"])
    parser.add_argument("--severities", type=int, nargs="*", default=defaults["severities"])
    parser.add_argument(
        "--fault-plan",
        nargs="*",
        default=defaults["fault_plan"],
        help="Exact mixed-severity plan, e.g. fog_sim:3 rain_sim:5 snow_sim:5 lidar_crosstalk_noise:1 fov_filter:1.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        default=not bool(defaults["shuffle"]),
        help="Keep frame/fault/severity order deterministic.",
    )
    parser.add_argument(
        "--no-previews",
        action="store_true",
        help="Skip the six diagnostic PNGs per sample and save only training data plus manifests.",
    )
    parser.add_argument("--grid-size", type=int, default=defaults["grid_size"])
    parser.add_argument("--x-min", type=float, default=defaults["x_min"])
    parser.add_argument("--x-max", type=float, default=defaults["x_max"])
    parser.add_argument("--y-min", type=float, default=defaults["y_min"])
    parser.add_argument("--y-max", type=float, default=defaults["y_max"])
    parser.add_argument("--resolution", type=float, default=defaults["resolution"])
    parser.add_argument("--min-range", type=float, default=defaults["min_range"])
    parser.add_argument("--max-range", type=float, default=defaults["max_range"])
    parser.add_argument(
        "--movement-tolerance-m",
        type=float,
        default=defaults["movement_tolerance_m"],
        help="Maximum clean-to-faulty displacement treated as unchanged. Defaults to 0.05 m.",
    )
    parser.add_argument("--fog-noise", type=int, default=defaults["fog_noise"])
    parser.add_argument(
        "--num-workers",
        type=int,
        default=defaults["num_workers"],
        help="Number of parallel sample-generation worker processes. Use 1 for the original sequential behavior.",
    )
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def validate_generation_args(args):
    require_directory(args.data_root, "Hercules data root")
    require_directory(args.injector_root, "3D corruptions injector root")
    require_directory(args.fog_root, "LiDAR fog simulator root")
    require_positive(args.num_samples, "num_samples")
    require_positive(args.grid_size, "grid_size")
    require_positive(args.resolution, "resolution")
    require_positive(args.num_workers, "num_workers")
    require_positive(args.movement_tolerance_m, "movement_tolerance_m")
    if args.train_ratio <= 0.0 or args.val_ratio <= 0.0:
        raise ValueError("--train-ratio and --val-ratio must be positive.")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("--train-ratio + --val-ratio must be less than 1.0 so a test split remains.")
    require_range(args.x_min, args.x_max, "x range")
    require_range(args.y_min, args.y_max, "y range")
    require_range(args.min_range, args.max_range, "point range")


def point_counts_grid(points, x_min, x_max, y_min, y_max, grid_rows, grid_cols):
    counts = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    if len(points) == 0:
        return counts

    x_cell_size = (x_max - x_min) / grid_rows
    y_cell_size = (y_max - y_min) / grid_cols
    cols = np.floor((points[:, 1] - y_min) / y_cell_size).astype(np.int32)
    rows_from_bottom = np.floor((points[:, 0] - x_min) / x_cell_size).astype(np.int32)
    rows = grid_rows - 1 - rows_from_bottom
    valid = (rows >= 0) & (rows < grid_rows) & (cols >= 0) & (cols < grid_cols)
    np.add.at(counts, (rows[valid], cols[valid]), 1.0)
    return counts


def make_reliability_maps(
    clean_points,
    clean_point_ids,
    faulty_points,
    faulty_point_ids,
    faulty_source_ids,
    movement_tolerance_m,
    x_min,
    x_max,
    y_min,
    y_max,
    grid_rows,
    grid_cols,
):
    """Build reliability targets from exact point provenance rather than occupancy matching."""
    clean_point_ids = np.asarray(clean_point_ids, dtype=np.int64)
    faulty_point_ids = np.asarray(faulty_point_ids, dtype=np.int64)
    faulty_source_ids = np.asarray(faulty_source_ids, dtype=np.int64)
    if clean_point_ids.shape != (len(clean_points),):
        raise ValueError("clean_point_ids must contain one ID per clean point.")
    if faulty_point_ids.shape != (len(faulty_points),):
        raise ValueError("faulty_point_ids must contain one ID per faulty point.")
    if faulty_source_ids.shape != (len(faulty_points),):
        raise ValueError("faulty_source_ids must contain one source ID per faulty point.")
    if len(np.unique(clean_point_ids)) != len(clean_point_ids):
        raise ValueError("clean_point_ids must be unique within a frame.")
    if len(np.unique(faulty_point_ids)) != len(faulty_point_ids):
        raise ValueError("faulty_point_ids must be unique within a frame.")

    clean_counts = point_counts_grid(clean_points[:, :4], x_min, x_max, y_min, y_max, grid_rows, grid_cols)
    faulty_counts = point_counts_grid(faulty_points[:, :4], x_min, x_max, y_min, y_max, grid_rows, grid_cols)

    clean_index_by_id = {int(point_id): index for index, point_id in enumerate(clean_point_ids)}
    original_mask = faulty_source_ids >= 0
    original_source_ids = faulty_source_ids[original_mask]
    if len(np.unique(original_source_ids)) != len(original_source_ids):
        raise ValueError("An injector produced duplicate rows for the same clean source ID.")
    unknown_ids = sorted(set(map(int, original_source_ids)) - set(clean_index_by_id))
    if unknown_ids:
        raise ValueError(f"Faulty points reference unknown clean source IDs: {unknown_ids[:5]}")

    original_faulty_indices = np.flatnonzero(original_mask)
    original_clean_indices = np.array(
        [clean_index_by_id[int(source_id)] for source_id in original_source_ids],
        dtype=np.int64,
    )
    displacement = np.zeros(len(original_faulty_indices), dtype=np.float32)
    if len(original_faulty_indices):
        coordinate_delta = (
            faulty_points[original_faulty_indices, :3]
            - clean_points[original_clean_indices, :3]
        )
        displacement = np.linalg.norm(coordinate_delta, axis=1)
    moved_original = displacement > movement_tolerance_m

    present_clean = np.zeros(len(clean_points), dtype=bool)
    present_clean[original_clean_indices] = True
    missing_clean = ~present_clean

    correct_clean_indices = original_clean_indices[~moved_original]
    moved_clean_indices = original_clean_indices[moved_original]
    moved_faulty_indices = original_faulty_indices[moved_original]
    added_faulty_indices = np.flatnonzero(~original_mask)

    clean_point_status = np.full(len(clean_points), POINT_STATUS_MISSING, dtype=np.int8)
    clean_point_status[correct_clean_indices] = POINT_STATUS_CORRECT
    clean_point_status[moved_clean_indices] = POINT_STATUS_MOVED
    faulty_point_status = np.full(len(faulty_points), POINT_STATUS_ADDED, dtype=np.int8)
    faulty_point_status[original_faulty_indices[~moved_original]] = POINT_STATUS_CORRECT
    faulty_point_status[moved_faulty_indices] = POINT_STATUS_MOVED

    correct_points = clean_points[correct_clean_indices]
    missing_points = clean_points[missing_clean]
    moved_points = faulty_points[moved_faulty_indices]
    added_points = faulty_points[added_faulty_indices]

    clean_point_counts = point_counts_grid(
        correct_points, x_min, x_max, y_min, y_max, grid_rows, grid_cols
    )
    missing_faulty = point_counts_grid(
        missing_points, x_min, x_max, y_min, y_max, grid_rows, grid_cols
    )
    moved_faulty = point_counts_grid(
        moved_points, x_min, x_max, y_min, y_max, grid_rows, grid_cols
    )
    added_faulty = point_counts_grid(
        added_points, x_min, x_max, y_min, y_max, grid_rows, grid_cols
    )
    faulty_point_counts = missing_faulty + moved_faulty + added_faulty

    denominator = clean_point_counts + faulty_point_counts
    reliability = np.ones_like(clean_counts, dtype=np.float32)
    occupied = denominator > 0
    reliability[occupied] = clean_point_counts[occupied] / denominator[occupied]
    fault_heatmap = 1.0 - np.clip(reliability, 0.0, 1.0)

    return {
        "clean_counts": clean_counts,
        "faulty_counts": faulty_counts,
        "clean_point_counts": clean_point_counts,
        "faulty_point_counts": faulty_point_counts,
        "missing_faulty_counts": missing_faulty,
        "moved_faulty_counts": moved_faulty,
        "added_faulty_counts": added_faulty,
        "correct_counts": clean_point_counts,
        "missing_counts": missing_faulty,
        "wrong_counts": added_faulty,
        "fault_heatmap": fault_heatmap.astype(np.float32),
        "reliability_map": reliability.astype(np.float32),
        "correct_point_ids": clean_point_ids[correct_clean_indices],
        "missing_point_ids": clean_point_ids[missing_clean],
        "moved_point_ids": faulty_point_ids[moved_faulty_indices],
        "moved_source_ids": faulty_source_ids[moved_faulty_indices],
        "moved_displacement_m": displacement[moved_original],
        "added_point_ids": faulty_point_ids[added_faulty_indices],
        "clean_point_status": clean_point_status,
        "faulty_point_status": faulty_point_status,
    }


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


def add_reliability_colorbar(rgb):
    bar_width = 34
    label_width = 104
    pad = 8
    height = rgb.shape[0]
    canvas = np.zeros((height, rgb.shape[1] + bar_width + label_width + pad, 3), dtype=np.uint8)
    canvas[:, : rgb.shape[1]] = rgb
    x0 = rgb.shape[1] + pad

    values = np.linspace(1.0, 0.0, height, dtype=np.float32)
    bar = np.zeros((height, bar_width, 3), dtype=np.uint8)
    bar[..., 0] = np.clip((1.0 - values[:, None]) * 255, 0, 255).astype(np.uint8)
    bar[..., 2] = np.clip(values[:, None] * 255, 0, 255).astype(np.uint8)
    canvas[:, x0 : x0 + bar_width] = bar

    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    text_x = x0 + bar_width + 8
    draw.text((text_x, 8), "Reliable", fill=(80, 160, 255), font=font)
    draw.text((text_x, 26), "1.0", fill=(80, 160, 255), font=font)
    draw.text((text_x, height // 2 - 9), "0.5", fill=(210, 120, 255), font=font)
    draw.text((text_x, height - 42), "0.0", fill=(255, 90, 90), font=font)
    draw.text((text_x, height - 24), "Unreliable", fill=(255, 90, 90), font=font)
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


def normalize_scene_filter(scene_names):
    if not scene_names:
        return None
    return {str(scene).strip().lower() for scene in scene_names if str(scene).strip()}


def list_all_aeva_bins(data_root, dedupe=True, include_scenes=None, exclude_scenes=None):
    include_scenes = normalize_scene_filter(include_scenes)
    exclude_scenes = normalize_scene_filter(exclude_scenes)
    aeva_dirs = []
    for candidate in sorted(data_root.rglob("Aeva")):
        if candidate.is_dir() and list(candidate.glob("*.bin")):
            relative = candidate.relative_to(data_root)
            scene = relative.parts[0].lower() if relative.parts else ""
            if include_scenes is not None and scene not in include_scenes:
                continue
            if exclude_scenes is not None and scene in exclude_scenes:
                continue
            aeva_dirs.append(candidate)
    if not aeva_dirs:
        raise FileNotFoundError(f"No Hercules Aeva folders with .bin files found under {data_root}")

    bins = []
    for aeva_dir in aeva_dirs:
        bins.extend(list_aeva_bins(aeva_dir))
    bins = sorted(bins, key=lambda path: str(path.relative_to(data_root)).lower())
    if dedupe:
        unique_bins = {}
        for bin_path in bins:
            source_meta = hercules_source_metadata(bin_path, data_root)
            key = (source_meta["scene"], bin_path.stem)
            current = unique_bins.get(key)
            if current is None or len(bin_path.parts) < len(current.parts):
                unique_bins[key] = bin_path
        bins = sorted(unique_bins.values(), key=lambda path: str(path.relative_to(data_root)).lower())
    return bins, aeva_dirs


def select_temporal_split_bins(bins, data_root, split_name, train_ratio=0.70, val_ratio=0.15):
    """Select a chronological train/val/test slice inside each Aeva folder."""
    grouped = {}
    for bin_path in bins:
        source_dir = hercules_source_metadata(bin_path, data_root)["source_aeva_dir"]
        grouped.setdefault(source_dir, []).append(bin_path)

    selected = []
    split_counts = []
    for source_dir, folder_bins in sorted(grouped.items()):
        folder_bins = sorted(folder_bins, key=lambda path: path.stem)
        count = len(folder_bins)
        train_end = int(count * train_ratio)
        val_end = int(count * (train_ratio + val_ratio))
        if split_name == "train":
            split_bins = folder_bins[:train_end]
        elif split_name == "val":
            split_bins = folder_bins[train_end:val_end]
        elif split_name == "test":
            split_bins = folder_bins[val_end:]
        else:
            raise ValueError(f"Unknown temporal split: {split_name}")
        selected.extend(split_bins)
        split_counts.append((source_dir, count, len(split_bins)))

    selected = sorted(selected, key=lambda path: str(path.relative_to(data_root)).lower())
    if not selected:
        raise FileNotFoundError(f"No frames selected for temporal split {split_name!r}.")
    return selected, split_counts


def hercules_source_metadata(bin_path, data_root):
    relative = bin_path.relative_to(data_root)
    parts = relative.parts
    scene = parts[0] if parts else ""
    session = ""
    if "LiDAR" in parts:
        lidar_index = parts.index("LiDAR")
        if lidar_index > 1:
            session = parts[lidar_index - 1]
    source_dir = str(bin_path.parent)
    return {
        "scene": scene,
        "day": scene,
        "session": session,
        "source_relative_path": str(relative),
        "source_aeva_dir": source_dir,
    }


def side_by_side(images):
    max_height = max(image.shape[0] for image in images)
    padded = []
    for image in images:
        if image.shape[0] == max_height:
            padded.append(image)
            continue
        canvas = np.zeros((max_height, image.shape[1], 3), dtype=np.uint8)
        canvas[: image.shape[0]] = image
        padded.append(canvas)
    return np.concatenate(padded, axis=1)


def worker_init(context):
    global WORKER_CONTEXT
    WORKER_CONTEXT = dict(context)
    injector_root = Path(WORKER_CONTEXT["injector_root"])
    WORKER_CONTEXT["lidar_corruptions"] = load_fault_injector(injector_root)


def existing_sample_matches(npz_path, cfg, source_meta, timestamp, fault, severity):
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            _ = data["fault_heatmap"].shape
    except Exception as exc:
        LOGGER.warning("Regenerating unreadable existing sample %s: %s", npz_path.name, exc)
        return False

    expected = {
        "dataset": "Hercules",
        "scene": source_meta["scene"],
        "session": source_meta["session"],
        "timestamp": timestamp,
        "fault": fault,
        "severity": severity,
        "grid_size": cfg["grid_size"],
        "image_height": cfg["image_height"],
        "image_width": cfg["image_width"],
        "x_range": [cfg["x_min"], cfg["x_max"]],
        "y_range": [cfg["y_min"], cfg["y_max"]],
        "resolution": cfg["resolution"],
        "ground_truth_method": GROUND_TRUTH_METHOD,
        "visualization_method": VISUALIZATION_METHOD,
        "movement_tolerance_m": cfg["movement_tolerance_m"],
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
            return False
    return True


def create_one_sample(task):
    if WORKER_CONTEXT is None:
        raise RuntimeError("Worker context was not initialized.")

    cfg = WORKER_CONTEXT
    index = task["index"]
    bin_path = Path(task["bin_path"])
    data_root = Path(cfg["data_root"])
    output_root = Path(cfg["output_root"])
    injector_root = Path(cfg["injector_root"])
    fog_root = Path(cfg["fog_root"])
    fault = task["fault"]
    severity = task["severity"]

    timestamp = bin_path.stem
    source_meta = hercules_source_metadata(bin_path, data_root)
    stem = f"{index:04d}_{timestamp}_{fault}_s{severity}"

    fault_png = output_root / f"{stem}_fault_heatmap.png"
    reliability_png = output_root / f"{stem}_reliability_map.png"
    overlay_png = output_root / f"{stem}_fault_overlay.png"
    clean_png = output_root / f"{stem}_clean_bev.png"
    marked_png = output_root / f"{stem}_ideal_bev_changes_marked.png"
    comparison_png = output_root / f"{stem}_comparison.png"
    npz_path = output_root / f"{stem}.npz"

    if npz_path.exists() and existing_sample_matches(npz_path, cfg, source_meta, timestamp, fault, severity):
        return {"index": index, "skipped": True, "npz": str(npz_path)}

    clean_raw = read_hercules_aeva_bin(bin_path)
    clean_points = filter_pointcloud(clean_raw, cfg["min_range"], cfg["max_range"])[:, :4]
    clean_point_ids = np.arange(len(clean_points), dtype=np.int64)
    clean_rgb, clean_layers = clean_bev_rgb(
        clean_points,
        x_range=(cfg["x_min"], cfg["x_max"]),
        y_range=(cfg["y_min"], cfg["y_max"]),
        resolution=cfg["resolution"],
    )
    injection, fog_counts = inject_fault(
        fault,
        clean_points,
        clean_point_ids,
        severity,
        injector_root,
        fog_root,
        cfg["fog_noise"],
        cfg["lidar_corruptions"],
    )
    label_counts = lisa_label_counts(injection.points)
    _, range_mask = filter_pointcloud(
        injection.points,
        cfg["min_range"],
        cfg["max_range"],
        return_mask=True,
    )
    active_mask = range_mask & (injection.injector_labels != 0)
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

    if cfg["save_previews"]:
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
    np.savez_compressed(
        npz_path,
        **maps,
        clean_rgb=clean_rgb,
        clean_density=clean_layers["raw_density"],
        faulty_rgb=faulty_rgb,
        faulty_density=faulty_layers["raw_density"],
        clean_point_ids=clean_point_ids,
        faulty_point_ids=faulty_point_ids,
        faulty_source_ids=faulty_source_ids,
        faulty_injector_labels=faulty_injector_labels,
        metadata_json=json.dumps(
            {
                "dataset": "Hercules",
                "day": source_meta["day"],
                "scene": source_meta["scene"],
                "session": source_meta["session"],
                "source_relative_path": source_meta["source_relative_path"],
                "source_aeva_dir": source_meta["source_aeva_dir"],
                "timestamp": timestamp,
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
                "ground_truth_method": GROUND_TRUTH_METHOD,
                "visualization_method": VISUALIZATION_METHOD,
                "movement_tolerance_m": cfg["movement_tolerance_m"],
                "definition": "reliability=correct/(correct+missing+moved+added), using exact source IDs; weather replacements have new point IDs and no source ID",
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
            },
            indent=2,
        ),
    )

    total_clean = float(np.sum(maps["clean_point_counts"]))
    total_faulty = float(np.sum(maps["faulty_point_counts"]))
    total_missing = float(np.sum(maps["missing_faulty_counts"]))
    total_moved = float(np.sum(maps["moved_faulty_counts"]))
    total_added = float(np.sum(maps["added_faulty_counts"]))
    return {
        "index": index,
        "scene": source_meta["scene"],
        "day": source_meta["day"],
        "session": source_meta["session"],
        "source_relative_path": source_meta["source_relative_path"],
        "source_aeva_dir": source_meta["source_aeva_dir"],
        "timestamp": timestamp,
        "fault": fault,
        "severity": severity,
        "clean_points": len(clean_points),
        "faulty_points": len(faulty_points),
        "total_clean_reliable_points": total_clean,
        "total_faulty_unreliable_points": total_faulty,
        "total_missing_faulty_points": total_missing,
        "total_moved_faulty_points": total_moved,
        "total_added_faulty_points": total_added,
        "global_reliability": total_clean / max(total_clean + total_faulty, 1.0),
        "global_error_ratio": total_faulty / max(total_clean + total_faulty, 1.0),
        "mean_fault_heatmap": float(np.mean(maps["fault_heatmap"])),
        "max_fault_heatmap": float(np.max(maps["fault_heatmap"])),
        "fault_heatmap_png": str(fault_png) if cfg["save_previews"] else "",
        "reliability_png": str(reliability_png) if cfg["save_previews"] else "",
        "overlay_png": str(overlay_png) if cfg["save_previews"] else "",
        "clean_png": str(clean_png) if cfg["save_previews"] else "",
        "marked_png": str(marked_png) if cfg["save_previews"] else "",
        "comparison_png": str(comparison_png) if cfg["save_previews"] else "",
        "npz": str(npz_path),
        **change_counts,
        **label_counts,
        **fog_counts,
    }


def main():
    args = parse_args()
    setup_logging(args.log_level)
    validate_generation_args(args)

    image_height = int(np.ceil((args.x_max - args.x_min) / args.resolution))
    image_width = int(np.ceil((args.y_max - args.y_min) / args.resolution))

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    injector_root = Path(args.injector_root)
    fog_root = Path(args.fog_root)

    if args.all_scenes:
        bins, aeva_dirs = list_all_aeva_bins(
            data_root,
            dedupe=not args.keep_duplicate_frames,
            include_scenes=args.include_scenes,
            exclude_scenes=args.exclude_scenes,
        )
        source_description = f"{len(aeva_dirs)} Aeva folders under {data_root}"
    else:
        aeva_dir = find_aeva_dir(data_root, args.day, args.session)
        bins = list_aeva_bins(aeva_dir)
        source_description = str(aeva_dir)
    if args.frames:
        invalid_frames = [frame for frame in args.frames if frame <= 0 or frame > len(bins)]
        if invalid_frames:
            raise ValueError(f"Requested frame indexes are out of range: {invalid_frames}")
        bins = [bins[frame - 1] for frame in args.frames]
    if args.temporal_split:
        bins, split_counts = select_temporal_split_bins(
            bins,
            data_root,
            args.temporal_split,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        LOGGER.info(
            "Temporal split %s selected %d frames using train=%.3f val=%.3f test=%.3f",
            args.temporal_split,
            len(bins),
            args.train_ratio,
            args.val_ratio,
            1.0 - args.train_ratio - args.val_ratio,
        )
        for source_dir, total_count, split_count in split_counts:
            LOGGER.debug(
                "Temporal split %s: %s -> %d/%d frames",
                args.temporal_split,
                source_dir,
                split_count,
                total_count,
            )
    if not bins:
        raise FileNotFoundError("No Hercules Aeva frames selected.")

    plan = build_fault_plan(args.fault_plan, args.faults, args.severities, FAULT_PLAN)
    LOGGER.info("Selected %d frame candidates from %s", len(bins), source_description)
    LOGGER.info("Fault plan: %s", ", ".join(f"{fault}:S{severity}" for fault, severity in plan))
    samples = choose_samples(bins, args.num_samples, args.seed, plan, shuffle=not args.no_shuffle)
    output_root.mkdir(parents=True, exist_ok=True)

    worker_context = {
        "data_root": str(data_root),
        "output_root": str(output_root),
        "injector_root": str(injector_root),
        "fog_root": str(fog_root),
        "fog_noise": args.fog_noise,
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
    }
    tasks = [
        {"index": index, "bin_path": str(bin_path), "fault": fault, "severity": severity}
        for index, (bin_path, fault, severity) in enumerate(samples)
    ]
    rows = []
    skipped = 0

    if args.num_workers == 1:
        LOGGER.info("Creating samples sequentially")
        worker_init(worker_context)
        for completed, task in enumerate(tasks, start=1):
            result = create_one_sample(task)
            if result.get("skipped"):
                skipped += 1
                LOGGER.info(
                    "Skipping existing %04d/%04d: %s",
                    completed,
                    len(tasks),
                    Path(result["npz"]).name,
                )
            else:
                rows.append(result)
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
            future_to_task = {executor.submit(create_one_sample, task): task for task in tasks}
            completed = 0
            for future in as_completed(future_to_task):
                completed += 1
                result = future.result()
                if result.get("skipped"):
                    skipped += 1
                    LOGGER.info(
                        "Skipping existing %04d/%04d: %s",
                        completed,
                        len(tasks),
                        Path(result["npz"]).name,
                    )
                else:
                    rows.append(result)
                    LOGGER.info(
                        "Created %04d/%04d: %s",
                        completed,
                        len(tasks),
                        Path(result["npz"]).name,
                    )

    rows = sorted(rows, key=lambda row: row["index"])
    if skipped:
        LOGGER.info("Skipped %d existing samples", skipped)

    if rows:
        manifest_path = output_root / "manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    LOGGER.info("Saved grid heatmaps: %s", output_root)


if __name__ == "__main__":
    main()
