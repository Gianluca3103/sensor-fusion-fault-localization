"""Quantify aligned radar-BEV information about the clean LiDAR BEV."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as PolygonPath
import numpy as np
from scipy.ndimage import distance_transform_edt

from Fault_Localization_Model.io_utils import atomic_write_json
from Fault_Localization_Model.bev_utils import HEIGHT_RANGE_M
from Fault_Localization_Model.kradar_dataset import (
    load_kradar_annotations,
    resolve_kradar_label_path,
)
from Fault_Localization_Model.sample_utils import load_sample_metadata
from PFS.training_utils import _split_paths
from PFS_Radar.radar_data import radar_cache_path

from .coarse_reconstruction.coarse_config import (
    build_selector_config,
    load_config,
)
from .fault_selector_cache import (
    InvalidSelectorCacheError,
    load_selector_cache,
    selector_cache_path,
)


TOLERANCES_M = (0.2, 0.5, 1.0)
RANGE_BINS_M = (
    ("0_15m", 0.0, 15.0),
    ("15_30m", 15.0, 30.0),
    ("30_45m", 30.0, 45.0),
    ("45_60m", 45.0, 60.0),
    ("over_60m", 60.0, math.inf),
)
POWER_EDGES = np.linspace(0.0, 1.0, 6, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--radar-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/coarse_reconstruction.json"),
        help="Configuration used to validate available Fault Selector caches.",
    )
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=("train", "val"))
    parser.add_argument("--limit-train-samples", type=int)
    parser.add_argument("--limit-val-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--occupancy-threshold", type=float, default=0.5)
    parser.add_argument("--dynamic-threshold", type=float, default=0.0)
    parser.add_argument("--visualize-samples", type=int, default=5)
    parser.add_argument("--scatter-samples", type=int, default=100_000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--kradar-root", type=Path)
    parser.add_argument("--revised-label-root", type=Path)
    return parser.parse_args()


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _distance_to_support(
    support: np.ndarray,
    resolution: tuple[float, float],
) -> np.ndarray:
    if not support.any():
        return np.full(support.shape, np.inf, dtype=np.float32)
    return distance_transform_edt(~support, sampling=resolution).astype(
        np.float32, copy=False
    )


def _radial_grid(metadata: dict, shape: tuple[int, int]) -> tuple[np.ndarray, float, float]:
    height, width = shape
    x_min, x_max = (float(value) for value in metadata["x_range"])
    y_min, y_max = (float(value) for value in metadata["y_range"])
    x_resolution = (x_max - x_min) / height
    y_resolution = (y_max - y_min) / width
    x = x_max - (np.arange(height, dtype=np.float32) + 0.5) * x_resolution
    y = y_min + (np.arange(width, dtype=np.float32) + 0.5) * y_resolution
    return np.hypot(x[:, None], y[None, :]), x_resolution, y_resolution


def _range_masks(distance_m: np.ndarray) -> dict[str, np.ndarray]:
    result = {"all": np.ones(distance_m.shape, dtype=bool)}
    for name, lower, upper in RANGE_BINS_M:
        result[name] = (distance_m >= lower) & (distance_m < upper)
    return result


class CorrespondenceCounts:
    def __init__(self) -> None:
        self.radar_cells = 0
        self.lidar_cells = 0
        self.exact_radar_matches = 0
        self.exact_lidar_matches = 0
        self.radar_tolerant_matches = {value: 0 for value in TOLERANCES_M}
        self.lidar_tolerant_matches = {value: 0 for value in TOLERANCES_M}

    def update(
        self,
        radar: np.ndarray,
        lidar: np.ndarray,
        distance_to_lidar: np.ndarray,
        distance_to_radar: np.ndarray,
        selection: np.ndarray,
    ) -> None:
        radar_selected = radar & selection
        lidar_selected = lidar & selection
        self.radar_cells += int(radar_selected.sum())
        self.lidar_cells += int(lidar_selected.sum())
        overlap = radar & lidar & selection
        exact = int(overlap.sum())
        self.exact_radar_matches += exact
        self.exact_lidar_matches += exact
        for tolerance in TOLERANCES_M:
            self.radar_tolerant_matches[tolerance] += int(
                (radar_selected & (distance_to_lidar <= tolerance + 1e-6)).sum()
            )
            self.lidar_tolerant_matches[tolerance] += int(
                (lidar_selected & (distance_to_radar <= tolerance + 1e-6)).sum()
            )

    def summary(self) -> dict[str, int | float | None]:
        result = {
            "radar_occupied_cells": self.radar_cells,
            "clean_lidar_occupied_cells": self.lidar_cells,
            "p_clean_lidar_occupied_given_radar_occupied": _safe_ratio(
                self.exact_radar_matches, self.radar_cells
            ),
            "p_radar_occupied_given_clean_lidar_occupied": _safe_ratio(
                self.exact_lidar_matches, self.lidar_cells
            ),
        }
        for tolerance in TOLERANCES_M:
            label = str(tolerance).replace(".", "_")
            result[f"radar_to_lidar_correspondence_at_{label}m"] = _safe_ratio(
                self.radar_tolerant_matches[tolerance], self.radar_cells
            )
            result[f"lidar_to_radar_coverage_at_{label}m"] = _safe_ratio(
                self.lidar_tolerant_matches[tolerance], self.lidar_cells
            )
        return result

    def merge(self, other: "CorrespondenceCounts") -> None:
        self.radar_cells += other.radar_cells
        self.lidar_cells += other.lidar_cells
        self.exact_radar_matches += other.exact_radar_matches
        self.exact_lidar_matches += other.exact_lidar_matches
        for tolerance in TOLERANCES_M:
            self.radar_tolerant_matches[tolerance] += other.radar_tolerant_matches[tolerance]
            self.lidar_tolerant_matches[tolerance] += other.lidar_tolerant_matches[tolerance]


class FeatureAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.clean_occupancy_sum = 0.0
        self.clean_density_sum = 0.0
        self.clean_height_sum = 0.0

    def update(self, selected: np.ndarray, clean: np.ndarray) -> None:
        self.count += int(selected.sum())
        self.clean_occupancy_sum += float(clean[0][selected].sum())
        self.clean_density_sum += float(clean[1][selected].sum())
        self.clean_height_sum += float(clean[2][selected].sum())

    def summary(self) -> dict[str, int | float | None]:
        return {
            "cells": self.count,
            "mean_clean_occupancy": _safe_ratio(self.clean_occupancy_sum, self.count),
            "mean_clean_normalized_density": _safe_ratio(self.clean_density_sum, self.count),
            "mean_clean_normalized_height": _safe_ratio(self.clean_height_sum, self.count),
        }

    def merge(self, other: "FeatureAccumulator") -> None:
        self.count += other.count
        self.clean_occupancy_sum += other.clean_occupancy_sum
        self.clean_density_sum += other.clean_density_sum
        self.clean_height_sum += other.clean_height_sum


class HeightAccumulator:
    def __init__(self, scatter_limit: int, seed: int) -> None:
        self.count = 0
        self.absolute_error_sum = 0.0
        self.error_histogram = np.zeros(2001, dtype=np.int64)
        self.sum_x = self.sum_y = self.sum_x2 = self.sum_y2 = self.sum_xy = 0.0
        self.scatter_limit = scatter_limit
        self.scatter_x: list[float] = []
        self.scatter_y: list[float] = []
        self.rng = np.random.default_rng(seed)

    def update(self, radar_height: np.ndarray, lidar_height: np.ndarray, selected: np.ndarray) -> None:
        x = np.asarray(radar_height[selected], dtype=np.float64)
        y = np.asarray(lidar_height[selected], dtype=np.float64)
        if not len(x):
            return
        error = np.abs(x - y)
        self.count += len(x)
        self.absolute_error_sum += float(error.sum())
        bins = np.minimum((error * 2000).astype(np.int64), 2000)
        self.error_histogram += np.bincount(bins, minlength=2001)
        self.sum_x += float(x.sum())
        self.sum_y += float(y.sum())
        self.sum_x2 += float(np.square(x).sum())
        self.sum_y2 += float(np.square(y).sum())
        self.sum_xy += float((x * y).sum())
        remaining = self.scatter_limit - len(self.scatter_x)
        if remaining > 0:
            if len(x) > remaining:
                chosen = self.rng.choice(len(x), remaining, replace=False)
                x, y = x[chosen], y[chosen]
            self.scatter_x.extend(x.tolist())
            self.scatter_y.extend(y.tolist())

    def summary(self) -> dict[str, int | float | None]:
        if not self.count:
            return {
                "cells": 0,
                "mae": None,
                "median_absolute_error": None,
                "mae_m": None,
                "median_absolute_error_m": None,
                "correlation": None,
            }
        median_index = int(np.searchsorted(np.cumsum(self.error_histogram), (self.count + 1) // 2))
        numerator = self.count * self.sum_xy - self.sum_x * self.sum_y
        denominator = math.sqrt(
            max(self.count * self.sum_x2 - self.sum_x**2, 0.0)
            * max(self.count * self.sum_y2 - self.sum_y**2, 0.0)
        )
        mae = self.absolute_error_sum / self.count
        median = median_index / 2000.0
        height_span_m = HEIGHT_RANGE_M[1] - HEIGHT_RANGE_M[0]
        return {
            "cells": self.count,
            "mae": mae,
            "median_absolute_error": median,
            "mae_m": mae * height_span_m,
            "median_absolute_error_m": median * height_span_m,
            "correlation": float(numerator / denominator) if denominator else None,
        }

    def merge(self, other: "HeightAccumulator") -> None:
        self.count += other.count
        self.absolute_error_sum += other.absolute_error_sum
        self.error_histogram += other.error_histogram
        self.sum_x += other.sum_x
        self.sum_y += other.sum_y
        self.sum_x2 += other.sum_x2
        self.sum_y2 += other.sum_y2
        self.sum_xy += other.sum_xy
        remaining = self.scatter_limit - len(self.scatter_x)
        if remaining > 0:
            self.scatter_x.extend(other.scatter_x[:remaining])
            self.scatter_y.extend(other.scatter_y[:remaining])


class PowerAccumulator:
    def __init__(self) -> None:
        shape = (len(POWER_EDGES) - 1,)
        self.cells = np.zeros(shape, dtype=np.int64)
        self.exact = np.zeros(shape, dtype=np.int64)
        self.tolerant = {value: np.zeros(shape, dtype=np.int64) for value in TOLERANCES_M}

    def update(self, power: np.ndarray, selected: np.ndarray, distance_to_lidar: np.ndarray) -> None:
        values = power[selected]
        distances = distance_to_lidar[selected]
        if not len(values):
            return
        indices = np.minimum(np.searchsorted(POWER_EDGES, values, side="right") - 1, len(self.cells) - 1)
        indices = np.maximum(indices, 0)
        self.cells += np.bincount(indices, minlength=len(self.cells))
        self.exact += np.bincount(indices[distances <= 1e-6], minlength=len(self.cells))
        for tolerance in TOLERANCES_M:
            self.tolerant[tolerance] += np.bincount(
                indices[distances <= tolerance + 1e-6], minlength=len(self.cells)
            )

    def summary(self) -> list[dict]:
        rows = []
        for index, (lower, upper) in enumerate(zip(POWER_EDGES, POWER_EDGES[1:])):
            row = {
                "power_bin": f"[{lower:.1f},{upper:.1f}{']' if index == len(self.cells)-1 else ')'}",
                "lower": float(lower),
                "upper": float(upper),
                "radar_cells": int(self.cells[index]),
                "exact_correspondence": _safe_ratio(self.exact[index], self.cells[index]),
            }
            for tolerance in TOLERANCES_M:
                label = str(tolerance).replace(".", "_")
                row[f"correspondence_at_{label}m"] = _safe_ratio(
                    self.tolerant[tolerance][index], self.cells[index]
                )
            rows.append(row)
        return rows

    def merge(self, other: "PowerAccumulator") -> None:
        self.cells += other.cells
        self.exact += other.exact
        for tolerance in TOLERANCES_M:
            self.tolerant[tolerance] += other.tolerant[tolerance]


class AnalysisDomain:
    def __init__(self, scatter_limit: int, seed: int) -> None:
        self.correspondence = {
            name: CorrespondenceCounts() for name in ("all", *(item[0] for item in RANGE_BINS_M))
        }
        self.radar_features = FeatureAccumulator()
        self.height = HeightAccumulator(scatter_limit, seed)
        self.power = PowerAccumulator()
        self.dynamic = CorrespondenceCounts()
        self.dynamic_cells = 0
        self.dynamic_in_object_boxes = 0
        self.dynamic_object_labels_available = 0

    def summary(self) -> dict:
        return {
            "static_occupancy": {
                "overall": self.correspondence["all"].summary(),
                "by_distance": {
                    name: self.correspondence[name].summary()
                    for name, *_ in RANGE_BINS_M
                },
            },
            "clean_lidar_at_static_radar_cells": self.radar_features.summary(),
            "radar_height_vs_clean_lidar_height": self.height.summary(),
            "power_bins": self.power.summary(),
            "dynamic_speed": {
                **self.dynamic.summary(),
                "dynamic_cells": self.dynamic_cells,
                "labels_available_for_cells": self.dynamic_object_labels_available,
                "dynamic_cells_inside_labeled_object_footprints": self.dynamic_in_object_boxes,
                "fraction_inside_labeled_object_footprints": _safe_ratio(
                    self.dynamic_in_object_boxes, self.dynamic_object_labels_available
                ),
                "motion_ground_truth_available": False,
                "interpretation": (
                    "K-Radar detection labels provide object boxes but not object velocity; "
                    "box overlap is an object-region proxy, not confirmed moving-object overlap."
                ),
            },
        }

    def merge(self, other: "AnalysisDomain") -> None:
        for name in self.correspondence:
            self.correspondence[name].merge(other.correspondence[name])
        self.radar_features.merge(other.radar_features)
        self.height.merge(other.height)
        self.power.merge(other.power)
        self.dynamic.merge(other.dynamic)
        self.dynamic_cells += other.dynamic_cells
        self.dynamic_in_object_boxes += other.dynamic_in_object_boxes
        self.dynamic_object_labels_available += other.dynamic_object_labels_available


def _box_mask(metadata: dict, shape: tuple[int, int], kradar_root: Path, revised_root: Path | None) -> np.ndarray:
    label_path = resolve_kradar_label_path(
        kradar_root, metadata, revised_label_root=revised_root
    )
    annotations = load_kradar_annotations(label_path)
    height, width = shape
    distance, x_resolution, y_resolution = _radial_grid(metadata, shape)
    del distance
    x_min, x_max = (float(value) for value in metadata["x_range"])
    y_min, _y_max = (float(value) for value in metadata["y_range"])
    x_centers = x_max - (np.arange(height) + 0.5) * x_resolution
    y_centers = y_min + (np.arange(width) + 0.5) * y_resolution
    output = np.zeros(shape, dtype=bool)
    for annotation in annotations:
        cosine, sine = math.cos(annotation.yaw), math.sin(annotation.yaw)
        local = np.asarray(
            [
                [-annotation.length / 2, -annotation.width / 2],
                [annotation.length / 2, -annotation.width / 2],
                [annotation.length / 2, annotation.width / 2],
                [-annotation.length / 2, annotation.width / 2],
            ]
        )
        rotation = np.asarray([[cosine, -sine], [sine, cosine]])
        corners = local @ rotation.T + np.asarray([annotation.x, annotation.y])
        rows = np.flatnonzero((x_centers >= corners[:, 0].min()) & (x_centers <= corners[:, 0].max()))
        cols = np.flatnonzero((y_centers >= corners[:, 1].min()) & (y_centers <= corners[:, 1].max()))
        if not len(rows) or not len(cols):
            continue
        xx, yy = np.meshgrid(x_centers[rows], y_centers[cols], indexing="ij")
        inside = PolygonPath(corners).contains_points(
            np.column_stack((xx.ravel(), yy.ravel())), radius=1e-9
        ).reshape(len(rows), len(cols))
        output[np.ix_(rows, cols)] |= inside
    return output


def _load_sample(sample_path: Path, radar_root: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(sample_path, allow_pickle=False) as sample:
        clean = np.asarray(sample["clean_rgb"], dtype=np.float32).transpose(2, 0, 1) / 255.0
        metadata = json.loads(str(sample["metadata_json"].item()))
    path = radar_cache_path(radar_root, metadata)
    with np.load(path, allow_pickle=False) as cache:
        radar = np.asarray(cache["radar_bev"], dtype=np.float32)
    if clean.shape[0] != 3 or radar.shape[0] != 4 or clean.shape[1:] != radar.shape[1:]:
        raise ValueError(f"Misaligned BEVs for {sample_path}: clean={clean.shape}, radar={radar.shape}")
    return clean, radar, metadata


def _update_domain(
    accumulator: AnalysisDomain,
    clean: np.ndarray,
    radar: np.ndarray,
    domain: np.ndarray,
    range_masks: dict[str, np.ndarray],
    resolution: tuple[float, float],
    occupancy_threshold: float,
    dynamic_threshold: float,
    object_mask: np.ndarray | None,
) -> None:
    clean_occupancy = clean[0] >= occupancy_threshold
    static_occupancy = radar[0] >= occupancy_threshold
    dynamic = radar[2] > dynamic_threshold
    clean_domain = clean_occupancy & domain
    static_domain = static_occupancy & domain
    dynamic_domain = dynamic & domain
    distance_to_lidar = _distance_to_support(clean_domain, resolution)
    distance_to_static = _distance_to_support(static_domain, resolution)
    distance_to_dynamic = _distance_to_support(dynamic_domain, resolution)
    for name, range_mask in range_masks.items():
        accumulator.correspondence[name].update(
            static_occupancy,
            clean_occupancy,
            distance_to_lidar,
            distance_to_static,
            domain & range_mask,
        )
    accumulator.radar_features.update(static_domain, clean)
    geometric_support = domain & clean_occupancy & (radar[1] > 0)
    accumulator.height.update(radar[3], clean[2], geometric_support)
    accumulator.power.update(radar[1], static_domain, distance_to_lidar)
    accumulator.dynamic.update(
        dynamic,
        clean_occupancy,
        distance_to_lidar,
        distance_to_dynamic,
        domain,
    )
    dynamic_count = int(dynamic_domain.sum())
    accumulator.dynamic_cells += dynamic_count
    if object_mask is not None:
        accumulator.dynamic_object_labels_available += dynamic_count
        accumulator.dynamic_in_object_boxes += int((dynamic_domain & object_mask).sum())


def _save_sample_plots(
    destination: Path,
    clean: np.ndarray,
    radar: np.ndarray,
    metadata: dict,
    resolution: tuple[float, float],
) -> None:
    clean_occupancy = clean[0] >= 0.5
    radar_occupancy = radar[0] >= 0.5
    nearest = _distance_to_support(clean_occupancy, resolution)
    overlay = np.stack(
        (radar_occupancy, clean_occupancy, np.zeros_like(radar_occupancy)), axis=-1
    ).astype(np.float32)
    radar_distance = np.where(radar_occupancy, np.minimum(nearest, 2.0), np.nan)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="black")
    axes[0].imshow(overlay, interpolation="nearest")
    axes[0].set_title("Static radar vs clean LiDAR\nRed: radar | Green: LiDAR | Yellow: exact", color="white")
    image = axes[1].imshow(radar_distance, cmap="turbo", vmin=0.0, vmax=2.0, interpolation="nearest")
    axes[1].set_title("Radar cells: nearest clean-LiDAR distance (m)", color="white")
    figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.03)
    for axis in axes:
        axis.axis("off")
    figure.suptitle(
        f"sequence {metadata.get('sequence')} | radar {metadata.get('radar_index')} | LiDAR {metadata.get('lidar_index')}",
        color="white",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def _plot_summary(split: str, domain_name: str, domain: AnalysisDomain, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summary = domain.summary()
    distance_rows = summary["static_occupancy"]["by_distance"]
    labels = [item[0] for item in RANGE_BINS_M]
    correspondence = [distance_rows[label]["radar_to_lidar_correspondence_at_0_5m"] for label in labels]
    coverage = [distance_rows[label]["lidar_to_radar_coverage_at_0_5m"] for label in labels]
    figure, axis = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(labels))
    axis.plot(positions, correspondence, marker="o", label="Radar→LiDAR correspondence")
    axis.plot(positions, coverage, marker="o", label="LiDAR→radar coverage")
    axis.set_xticks(positions, labels, rotation=20)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Probability at 0.5 m")
    axis.set_title(f"{split}: correspondence vs ego distance ({domain_name})")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_root / f"{split}_{domain_name}_correspondence_vs_distance.png", dpi=160)
    plt.close(figure)

    power = summary["power_bins"]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        np.arange(len(power)),
        [row["correspondence_at_0_5m"] for row in power],
        marker="o",
    )
    axis.set_xticks(np.arange(len(power)), [row["power_bin"] for row in power], rotation=20)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("P(clean LiDAR within 0.5 m)")
    axis.set_xlabel("Normalized radar power")
    axis.set_title(f"{split}: correspondence vs radar power ({domain_name})")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_root / f"{split}_{domain_name}_correspondence_vs_power.png", dpi=160)
    plt.close(figure)

    if domain.height.scatter_x:
        figure, axis = plt.subplots(figsize=(6, 6))
        axis.hexbin(domain.height.scatter_x, domain.height.scatter_y, gridsize=70, mincnt=1, cmap="viridis")
        axis.plot((0, 1), (0, 1), "r--", linewidth=1)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel("Radar normalized upper height")
        axis.set_ylabel("Clean LiDAR normalized upper height")
        axis.set_title(f"{split}: radar vs LiDAR height ({domain_name})")
        figure.tight_layout()
        figure.savefig(output_root / f"{split}_{domain_name}_height_scatter.png", dpi=160)
        plt.close(figure)


def _write_flat_csv(path: Path, summaries: dict[str, dict]) -> None:
    rows = []
    for split, domains in summaries.items():
        for domain, summary in domains.items():
            overall = summary["static_occupancy"]["overall"]
            rows.append({"split": split, "domain": domain, **overall})
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _print_report(split: str, domains: dict[str, dict]) -> None:
    print(f"\n{split.upper()} RADAR INFORMATION REPORT")
    for domain_name, summary in domains.items():
        metrics = summary["static_occupancy"]["overall"]
        height = summary["radar_height_vs_clean_lidar_height"]
        dynamic = summary["dynamic_speed"]
        print(f"  [{domain_name}]")
        print(
            "    static exact correspondence/coverage: "
            f"{(metrics['p_clean_lidar_occupied_given_radar_occupied'] or 0):.2%} / "
            f"{(metrics['p_radar_occupied_given_clean_lidar_occupied'] or 0):.2%}"
        )
        for tolerance in TOLERANCES_M:
            label = str(tolerance).replace(".", "_")
            print(
                f"    {tolerance:.1f}m correspondence/coverage: "
                f"{(metrics[f'radar_to_lidar_correspondence_at_{label}m'] or 0):.2%} / "
                f"{(metrics[f'lidar_to_radar_coverage_at_{label}m'] or 0):.2%}"
            )
        print(
            "    height MAE/median/correlation: "
            f"{height['mae']} / {height['median_absolute_error']} / {height['correlation']}"
        )
        print(
            "    dynamic exact/0.5m correspondence: "
            f"{(dynamic['p_clean_lidar_occupied_given_radar_occupied'] or 0):.2%} / "
            f"{(dynamic['radar_to_lidar_correspondence_at_0_5m'] or 0):.2%}"
        )


def _analyze_chunk(
    paths: list[Path],
    data_root: Path,
    radar_root: Path,
    selector_config,
    occupancy_threshold: float,
    dynamic_threshold: float,
    scatter_limit: int,
    seed: int,
    kradar_root: Path | None,
    revised_label_root: Path | None,
) -> tuple[dict[str, AnalysisDomain], int, int, int, int]:
    domains = {
        "full_bev": AnalysisDomain(scatter_limit, seed),
        "reconstruction_mask": AnalysisDomain(scatter_limit, seed + 1),
    }
    masks_loaded = 0
    masks_missing = 0
    labels_loaded = 0
    labels_unavailable = 0
    for sample_path in paths:
        clean, radar, metadata = _load_sample(sample_path, radar_root)
        shape = clean.shape[1:]
        radial_distance, x_resolution, y_resolution = _radial_grid(metadata, shape)
        ranges = _range_masks(radial_distance)
        object_mask = None
        if kradar_root is not None:
            try:
                object_mask = _box_mask(
                    metadata, shape, kradar_root, revised_label_root
                )
                labels_loaded += 1
            except (FileNotFoundError, ValueError):
                labels_unavailable += 1
        _update_domain(
            domains["full_bev"], clean, radar, np.ones(shape, dtype=bool), ranges,
            (x_resolution, y_resolution), occupancy_threshold,
            dynamic_threshold, object_mask,
        )
        try:
            cache = load_selector_cache(
                selector_cache_path(sample_path, data_root), selector_config
            )
            reconstruction = cache["reconstruction_mask"].astype(bool)
            masks_loaded += 1
            _update_domain(
                domains["reconstruction_mask"], clean, radar, reconstruction,
                ranges, (x_resolution, y_resolution), occupancy_threshold,
                dynamic_threshold, object_mask,
            )
        except InvalidSelectorCacheError:
            masks_missing += 1
    return (
        domains, masks_loaded, masks_missing,
        labels_loaded, labels_unavailable,
    )


def _chunks(paths: list[Path], chunk_size: int) -> list[list[Path]]:
    return [paths[index:index + chunk_size] for index in range(0, len(paths), chunk_size)]


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.occupancy_threshold <= 1.0:
        raise ValueError("--occupancy-threshold must be in [0,1]")
    if args.dynamic_threshold < 0.0 or args.visualize_samples < 0 or args.scatter_samples < 0:
        raise ValueError("dynamic threshold and sample counts must be non-negative")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1")
    args.output_root.mkdir(parents=True, exist_ok=True)
    selector_config = build_selector_config(load_config(args.config))
    summaries = {}
    mask_availability = {}
    object_label_availability = {}
    for split_index, split in enumerate(args.splits):
        limit = args.limit_train_samples if split == "train" else args.limit_val_samples
        paths = _split_paths(args.data_root, split, limit, args.seed)
        domains = {
            "full_bev": AnalysisDomain(args.scatter_samples, args.seed + split_index * 10),
            "reconstruction_mask": AnalysisDomain(args.scatter_samples, args.seed + split_index * 10 + 1),
        }
        masks_loaded = 0
        masks_missing = 0
        labels_loaded = 0
        labels_unavailable = 0
        chunk_size = max(25, math.ceil(len(paths) / (args.num_workers * 12)))
        path_chunks = _chunks(paths, chunk_size)
        chunk_scatter_limit = math.ceil(args.scatter_samples / max(len(path_chunks), 1))

        def analyze(index_and_paths):
            chunk_index, chunk_paths = index_and_paths
            return _analyze_chunk(
                chunk_paths, args.data_root, args.radar_root, selector_config,
                args.occupancy_threshold, args.dynamic_threshold,
                chunk_scatter_limit, args.seed + split_index * 10_000 + chunk_index * 2,
                args.kradar_root, args.revised_label_root,
            )

        completed = 0
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            results = executor.map(analyze, enumerate(path_chunks))
            for chunk_paths, result in zip(path_chunks, results):
                chunk_domains, loaded, missing, label_ok, label_bad = result
                for name in domains:
                    domains[name].merge(chunk_domains[name])
                masks_loaded += loaded
                masks_missing += missing
                labels_loaded += label_ok
                labels_unavailable += label_bad
                completed += len(chunk_paths)
                if completed % 500 < len(chunk_paths) or completed == len(paths):
                    print(f"{split}: analyzed {completed}/{len(paths)}", flush=True)

        for visualized, sample_path in enumerate(paths[:args.visualize_samples]):
            clean, radar, metadata = _load_sample(sample_path, args.radar_root)
            _, x_resolution, y_resolution = _radial_grid(metadata, clean.shape[1:])
            _save_sample_plots(
                args.output_root / "plots" / split / f"{visualized:03d}_{sample_path.stem}.png",
                clean, radar, metadata, (x_resolution, y_resolution),
            )
        split_summary = {name: domain.summary() for name, domain in domains.items()}
        if not masks_loaded:
            split_summary.pop("reconstruction_mask")
        summaries[split] = split_summary
        mask_availability[split] = {
            "loaded": masks_loaded,
            "missing_or_incompatible": masks_missing,
        }
        object_label_availability[split] = {
            "loaded": labels_loaded,
            "missing_or_malformed": labels_unavailable,
            "requested": args.kradar_root is not None,
        }
        for name, domain in domains.items():
            if name in split_summary:
                _plot_summary(split, name, domain, args.output_root / "plots")
        _print_report(split, split_summary)

    report = {
        "definitions": {
            "radar_channels": [
                "static_occupancy", "normalized_power", "dynamic_speed", "robust_upper_height"
            ],
            "clean_lidar_channels": [
                "occupancy", "normalized_log_density", "robust_upper_height"
            ],
            "tolerances_m": TOLERANCES_M,
            "range_bins_m": [
                {
                    "name": name,
                    "lower": lower,
                    "upper": None if math.isinf(upper) else upper,
                }
                for name, lower, upper in RANGE_BINS_M
            ],
            "power_bins": POWER_EDGES.tolist(),
            "height_range_m": list(HEIGHT_RANGE_M),
            "height_error_units": (
                "mae and median_absolute_error are normalized; *_m fields use the "
                "shared LiDAR/radar height range."
            ),
            "masked_analysis": (
                "Both source and target support are restricted to the reconstruction mask."
            ),
        },
        "mask_availability": mask_availability,
        "object_label_availability": object_label_availability,
        "splits": summaries,
    }
    atomic_write_json(args.output_root / "radar_information_report.json", report)
    _write_flat_csv(args.output_root / "static_occupancy_summary.csv", summaries)
    print(f"\nSaved report and plots under: {args.output_root}")


if __name__ == "__main__":
    main()
