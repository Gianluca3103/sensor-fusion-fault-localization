"""Sensor-native and XYZ debugging views for conservative reconstruction."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import RangeSample
from .geometry import RangeGeometry, project_lidar
from .merge import MergeResult


def save_range_comparison(path: Path, sample: RangeSample, merged: MergeResult,
                          add_probability: np.ndarray, add_range: np.ndarray,
                          delete_probability: np.ndarray, geometry: RangeGeometry) -> None:
    final_projection = project_lidar(merged.points, geometry)
    figure, axes = plt.subplots(3, 3, figsize=(16, 11), constrained_layout=True)
    deleted_map = np.zeros(geometry.shape, dtype=np.float32)
    if len(sample.faulty_points) and len(merged.deleted_original_indices):
        is_deleted = np.zeros(len(sample.faulty_points), dtype=bool)
        is_deleted[merged.deleted_original_indices] = True
        nearest = sample.faulty_projection.nearest_original_index
        deleted_map[sample.faulty_projection.valid] = is_deleted[nearest[sample.faulty_projection.valid]]
    maps = (
        (sample.faulty_projection.range_m, "Faulty LiDAR range"),
        (sample.clean_projection.range_m, "Clean LiDAR range"),
        (sample.radar_features[1], "Radar log count"),
        (add_probability, "Predicted ADD probability"),
        (np.where(add_probability >= 0.5, add_range, 0), "Proposed ADD range"),
        (delete_probability, "Predicted DELETE probability"),
        (final_projection.range_m, "Final reconstructed range"),
        (deleted_map, "Deleted originals"),
        (np.bincount(merged.generated_rows * geometry.azimuth_bins + merged.generated_cols,
                     minlength=np.prod(geometry.shape)).reshape(geometry.shape), "Accepted generated returns"),
    )
    for axis, (array, title) in zip(axes.flat, maps):
        image = axis.imshow(array, aspect="auto", interpolation="nearest")
        axis.set_title(title)
        axis.set_xlabel("azimuth bin")
        axis.set_ylabel("beam")
        figure.colorbar(image, ax=axis, fraction=0.04)
    figure.savefig(path, dpi=140)
    plt.close(figure)

    figure = plt.figure(figsize=(15, 9), constrained_layout=True)
    views = (
        (sample.faulty_points, "Faulty original", "#be3434"),
        (sample.clean_points, "Clean target", "#2e9757"),
        (merged.generated_points, "Generated additions", "#167bbb"),
        (sample.faulty_points[merged.deleted_original_indices], "Deleted originals", "#ee9911"),
        (merged.points, "Final reconstructed", "#8844aa"),
    )
    for index, (points, title, color) in enumerate(views, start=1):
        axis = figure.add_subplot(2, 3, index, projection="3d")
        if len(points):
            selected = points[np.linspace(0, len(points) - 1, min(len(points), 10000), dtype=np.int64)]
            axis.scatter(selected[:, 0], selected[:, 1], selected[:, 2], s=0.3, c=color)
        axis.set_title(title)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_zlabel("z (m)")
    figure.savefig(path.with_name(path.stem + "_xyz.png"), dpi=140)
    plt.close(figure)
