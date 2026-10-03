"""Project annotated VoD object returns into the LiDAR range image.

Only clean training labels and clean returns are used for supervision. Neither
boxes nor class masks are model inputs or available to the generator at test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from Fault_Localization_Model.vod_dataset.vod_io import load_vod_lidar_to_camera
from .geometry import RangeGeometry, RangeProjection, angular_indices


CLASS_IDS = {"Car": 1, "Pedestrian": 2, "Cyclist": 3, "bicycle": 3}


def vod_label_paths(metadata: dict) -> tuple[Path, Path]:
    source = Path(str(metadata["source_relative_path"]))
    if source.parent.name != "velodyne":
        raise ValueError(f"Expected VoD LiDAR velodyne source: {source}")
    partition = source.parent.parent
    return partition / "label_2" / f"{source.stem}.txt", partition / "calib" / f"{source.stem}.txt"


def object_class_map(clean_points: np.ndarray, projection: RangeProjection,
                     metadata: dict) -> np.ndarray:
    """Classify only actual clean first returns inside labeled 3D boxes."""
    labels_path, calib_path = vod_label_paths(metadata)
    transform = load_vod_lidar_to_camera(calib_path)
    camera = clean_points[:, :3].astype(np.float64) @ transform[:3, :3].T + transform[:3, 3]
    point_class = np.zeros(len(clean_points), dtype=np.uint8)
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or fields[0] not in CLASS_IDS:
            continue
        if len(fields) < 15:
            raise ValueError(f"Malformed VoD box in {labels_path}: {line!r}")
        height, width, length, x, y, z, yaw = map(float, fields[8:15])
        if not np.isfinite([height, width, length, x, y, z, yaw]).all() or min(height, width, length) <= 0:
            raise ValueError(f"Invalid VoD box in {labels_path}: {line!r}")
        delta = camera - (x, y, z)
        local_x = np.cos(yaw) * delta[:, 0] - np.sin(yaw) * delta[:, 2]
        local_z = np.sin(yaw) * delta[:, 0] + np.cos(yaw) * delta[:, 2]
        # KITTI camera boxes use width along local X and length along local Z.
        inside = ((np.abs(local_x) <= width / 2) & (np.abs(local_z) <= length / 2)
                  & (delta[:, 1] >= -height) & (delta[:, 1] <= 0))
        point_class[inside & (point_class == 0)] = CLASS_IDS[fields[0]]
    result = np.zeros(projection.valid.shape, dtype=np.uint8)
    used = projection.valid
    result[used] = point_class[projection.nearest_original_index[used]]
    return result


def ground_return_mask(clean_points: np.ndarray, projection: RangeProjection,
                       object_class: np.ndarray, *, residual_m: float = 0.25) -> np.ndarray:
    """Estimate the local road plane from clean training returns only.

    The dominant low-height band seeds a robust plane fit. Uncertain frames
    return an empty mask, which avoids suppressing possible objects or walls.
    Annotated object returns are never called ground.
    """
    xyz = np.asarray(clean_points[:, :3], dtype=np.float64)
    result = np.zeros(projection.valid.shape, dtype=np.float32)
    if len(xyz) < 100:
        return result
    radius = np.linalg.norm(xyz, axis=1)
    usable = (np.isfinite(xyz).all(axis=1) & (xyz[:, 0] > 0)
              & (radius > 1) & (radius < 60))
    if usable.sum() < 100:
        return result
    low, high = np.quantile(xyz[usable, 2], [0.02, 0.50])
    if high - low < 0.1:
        return result
    edges = np.arange(low, high + 0.11, 0.1)
    counts, edges = np.histogram(xyz[usable, 2], bins=edges)
    mode_z = (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1]) / 2
    inlier = usable & (np.abs(xyz[:, 2] - mode_z) < 0.3)
    if inlier.sum() < 100:
        return result
    coefficients = None
    for _ in range(3):
        design = np.column_stack((xyz[inlier, 0], xyz[inlier, 1], np.ones(inlier.sum())))
        coefficients = np.linalg.lstsq(design, xyz[inlier, 2], rcond=None)[0]
        estimated_z = xyz[:, 0] * coefficients[0] + xyz[:, 1] * coefficients[1] + coefficients[2]
        inlier = usable & (np.abs(xyz[:, 2] - estimated_z) < residual_m)
        if inlier.sum() < 100:
            return result
    if np.linalg.norm(coefficients[:2]) > 0.2:
        return result
    used = projection.valid
    winners = projection.nearest_original_index[used]
    result[used] = (np.abs(xyz[winners, 2] - estimated_z[winners]) < residual_m)
    result[object_class > 0] = 0
    return result


def projected_box_rectangles(metadata: dict, geometry: RangeGeometry) -> list[tuple[str, int, int, int, int]]:
    """Visualization-only angular envelopes of annotated 3D box corners."""
    labels_path, calib_path = vod_label_paths(metadata)
    camera_from_lidar = load_vod_lidar_to_camera(calib_path)
    lidar_from_camera = np.linalg.inv(camera_from_lidar)
    rectangles = []
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 15 or fields[0] not in CLASS_IDS:
            continue
        height, width, length, x, y, z, yaw = map(float, fields[8:15])
        if min(height, width, length) <= 0 or not np.isfinite([height, width, length, x, y, z, yaw]).all():
            continue
        corners = np.asarray([(sx * width / 2, sy * height, sz * length / 2)
                              for sx in (-1, 1) for sy in (-1, 0) for sz in (-1, 1)])
        camera = np.empty_like(corners)
        camera[:, 0] = x + np.cos(yaw) * corners[:, 0] + np.sin(yaw) * corners[:, 2]
        camera[:, 1] = y + corners[:, 1]
        camera[:, 2] = z - np.sin(yaw) * corners[:, 0] + np.cos(yaw) * corners[:, 2]
        lidar = camera @ lidar_from_camera[:3, :3].T + lidar_from_camera[:3, 3]
        row, col, _, valid = angular_indices(lidar, geometry, require_beam_match=False)
        if valid.sum() < 2 or np.ptp(col[valid]) > geometry.azimuth_bins // 2:
            continue  # Box crosses the range-image seam; a rectangle would mislead.
        rectangles.append((fields[0], int(row[valid].min()), int(col[valid].min()),
                           int(row[valid].max()), int(col[valid].max())))
    return rectangles
