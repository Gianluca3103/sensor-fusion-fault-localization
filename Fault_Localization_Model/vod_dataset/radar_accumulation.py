"""Ego-motion compensated accumulation of View-of-Delft radar scans."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from Fault_Localization_Model.vod_dataset.vod_io import (
    VOD_RADAR_FIELDS,
    _named_transform,
    load_vod_radar,
)


def load_vod_odom_from_camera(path: str | Path) -> np.ndarray:
    """Load VoD's camera-to-odometry pose from one per-frame JSON file."""

    path = Path(path)
    values = None
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if "odomToCamera" in payload:
            values = payload["odomToCamera"]
            break
    if values is None:
        raise ValueError(f"odomToCamera was not found in {path}")
    transform = np.asarray(values, dtype=np.float64)
    if transform.size != 16 or not np.isfinite(transform).all():
        raise ValueError(f"Malformed odomToCamera in {path}")
    transform = transform.reshape(4, 4)
    if abs(np.linalg.det(transform[:3, :3])) < 1e-10:
        raise ValueError(f"Singular odomToCamera rotation in {path}")
    return transform


def radar_current_from_source(
    source_pose_path: str | Path,
    current_pose_path: str | Path,
    source_calibration_path: str | Path,
    current_calibration_path: str | Path,
) -> np.ndarray:
    """Return the rigid transform from a source radar scan to current radar."""

    odom_from_source_camera = load_vod_odom_from_camera(source_pose_path)
    odom_from_current_camera = load_vod_odom_from_camera(current_pose_path)
    source_camera_from_radar = _named_transform(source_calibration_path)
    current_camera_from_radar = _named_transform(current_calibration_path)
    return (
        np.linalg.inv(current_camera_from_radar)
        @ np.linalg.inv(odom_from_current_camera)
        @ odom_from_source_camera
        @ source_camera_from_radar
    )


def transform_radar_scan(
    radar_points: np.ndarray,
    current_from_source: np.ndarray,
    time_index: int,
) -> np.ndarray:
    """Transform radar XYZ and label the scan while retaining measured fields."""

    radar_points = np.asarray(radar_points, dtype=np.float32)
    if radar_points.ndim != 2 or radar_points.shape[1] != len(VOD_RADAR_FIELDS):
        raise ValueError(
            f"radar_points must have shape [N,{len(VOD_RADAR_FIELDS)}], got "
            f"{radar_points.shape}"
        )
    transform = np.asarray(current_from_source, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("current_from_source must be a finite 4x4 transform")
    homogeneous = np.column_stack(
        (radar_points[:, :3], np.ones(len(radar_points), dtype=np.float64))
    )
    output = radar_points.copy()
    output[:, :3] = (homogeneous @ transform.T)[:, :3].astype(np.float32)
    output[:, 6] = float(time_index)
    return output


def accumulate_vod_radar_scans(
    source_paths: list[str | Path],
    pose_paths: list[str | Path],
    calibration_paths: list[str | Path],
) -> np.ndarray:
    """Align chronological source scans into the final scan's radar frame."""

    if not source_paths:
        raise ValueError("At least one source radar scan is required")
    if not (len(source_paths) == len(pose_paths) == len(calibration_paths)):
        raise ValueError("Radar, pose, and calibration path counts must match")

    current_pose = pose_paths[-1]
    current_calibration = calibration_paths[-1]
    count = len(source_paths)
    scans = []
    for index, (radar_path, pose_path, calibration_path) in enumerate(
        zip(source_paths, pose_paths, calibration_paths)
    ):
        time_index = index - count + 1
        points = load_vod_radar(radar_path)
        if time_index == 0:
            transform = np.eye(4, dtype=np.float64)
        else:
            transform = radar_current_from_source(
                pose_path,
                current_pose,
                calibration_path,
                current_calibration,
            )
        scans.append(transform_radar_scan(points, transform, time_index))
    return np.concatenate(scans, axis=0)
