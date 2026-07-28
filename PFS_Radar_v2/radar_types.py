from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np


@dataclass(frozen=True)
class AdaptiveStackConfig:
    """Pose gates and soft weights for causal radar-frame accumulation."""

    max_frames: int | None = None
    max_age_s: float = 1.0
    max_translation_m: float = 4.0
    max_rotation_deg: float = 5.0
    weight_time_s: float = 0.5
    weight_translation_m: float = 2.0
    weight_rotation_deg: float = 3.0

    def validate(self) -> None:
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be None or at least 1")
        positive = {
            "max_age_s": self.max_age_s,
            "max_translation_m": self.max_translation_m,
            "max_rotation_deg": self.max_rotation_deg,
            "weight_time_s": self.weight_time_s,
            "weight_translation_m": self.weight_translation_m,
            "weight_rotation_deg": self.weight_rotation_deg,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class DopplerTrackingConfig:
    """Ego-Doppler compensation, clustering, and short-window tracking."""

    dynamic_threshold_mps: float = 1.0
    doppler_sign: str = "auto"
    sign_inference_min_speed_mps: float = 0.5
    cluster_eps_m: float = 1.2
    cluster_min_samples: int = 2
    association_distance_m: float = 3.0
    min_track_hits: int = 2
    velocity_smoothing: float = 0.5
    max_abs_velocity_mps: float = 30.0

    def validate(self) -> None:
        if self.doppler_sign not in {"auto", "1", "-1"}:
            raise ValueError("doppler_sign must be one of: auto, 1, -1")
        positive = {
            "dynamic_threshold_mps": self.dynamic_threshold_mps,
            "sign_inference_min_speed_mps": self.sign_inference_min_speed_mps,
            "cluster_eps_m": self.cluster_eps_m,
            "association_distance_m": self.association_distance_m,
            "max_abs_velocity_mps": self.max_abs_velocity_mps,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.cluster_min_samples < 1:
            raise ValueError("cluster_min_samples must be at least 1")
        if self.min_track_hits < 1:
            raise ValueError("min_track_hits must be at least 1")
        if not 0.0 <= self.velocity_smoothing < 1.0:
            raise ValueError("velocity_smoothing must lie in [0, 1)")


@dataclass
class SelectedFrame:
    path: Path
    timestamp: int
    pose: np.ndarray
    age_s: float
    translation_m: float
    rotation_deg: float
    weight: float
    pose_delta_ms: float


@dataclass
class ProcessedFrame:
    timestamp: int
    points: np.ndarray
    doppler_residual_mps: np.ndarray
    dynamic_mask: np.ndarray
    cluster_labels: np.ndarray
    weight: float
    doppler_sign: int
    sensor_speed_mps: float
    yaw_rate_dps: float


@dataclass
class ClusterObservation:
    frame_index: int
    timestamp: int
    label: int
    point_indices: np.ndarray
    centroid: np.ndarray
    doppler_velocity: np.ndarray
    track_id: int = -1


@dataclass
class TrackState:
    track_id: int
    position: np.ndarray
    velocity: np.ndarray
    last_timestamp: int
    hits: int
