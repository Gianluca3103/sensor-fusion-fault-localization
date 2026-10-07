"""Residual Stage-II settings; old radar-only checkpoints remain unchanged."""

from __future__ import annotations

from dataclasses import dataclass

from models.radar_lidar_stage2.config import Stage2Config


@dataclass(frozen=True)
class ResidualStage2Config(Stage2Config):
    ray_tolerance_m: float = 0.10
    observed_weight: float = 2.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.ray_tolerance_m <= 0 or self.observed_weight <= 0:
            raise ValueError("Residual ray tolerance and observed penalty must be positive")
