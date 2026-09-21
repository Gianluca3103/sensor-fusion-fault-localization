"""Configuration for the standalone 3D voxelization pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VoxelGridConfig:
    x_range: tuple[float, float] = (0.0, 64.0)
    y_range: tuple[float, float] = (-32.0, 32.0)
    z_range: tuple[float, float] = (-3.0, 5.0)
    voxel_size: tuple[float, float, float] = (0.2, 0.2, 0.25)
    coordinate_order: str = "zyx"

    def validate(self) -> None:
        if self.coordinate_order != "zyx":
            raise ValueError("Only sparse coordinate order 'zyx' is supported")
        for name, bounds in (
            ("x_range", self.x_range),
            ("y_range", self.y_range),
            ("z_range", self.z_range),
        ):
            if len(bounds) != 2 or bounds[0] >= bounds[1]:
                raise ValueError(f"{name} must contain increasing [min, max] bounds")
        if len(self.voxel_size) != 3 or any(value <= 0 for value in self.voxel_size):
            raise ValueError("voxel_size must contain three positive values")
        for name, bounds, size in zip(
            ("x", "y", "z"),
            (self.x_range, self.y_range, self.z_range),
            self.voxel_size,
        ):
            cells = (bounds[1] - bounds[0]) / size
            if abs(cells - round(cells)) > 1e-8:
                raise ValueError(
                    f"{name} range must be exactly divisible by its voxel size; got {cells}"
                )

    @property
    def dimensions_xyz(self) -> tuple[int, int, int]:
        self.validate()
        ranges = (self.x_range, self.y_range, self.z_range)
        return tuple(
            int(round((upper - lower) / size))
            for (lower, upper), size in zip(ranges, self.voxel_size)
        )

    @property
    def dimensions_zyx(self) -> tuple[int, int, int]:
        x, y, z = self.dimensions_xyz
        return z, y, x

    @property
    def mins_xyz(self) -> tuple[float, float, float]:
        return self.x_range[0], self.y_range[0], self.z_range[0]

    @property
    def maxs_xyz(self) -> tuple[float, float, float]:
        return self.x_range[1], self.y_range[1], self.z_range[1]


@dataclass(frozen=True)
class ModalityVoxelConfig:
    # None is deliberately unbounded.  Use the statistics tool before choosing
    # a finite value; no arbitrary learned-model constraint is imposed here.
    max_points_per_voxel: int | None = None

    def validate(self) -> None:
        if self.max_points_per_voxel is not None and self.max_points_per_voxel < 1:
            raise ValueError("max_points_per_voxel must be positive or null")


@dataclass(frozen=True)
class VoxelizationConfig:
    grid: VoxelGridConfig = VoxelGridConfig()
    lidar: ModalityVoxelConfig = ModalityVoxelConfig()
    radar: ModalityVoxelConfig = ModalityVoxelConfig()
    cache_version: int = 1
    compression_level: int = 6

    def validate(self) -> None:
        self.grid.validate()
        self.lidar.validate()
        self.radar.validate()
        if self.cache_version < 1:
            raise ValueError("cache_version must be positive")
        if not 0 <= self.compression_level <= 9:
            raise ValueError("compression_level must be between 0 and 9")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def _pair(payload: dict[str, Any], name: str, default: tuple[float, float]):
    values = payload.get(name, default)
    return float(values[0]), float(values[1])


def load_voxelization_config(path: str | Path) -> VoxelizationConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload = payload.get("voxelization_3d", payload)
    grid_payload = payload.get("grid", {})
    grid = VoxelGridConfig(
        x_range=_pair(grid_payload, "x_range", (0.0, 64.0)),
        y_range=_pair(grid_payload, "y_range", (-32.0, 32.0)),
        z_range=_pair(grid_payload, "z_range", (-3.0, 5.0)),
        voxel_size=tuple(
            float(value)
            for value in grid_payload.get("voxel_size", (0.2, 0.2, 0.25))
        ),
        coordinate_order=str(grid_payload.get("coordinate_order", "zyx")),
    )
    config = VoxelizationConfig(
        grid=grid,
        lidar=ModalityVoxelConfig(
            payload.get("lidar", {}).get("max_points_per_voxel")
        ),
        radar=ModalityVoxelConfig(
            payload.get("radar", {}).get("max_points_per_voxel")
        ),
        cache_version=int(payload.get("cache_version", 1)),
        compression_level=int(payload.get("compression_level", 6)),
    )
    config.validate()
    return config
