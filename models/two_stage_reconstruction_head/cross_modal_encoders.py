"""Spatially aligned radar and LiDAR encoders for conditional reconstruction.

All inputs use the current LiDAR coordinate frame. The clean LiDAR encoder is a
training teacher; inference has access only to radar and observed faulty LiDAR.
These encoders produce aligned 3D feature grids, not generated points.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class EncoderGrid:
    minimum_xyz: tuple[float, float, float] = (0.0, -40.0, -4.0)
    maximum_xyz: tuple[float, float, float] = (80.0, 40.0, 6.0)
    voxel_size_xyz: tuple[float, float, float] = (2.0, 2.0, 1.0)

    def __post_init__(self) -> None:
        for low, high, step in zip(
            self.minimum_xyz, self.maximum_xyz, self.voxel_size_xyz
        ):
            cells = (high - low) / step
            if step <= 0 or cells <= 0 or abs(cells - round(cells)) > 1e-6:
                raise ValueError("Each grid extent must contain whole positive voxels")

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        x, y, z = (
            round((high - low) / step)
            for low, high, step in zip(
                self.minimum_xyz, self.maximum_xyz, self.voxel_size_xyz
            )
        )
        return z, y, x


@dataclass
class EncodedGrid:
    features: torch.Tensor  # [batch, channels, z, y, x]
    occupied: torch.Tensor  # [batch, 1, z, y, x]
    support: torch.Tensor  # occupied cells and their immediate neighbors
    point_features: torch.Tensor  # [batch, points, channels], fine detail
    point_xyz: torch.Tensor  # [batch, points, 3], current LiDAR frame
    point_valid: torch.Tensor  # [batch, points], inside the shared grid

    def active_tokens(
        self, *, occupied_only: bool = False
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return per-scene (features, ZYX indices) for local attention.

        Ragged lists prevent global attention over the entire dense 3D grid.
        The future diffusion queries can select nearby positions from these.
        """
        mask = self.occupied if occupied_only else self.support
        result = []
        for batch_index in range(self.features.shape[0]):
            positions = torch.nonzero(mask[batch_index, 0], as_tuple=False)
            z, y, x = positions.unbind(dim=1)
            tokens = self.features[batch_index, :, z, y, x].T
            result.append((tokens, positions))
        return result


def _validate_points(
    points: torch.Tensor, valid: torch.Tensor, columns: int
) -> None:
    if points.ndim != 3 or points.shape[-1] != columns:
        raise ValueError(f"Expected [batch, points, {columns}] point tensor")
    if valid.shape != points.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("valid must be a boolean [batch, points] mask")
    if not torch.isfinite(points[valid]).all():
        raise ValueError("Valid points contain NaN or Inf")


def _grid_indices(
    points: torch.Tensor, valid: torch.Tensor, grid: EncoderGrid
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    minimum = points.new_tensor(grid.minimum_xyz)
    size = points.new_tensor(grid.voxel_size_xyz)
    coordinates = (points[..., :3] - minimum) / size
    xyz = torch.floor(coordinates).long()
    z_count, y_count, x_count = grid.shape_zyx
    inside = (
        valid
        & (xyz[..., 0] >= 0) & (xyz[..., 0] < x_count)
        & (xyz[..., 1] >= 0) & (xyz[..., 1] < y_count)
        & (xyz[..., 2] >= 0) & (xyz[..., 2] < z_count)
    )
    # Invalid padded rows never reach the scatter. Clamping avoids a bad
    # intermediate index when padded coordinates contain arbitrary values.
    xyz = torch.stack((
        xyz[..., 0].clamp(0, x_count - 1),
        xyz[..., 1].clamp(0, y_count - 1),
        xyz[..., 2].clamp(0, z_count - 1),
    ), dim=-1)
    batch = torch.arange(points.shape[0], device=points.device)[:, None]
    flat = (((batch * z_count + xyz[..., 2]) * y_count + xyz[..., 1])
            * x_count + xyz[..., 0])
    local_xyz = coordinates - torch.floor(coordinates) - 0.5
    normalized_xyz = coordinates / coordinates.new_tensor((x_count, y_count, z_count))
    return flat, inside, torch.cat((local_xyz, normalized_xyz), dim=-1)


def _scatter_mean_max(
    features: torch.Tensor,
    flat: torch.Tensor,
    keep: torch.Tensor,
    grid: EncoderGrid,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, _, width = features.shape
    z_count, y_count, x_count = grid.shape_zyx
    total = batch * z_count * y_count * x_count
    indices = flat[keep]
    values = features[keep]
    counts = features.new_zeros(total, 1)
    counts.index_add_(0, indices, features.new_ones(len(indices), 1))
    sums = features.new_zeros(total, width)
    sums.index_add_(0, indices, values)
    maximum = features.new_full((total, width), -torch.inf)
    maximum.scatter_reduce_(
        0, indices[:, None].expand(-1, width), values,
        reduce="amax", include_self=True,
    )
    maximum = torch.where(counts > 0, maximum, torch.zeros_like(maximum))
    def as_grid(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(batch, z_count, y_count, x_count, -1).permute(0, 4, 1, 2, 3)
    return as_grid(sums / counts.clamp_min(1)), as_grid(maximum), as_grid(counts)


class _SpatialContext(nn.Module):
    def __init__(self, incoming: int, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv3d(incoming, width, 3, padding=1),
            nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv3d(width, width, 3, padding=1),
            nn.GroupNorm(8, width), nn.SiLU(),
        )

    def forward(
        self, values: torch.Tensor, counts: torch.Tensor,
        point_features: torch.Tensor, point_xyz: torch.Tensor,
        point_valid: torch.Tensor,
    ) -> EncodedGrid:
        occupied = counts > 0
        support = F.max_pool3d(occupied.float(), 3, stride=1, padding=1) > 0
        encoded = self.network(values) * support
        return EncodedGrid(encoded, occupied, support,
                           point_features * point_valid[..., None],
                           point_xyz, point_valid)


class RadarGeometryEncoder(nn.Module):
    """Encode VoD [XYZ, RCS, radial velocity, compensated velocity, age].

    Current and historical returns are pooled separately so moving targets do
    not disappear into a single mean. The model sees time, Doppler and RCS;
    ego-motion aligned XYZ alone is insufficient to distinguish trajectories.
    """

    def __init__(self, grid: EncoderGrid = EncoderGrid(), width: int = 32,
                 history_scans: int = 20) -> None:
        super().__init__()
        if width < 8 or width % 8 or history_scans < 1:
            raise ValueError("width must be a multiple of 8; history_scans must be positive")
        self.grid = grid
        self.history_scans = history_scans
        self.point_mlp = nn.Sequential(
            nn.Linear(12, width), nn.LayerNorm(width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
        )
        self.context = _SpatialContext(4 * width + 2, width)

    def forward(self, points: torch.Tensor, valid: torch.Tensor) -> EncodedGrid:
        _validate_points(points, valid, 7)
        time_indices = points[..., 6][valid]
        if ((time_indices > 0) | (time_indices < 1 - self.history_scans)).any():
            raise ValueError("Radar time indices fall outside the configured history")
        if ((time_indices - time_indices.round()).abs() > 1e-4).any():
            raise ValueError("Radar time indices must identify discrete scans")
        points = torch.where(valid[..., None], points, torch.zeros_like(points))
        flat, inside, spatial = _grid_indices(points, valid, self.grid)
        age = -points[..., 6:7] / max(self.history_scans - 1, 1)
        attributes = torch.cat((
            torch.tanh(points[..., 3:4] / 30.0),
            torch.tanh(points[..., 4:6] / 20.0),
            age,
            torch.sin(age * torch.pi),
            (points[..., 6:7] == 0).to(points.dtype),
        ), dim=-1)
        embedded = self.point_mlp(torch.cat((spatial, attributes), dim=-1))
        current = inside & (points[..., 6] == 0)
        history = inside & (points[..., 6] < 0)
        current_mean, current_max, current_count = _scatter_mean_max(
            embedded, flat, current, self.grid
        )
        history_mean, history_max, history_count = _scatter_mean_max(
            embedded, flat, history, self.grid
        )
        values = torch.cat((
            current_mean, current_max, history_mean, history_max,
            torch.log1p(current_count), torch.log1p(history_count),
        ), dim=1)
        return self.context(values, current_count + history_count,
                            embedded, points[..., :3], inside)


class LidarGeometryEncoder(nn.Module):
    """Encode observed or clean VoD [XYZ, reflectivity] on the same 3D grid."""

    def __init__(self, grid: EncoderGrid = EncoderGrid(), width: int = 32) -> None:
        super().__init__()
        if width < 8 or width % 8:
            raise ValueError("width must be a multiple of 8")
        self.grid = grid
        self.point_mlp = nn.Sequential(
            nn.Linear(7, width), nn.LayerNorm(width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
        )
        self.context = _SpatialContext(2 * width + 1, width)

    def forward(self, points: torch.Tensor, valid: torch.Tensor) -> EncodedGrid:
        _validate_points(points, valid, 4)
        points = torch.where(valid[..., None], points, torch.zeros_like(points))
        flat, inside, spatial = _grid_indices(points, valid, self.grid)
        reflectivity = torch.sign(points[..., 3:4]) * torch.log1p(points[..., 3:4].abs())
        embedded = self.point_mlp(torch.cat((spatial, torch.tanh(reflectivity / 5)), dim=-1))
        mean, maximum, count = _scatter_mean_max(embedded, flat, inside, self.grid)
        values = torch.cat((mean, maximum, torch.log1p(count)), dim=1)
        return self.context(values, count, embedded, points[..., :3], inside)


class RadarLidarEncoders(nn.Module):
    """Train with a clean teacher; infer from radar and faulty LiDAR alone."""

    def __init__(self, grid: EncoderGrid = EncoderGrid(), width: int = 32,
                 history_scans: int = 20) -> None:
        super().__init__()
        self.radar = RadarGeometryEncoder(grid, width, history_scans)
        self.observed_lidar = LidarGeometryEncoder(grid, width)
        self.clean_teacher = LidarGeometryEncoder(grid, width)

    def forward(
        self,
        radar: torch.Tensor,
        radar_valid: torch.Tensor,
        observed_lidar: torch.Tensor,
        observed_valid: torch.Tensor,
        *,
        clean_lidar: torch.Tensor | None = None,
        clean_valid: torch.Tensor | None = None,
    ) -> dict[str, EncodedGrid]:
        if (clean_lidar is None) != (clean_valid is None):
            raise ValueError("Clean LiDAR points and mask must be supplied together")
        if clean_lidar is not None and not self.training:
            raise ValueError("Clean LiDAR is training supervision, not an inference input")
        result = {
            "radar": self.radar(radar, radar_valid),
            "observed_lidar": self.observed_lidar(observed_lidar, observed_valid),
        }
        if clean_lidar is not None:
            result["clean_teacher"] = self.clean_teacher(clean_lidar, clean_valid)
        return result
