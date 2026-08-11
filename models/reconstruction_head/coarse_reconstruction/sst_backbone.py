"""Single-stride sparse regional Transformer for coarse BEV reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from ..encoders import _group_count
from ..pointpillars import PointPillarsOutput


@dataclass(frozen=True)
class SSTConfig:
    token_dim: int = 128
    num_blocks: int = 6
    num_heads: int = 8
    mlp_hidden_dim: int = 256
    region_size_cells: int = 12
    shift_size_cells: int = 6
    dropout: float = 0.0
    include_repair_tokens: bool = False

    def validate(self) -> None:
        for name in (
            "token_dim",
            "num_blocks",
            "num_heads",
            "mlp_hidden_dim",
            "region_size_cells",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"sst.{name} must be positive")
        if self.token_dim % self.num_heads:
            raise ValueError("sst.token_dim must be divisible by sst.num_heads")
        if not 0 <= self.shift_size_cells < self.region_size_cells:
            raise ValueError(
                "sst.shift_size_cells must be in [0, region_size_cells)"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("sst.dropout must be in [0,1)")
        if not isinstance(self.include_repair_tokens, bool):
            raise ValueError("sst.include_repair_tokens must be boolean")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SparseMultimodalTokens:
    features: torch.Tensor
    coordinates: torch.Tensor
    trusted_lidar_coordinates: torch.Tensor
    statistics: dict[str, torch.Tensor]


def regional_group_indices(
    coordinates: torch.Tensor,
    region_size_cells: int,
    *,
    shift_size_cells: int = 0,
) -> torch.Tensor:
    """Return `[batch, region_row, region_col]` without moving coordinates."""

    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape [N,3] as batch,row,col")
    if region_size_cells < 1:
        raise ValueError("region_size_cells must be positive")
    if not 0 <= shift_size_cells < region_size_cells:
        raise ValueError("shift_size_cells must be smaller than the region size")
    groups = coordinates.clone()
    groups[:, 1:] = torch.div(
        groups[:, 1:] + shift_size_cells,
        region_size_cells,
        rounding_mode="floor",
    )
    return groups


class SparseTokenBuilder(nn.Module):
    """Fuse independent PFN outputs at the union of active XY pillars."""

    def __init__(self, lidar_channels: int, radar_channels: int, token_dim: int):
        super().__init__()
        self.input_channels = int(lidar_channels + radar_channels + 2)
        self.projection = nn.Linear(self.input_channels, token_dim)

    @staticmethod
    def _flat_keys(
        coordinates: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        return (
            coordinates[:, 0] * height * width
            + coordinates[:, 1] * width
            + coordinates[:, 2]
        )

    @staticmethod
    def _coordinates_from_keys(
        keys: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        batch = torch.div(keys, height * width, rounding_mode="floor")
        spatial = keys.remainder(height * width)
        row = torch.div(spatial, width, rounding_mode="floor")
        col = spatial.remainder(width)
        return torch.stack((batch, row, col), dim=1)

    def forward(
        self,
        lidar: PointPillarsOutput,
        radar: PointPillarsOutput,
        reconstruction_mask: torch.Tensor,
        healthy_context_mask: torch.Tensor,
        *,
        include_repair_tokens: bool,
        radar_enabled: bool,
    ) -> SparseMultimodalTokens:
        batch, _, height, width = lidar.dense_features.shape
        expected_mask_shape = (batch, 1, height, width)
        if reconstruction_mask.shape != expected_mask_shape:
            raise ValueError(
                f"reconstruction_mask must have shape {expected_mask_shape}"
            )
        if healthy_context_mask.shape != expected_mask_shape:
            raise ValueError(
                f"healthy_context_mask must have shape {expected_mask_shape}"
            )
        lidar_coordinates = lidar.sparse_coordinates
        if len(lidar_coordinates):
            lidar_untrusted = reconstruction_mask[
                lidar_coordinates[:, 0],
                0,
                lidar_coordinates[:, 1],
                lidar_coordinates[:, 2],
            ] > 0.5
            trusted_lidar_coordinates = lidar_coordinates[~lidar_untrusted]
        else:
            trusted_lidar_coordinates = lidar_coordinates

        coordinate_parts = [trusted_lidar_coordinates]
        if radar_enabled:
            coordinate_parts.append(radar.sparse_coordinates)
        if include_repair_tokens:
            repair = torch.nonzero(
                reconstruction_mask[:, 0] > 0.5,
                as_tuple=False,
            )
            coordinate_parts.append(repair)
        nonempty_parts = [part for part in coordinate_parts if len(part)]
        if not nonempty_parts:
            raise ValueError("SST active token set is empty")
        union_keys = torch.unique(
            torch.cat(
                [self._flat_keys(part, height, width) for part in nonempty_parts]
            ),
            sorted=True,
        )
        coordinates = self._coordinates_from_keys(union_keys, height, width)
        batch_index, rows, cols = coordinates.unbind(dim=1)

        trusted_lidar_dense = lidar.dense_features * (1.0 - reconstruction_mask)
        lidar_features = trusted_lidar_dense[batch_index, :, rows, cols]
        if radar_enabled:
            radar_features = radar.dense_features[batch_index, :, rows, cols]
        else:
            radar_features = lidar_features.new_zeros(
                (len(coordinates), radar.dense_features.shape[1])
            )
        repair_values = reconstruction_mask[batch_index, 0, rows, cols, None]
        healthy_values = healthy_context_mask[batch_index, 0, rows, cols, None]
        fused = torch.cat(
            (lidar_features, radar_features, repair_values, healthy_values),
            dim=1,
        )
        if fused.shape[1] != self.input_channels:
            raise RuntimeError(
                f"SST fusion expected {self.input_channels} channels, got "
                f"{fused.shape[1]}"
            )
        if len(trusted_lidar_coordinates):
            values = reconstruction_mask[
                trusted_lidar_coordinates[:, 0],
                0,
                trusted_lidar_coordinates[:, 1],
                trusted_lidar_coordinates[:, 2],
            ]
            if torch.any(values > 0.5):
                raise AssertionError(
                    "Trusted LiDAR SST tokens include reconstruction-mask cells"
                )
        return SparseMultimodalTokens(
            features=self.projection(fused),
            coordinates=coordinates,
            trusted_lidar_coordinates=trusted_lidar_coordinates,
            statistics={
                "lidar_nonempty_pillars": lidar.statistics[
                    "nonempty_pillars"
                ].sum(),
                "trusted_lidar_pillars": torch.as_tensor(
                    len(trusted_lidar_coordinates),
                    device=coordinates.device,
                ),
                "radar_nonempty_pillars": radar.statistics[
                    "nonempty_pillars"
                ].sum() if radar_enabled else coordinates.new_zeros(()),
                "union_token_count": torch.as_tensor(
                    len(coordinates), device=coordinates.device
                ),
            },
        )


class LearnedXYPositionEncoding(nn.Module):
    def __init__(self, token_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, token_dim),
            nn.SiLU(inplace=True),
            nn.Linear(token_dim, token_dim),
        )

    def forward(
        self,
        coordinates: torch.Tensor,
        height: int,
        width: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        row = coordinates[:, 1].to(dtype)
        col = coordinates[:, 2].to(dtype)
        xy = torch.stack(
            (
                2.0 * col / max(width - 1, 1) - 1.0,
                2.0 * row / max(height - 1, 1) - 1.0,
            ),
            dim=1,
        )
        return self.net(xy)


class SparseRegionalAttention(nn.Module):
    """Pre-norm self-attention restricted to independent sparse regions."""

    def __init__(self, config: SSTConfig, *, shift_size_cells: int):
        super().__init__()
        self.region_size_cells = config.region_size_cells
        self.shift_size_cells = int(shift_size_cells)
        self.position_encoding = LearnedXYPositionEncoding(config.token_dim)
        self.attention_norm = nn.LayerNorm(config.token_dim)
        self.attention = nn.MultiheadAttention(
            config.token_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(config.token_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.token_dim, config.mlp_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden_dim, config.token_dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        grid_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = grid_shape
        group_coordinates = regional_group_indices(
            coordinates,
            self.region_size_cells,
            shift_size_cells=self.shift_size_cells,
        )
        _, group_assignment = torch.unique(
            group_coordinates,
            dim=0,
            sorted=True,
            return_inverse=True,
        )
        counts = torch.bincount(group_assignment)
        group_count = len(counts)
        maximum_tokens = int(counts.max())
        order = torch.argsort(group_assignment, stable=True)
        sorted_groups = group_assignment[order]
        starts = torch.cumsum(counts, dim=0) - counts
        positions = torch.arange(
            len(features), device=features.device
        ) - torch.repeat_interleave(starts, counts)
        normalized = self.attention_norm(features)
        normalized = normalized + self.position_encoding(
            coordinates,
            height,
            width,
            normalized.dtype,
        )
        packed = features.new_zeros(
            (group_count, maximum_tokens, features.shape[1])
        )
        valid = torch.zeros(
            (group_count, maximum_tokens),
            dtype=torch.bool,
            device=features.device,
        )
        packed[sorted_groups, positions] = normalized[order]
        valid[sorted_groups, positions] = True
        attended, _ = self.attention(
            packed,
            packed,
            packed,
            key_padding_mask=~valid,
            need_weights=False,
        )
        attended_sorted = attended[sorted_groups, positions]
        attended_unordered = torch.empty_like(features)
        attended_unordered[order] = attended_sorted
        features = features + attended_unordered
        features = features + self.mlp(self.mlp_norm(features))
        return features, counts


class SSTBlock(nn.Module):
    def __init__(self, config: SSTConfig):
        super().__init__()
        self.normal_attention = SparseRegionalAttention(
            config,
            shift_size_cells=0,
        )
        self.shifted_attention = SparseRegionalAttention(
            config,
            shift_size_cells=config.shift_size_cells,
        )

    def forward(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        grid_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        features, normal_counts = self.normal_attention(
            features, coordinates, grid_shape
        )
        features, shifted_counts = self.shifted_attention(
            features, coordinates, grid_shape
        )
        return features, (normal_counts, shifted_counts)


class SSTBackbone(nn.Module):
    def __init__(self, config: SSTConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.blocks = nn.ModuleList(
            SSTBlock(config) for _ in range(config.num_blocks)
        )

    def forward(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        grid_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        coordinates_before = coordinates.clone()
        first_block_features = None
        final_normal_counts = None
        for index, block in enumerate(self.blocks):
            features, (normal_counts, _) = block(
                features, coordinates, grid_shape
            )
            if index == 0:
                first_block_features = features
            final_normal_counts = normal_counts
        if not torch.equal(coordinates, coordinates_before):
            raise AssertionError("SST changed sparse pillar coordinates")
        assert first_block_features is not None
        assert final_normal_counts is not None
        return features, {
            "coordinates_before": coordinates_before,
            "coordinates_after": coordinates,
            "sst_block_1": first_block_features,
            "sst_block_final": features,
            "tokens_per_region": final_normal_counts,
        }


class SparseToDenseScatter(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = int(channels)

    def forward(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        batch_size: int,
        grid_shape: tuple[int, int],
    ) -> torch.Tensor:
        height, width = grid_shape
        dense = features.new_zeros(
            (batch_size, self.channels, height * width)
        )
        flat = coordinates[:, 1] * width + coordinates[:, 2]
        dense[coordinates[:, 0], :, flat] = features
        return dense.reshape(batch_size, self.channels, height, width)


class SSTReconstructionHead(nn.Module):
    def __init__(self, token_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(token_dim, 128, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(128), 128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(64), 64),
            nn.SiLU(inplace=True),
        )
        self.out_channels = 64

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.layers(tensor)


def region_statistics(
    counts: torch.Tensor,
    *,
    batch_size: int,
    grid_shape: tuple[int, int],
    region_size_cells: int,
) -> dict[str, torch.Tensor]:
    height, width = grid_shape
    total_regions = (
        batch_size
        * math.ceil(height / region_size_cells)
        * math.ceil(width / region_size_cells)
    )
    counts_float = counts.to(torch.float32)
    maximum_capacity = region_size_cells**2
    return {
        "minimum": counts_float.min(),
        "mean": counts_float.mean(),
        "median": counts_float.median(),
        "maximum": counts_float.max(),
        "empty_region_fraction": counts_float.new_tensor(
            1.0 - len(counts) / total_regions
        ),
        "maximum_capacity_region_fraction": (
            counts == maximum_capacity
        ).to(torch.float32).sum() / total_regions,
    }
