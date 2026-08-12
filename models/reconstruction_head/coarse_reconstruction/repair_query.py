"""Regional repair-query decoder for sparse LiDAR/Radar reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import torch
from torch import nn

from ..pointpillars import PointPillarsOutput
from .sst_backbone import regional_group_indices


def _profile_start(tensor: torch.Tensor, enabled: bool) -> float | None:
    if not enabled:
        return None
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)
    return time.perf_counter()


def _profile_elapsed_ms(tensor: torch.Tensor, started: float | None) -> float:
    if started is None:
        return 0.0
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)
    return 1000.0 * (time.perf_counter() - started)


@dataclass(frozen=True)
class RepairQueryConfig:
    token_dim: int = 128
    num_heads: int = 8
    num_decoder_blocks: int = 3
    mlp_hidden_dim: int = 256
    dropout: float = 0.0
    region_size_cells: int = 12
    context_region_radius: int = 1
    use_radar_query_feature: bool = True
    use_presence_flags: bool = True
    shifted_query_attention: bool = True

    def validate(self) -> None:
        for name in (
            "token_dim",
            "num_heads",
            "num_decoder_blocks",
            "mlp_hidden_dim",
            "region_size_cells",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"repair_query.{name} must be positive")
        if self.token_dim % self.num_heads:
            raise ValueError(
                "repair_query.token_dim must be divisible by num_heads"
            )
        if self.context_region_radius < 0:
            raise ValueError(
                "repair_query.context_region_radius cannot be negative"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("repair_query.dropout must be in [0,1)")
        for name in (
            "use_radar_query_feature",
            "use_presence_flags",
            "shifted_query_attention",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"repair_query.{name} must be boolean")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RepairQueryTokens:
    context_features: torch.Tensor
    context_coordinates: torch.Tensor
    query_features: torch.Tensor
    query_coordinates: torch.Tensor
    trusted_lidar_coordinates: torch.Tensor
    radar_coordinates: torch.Tensor
    statistics: dict[str, torch.Tensor]


@dataclass(frozen=True)
class PackedLayout:
    counts: torch.Tensor
    order: torch.Tensor
    groups: torch.Tensor
    positions: torch.Tensor
    valid: torch.Tensor


def _flat_keys(
    coordinates: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    return (
        coordinates[:, 0] * height * width
        + coordinates[:, 1] * width
        + coordinates[:, 2]
    )


def _coordinates_from_keys(
    keys: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    batch = torch.div(keys, height * width, rounding_mode="floor")
    spatial = keys.remainder(height * width)
    row = torch.div(spatial, width, rounding_mode="floor")
    col = spatial.remainder(width)
    return torch.stack((batch, row, col), dim=1)


def _presence_grid(
    coordinates: torch.Tensor,
    batch_size: int,
    height: int,
    width: int,
) -> torch.Tensor:
    present = torch.zeros(
        (batch_size, height, width),
        dtype=torch.bool,
        device=coordinates.device,
    )
    if len(coordinates):
        present[
            coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
        ] = True
    return present


def normalized_xy(
    coordinates: torch.Tensor,
    grid_shape: tuple[int, int],
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert `[batch,row,col]` into normalized physical X/Y coordinates."""
    height, width = grid_shape
    row = coordinates[:, 1].to(dtype)
    col = coordinates[:, 2].to(dtype)
    # BEV row zero is maximum forward X; columns increase with physical Y.
    x_norm = 1.0 - 2.0 * row / max(height - 1, 1)
    y_norm = -1.0 + 2.0 * col / max(width - 1, 1)
    return torch.stack((x_norm, y_norm), dim=1)


class XYPositionEncoder(nn.Module):
    def __init__(self, token_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )

    def forward(
        self,
        coordinates: torch.Tensor,
        grid_shape: tuple[int, int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        xy = normalized_xy(coordinates, grid_shape, dtype=dtype)
        return self.layers(xy)


def _pack_layout(group_assignment: torch.Tensor) -> PackedLayout:
    counts = torch.bincount(group_assignment)
    order = torch.argsort(group_assignment, stable=True)
    groups = group_assignment[order]
    starts = torch.cumsum(counts, dim=0) - counts
    positions = torch.arange(
        len(group_assignment), device=group_assignment.device
    ) - torch.repeat_interleave(starts, counts)
    valid = torch.zeros(
        (len(counts), int(counts.max())),
        dtype=torch.bool,
        device=group_assignment.device,
    )
    valid[groups, positions] = True
    return PackedLayout(counts, order, groups, positions, valid)


def _pack_features(features: torch.Tensor, layout: PackedLayout) -> torch.Tensor:
    packed = features.new_zeros((*layout.valid.shape, features.shape[1]))
    packed[layout.groups, layout.positions] = features[layout.order]
    return packed


def _unpack_features(packed: torch.Tensor, layout: PackedLayout) -> torch.Tensor:
    sorted_features = packed[layout.groups, layout.positions]
    output = sorted_features.new_empty(sorted_features.shape)
    output[layout.order] = sorted_features
    return output


def _group_assignment(
    coordinates: torch.Tensor,
    region_size_cells: int,
    shift_size_cells: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    groups = regional_group_indices(
        coordinates,
        region_size_cells,
        shift_size_cells=shift_size_cells,
    )
    unique_groups, assignment = torch.unique(
        groups, dim=0, sorted=True, return_inverse=True
    )
    return unique_groups, assignment


def _context_neighborhood_layout(
    query_coordinates: torch.Tensor,
    context_coordinates: torch.Tensor,
    grid_shape: tuple[int, int],
    region_size_cells: int,
    radius: int,
) -> tuple[PackedLayout, PackedLayout, torch.Tensor]:
    """Pack queries and duplicated local context by active query region."""
    height, width = grid_shape
    region_rows = math.ceil(height / region_size_cells)
    region_cols = math.ceil(width / region_size_cells)
    query_regions = regional_group_indices(query_coordinates, region_size_cells)
    context_regions = regional_group_indices(
        context_coordinates, region_size_cells
    )
    query_keys = (
        query_regions[:, 0] * region_rows * region_cols
        + query_regions[:, 1] * region_cols
        + query_regions[:, 2]
    )
    active_keys, query_assignment = torch.unique(
        query_keys, sorted=True, return_inverse=True
    )
    query_layout = _pack_layout(query_assignment)

    context_indices = []
    target_assignments = []
    base_indices = torch.arange(
        len(context_coordinates), device=context_coordinates.device
    )
    for row_offset in range(-radius, radius + 1):
        for col_offset in range(-radius, radius + 1):
            target_row = context_regions[:, 1] + row_offset
            target_col = context_regions[:, 2] + col_offset
            in_grid = (
                (target_row >= 0)
                & (target_row < region_rows)
                & (target_col >= 0)
                & (target_col < region_cols)
            )
            target_keys = (
                context_regions[:, 0] * region_rows * region_cols
                + target_row * region_cols
                + target_col
            )
            positions = torch.searchsorted(active_keys, target_keys)
            safe_positions = positions.clamp(max=max(len(active_keys) - 1, 0))
            matched = in_grid & (positions < len(active_keys))
            matched &= active_keys[safe_positions] == target_keys
            if torch.any(matched):
                context_indices.append(base_indices[matched])
                target_assignments.append(positions[matched])
    if not context_indices:
        empty = torch.empty(
            0, dtype=torch.long, device=query_coordinates.device
        )
        empty_layout = PackedLayout(
            counts=torch.zeros(
                len(active_keys), dtype=torch.long, device=empty.device
            ),
            order=empty,
            groups=empty,
            positions=empty,
            valid=torch.zeros(
                (len(active_keys), 0), dtype=torch.bool, device=empty.device
            ),
        )
        return query_layout, empty_layout, empty
    context_indices_tensor = torch.cat(context_indices)
    context_assignment = torch.cat(target_assignments)
    context_layout = _pack_layout(context_assignment)
    # Every active query group has at least one query, but not necessarily context.
    if len(context_layout.counts) < len(active_keys):
        padded_counts = torch.zeros_like(active_keys)
        padded_counts[: len(context_layout.counts)] = context_layout.counts
        context_layout = PackedLayout(
            padded_counts,
            context_layout.order,
            context_layout.groups,
            context_layout.positions,
            torch.nn.functional.pad(
                context_layout.valid,
                (0, 0, 0, len(active_keys) - len(context_layout.counts)),
            ),
        )
    return query_layout, context_layout, context_indices_tensor


class RepairQueryTokenBuilder(nn.Module):
    def __init__(
        self,
        lidar_channels: int,
        radar_channels: int,
        config: RepairQueryConfig,
        position_encoder: XYPositionEncoder,
    ):
        super().__init__()
        self.config = config
        self.position_encoder = position_encoder
        context_channels = lidar_channels + radar_channels
        if config.use_presence_flags:
            context_channels += 2
        self.context_projection = nn.Linear(context_channels, config.token_dim)
        self.repair_embedding = nn.Parameter(torch.empty(config.token_dim))
        nn.init.normal_(self.repair_embedding, std=0.02)
        self.radar_query_projection = nn.Linear(
            radar_channels, config.token_dim
        )
        self.radar_presence_projection = (
            nn.Linear(1, config.token_dim)
            if config.use_presence_flags
            else None
        )

    def forward(
        self,
        lidar: PointPillarsOutput,
        radar: PointPillarsOutput,
        reconstruction_mask: torch.Tensor,
        *,
        radar_enabled: bool,
    ) -> RepairQueryTokens:
        batch, _, height, width = lidar.dense_features.shape
        lidar_coordinates = lidar.sparse_coordinates
        if len(lidar_coordinates):
            untrusted = reconstruction_mask[
                lidar_coordinates[:, 0],
                0,
                lidar_coordinates[:, 1],
                lidar_coordinates[:, 2],
            ] > 0.5
            trusted_lidar_coordinates = lidar_coordinates[~untrusted]
        else:
            trusted_lidar_coordinates = lidar_coordinates
        radar_coordinates = (
            radar.sparse_coordinates
            if radar_enabled
            else radar.sparse_coordinates.new_empty((0, 3))
        )
        coordinate_parts = [trusted_lidar_coordinates, radar_coordinates]
        coordinate_parts = [part for part in coordinate_parts if len(part)]
        if coordinate_parts:
            union_keys = torch.unique(
                torch.cat(
                    [_flat_keys(part, height, width) for part in coordinate_parts]
                ),
                sorted=True,
            )
            context_coordinates = _coordinates_from_keys(
                union_keys, height, width
            )
        else:
            context_coordinates = lidar_coordinates.new_empty((0, 3))
        batch_index, rows, cols = context_coordinates.unbind(dim=1)
        trusted_dense = lidar.dense_features * (1.0 - reconstruction_mask)
        lidar_features = trusted_dense[batch_index, :, rows, cols]
        radar_features = radar.dense_features[batch_index, :, rows, cols]
        lidar_present_grid = _presence_grid(
            trusted_lidar_coordinates, batch, height, width
        )
        radar_present_grid = _presence_grid(
            radar_coordinates, batch, height, width
        )
        lidar_present = lidar_present_grid[batch_index, rows, cols, None]
        radar_present = radar_present_grid[batch_index, rows, cols, None]
        context_parts = [lidar_features, radar_features]
        if self.config.use_presence_flags:
            context_parts.extend(
                [
                    lidar_present.to(lidar_features.dtype),
                    radar_present.to(lidar_features.dtype),
                ]
            )
        context_features = self.context_projection(torch.cat(context_parts, dim=1))
        context_features = context_features + self.position_encoder(
            context_coordinates, (height, width), context_features.dtype
        )

        query_coordinates = torch.nonzero(
            reconstruction_mask[:, 0] > 0.5, as_tuple=False
        )
        query_batch, query_rows, query_cols = query_coordinates.unbind(dim=1)
        query_radar = radar.dense_features[
            query_batch, :, query_rows, query_cols
        ]
        query_radar_present = radar_present_grid[
            query_batch, query_rows, query_cols, None
        ]
        query_features = self.repair_embedding.to(query_radar.dtype).expand(
            len(query_coordinates), -1
        )
        query_features = query_features + self.position_encoder(
            query_coordinates, (height, width), query_radar.dtype
        )
        if self.config.use_radar_query_feature:
            query_features = query_features + self.radar_query_projection(
                query_radar
            )
        if self.radar_presence_projection is not None:
            query_features = query_features + self.radar_presence_projection(
                query_radar_present.to(query_radar.dtype)
            )

        if len(trusted_lidar_coordinates):
            leaked = reconstruction_mask[
                trusted_lidar_coordinates[:, 0],
                0,
                trusted_lidar_coordinates[:, 1],
                trusted_lidar_coordinates[:, 2],
            ] > 0.5
            if torch.any(leaked):
                raise AssertionError(
                    "Trusted LiDAR repair-query context leaks into repair mask"
                )
        repair_per_sample = torch.bincount(
            query_coordinates[:, 0], minlength=batch
        )
        trusted_per_sample = torch.bincount(
            trusted_lidar_coordinates[:, 0], minlength=batch
        )
        radar_per_sample = torch.bincount(
            radar_coordinates[:, 0], minlength=batch
        )
        context_per_sample = torch.bincount(
            context_coordinates[:, 0], minlength=batch
        )
        radar_repair_per_sample = torch.zeros(
            batch, dtype=torch.long, device=query_coordinates.device
        )
        radar_repair_per_sample.index_add_(
            0, query_coordinates[:, 0], query_radar_present[:, 0].long()
        )
        denominator = repair_per_sample.clamp_min(1).to(torch.float32)
        statistics = {
            "repair_queries_per_sample": repair_per_sample,
            "trusted_lidar_context_tokens_per_sample": trusted_per_sample,
            "radar_context_tokens_per_sample": radar_per_sample,
            "union_context_tokens_per_sample": context_per_sample,
            "repair_with_radar_fraction": radar_repair_per_sample / denominator,
            "repair_without_radar_fraction": 1.0 - radar_repair_per_sample / denominator,
            "trusted_lidar_tokens_inside_repair_mask": leaked.long().sum()
            if len(trusted_lidar_coordinates)
            else repair_per_sample.new_zeros(()),
        }
        return RepairQueryTokens(
            context_features=context_features,
            context_coordinates=context_coordinates,
            query_features=query_features,
            query_coordinates=query_coordinates,
            trusted_lidar_coordinates=trusted_lidar_coordinates,
            radar_coordinates=radar_coordinates,
            statistics=statistics,
        )


class RepairQueryDecoderBlock(nn.Module):
    def __init__(self, config: RepairQueryConfig, block_index: int):
        super().__init__()
        self.config = config
        self.shift_size_cells = (
            config.region_size_cells // 2
            if config.shifted_query_attention and block_index % 2 == 1
            else 0
        )
        self.self_norm = nn.LayerNorm(config.token_dim)
        self.self_attention = nn.MultiheadAttention(
            config.token_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.cross_query_norm = nn.LayerNorm(config.token_dim)
        self.cross_context_norm = nn.LayerNorm(config.token_dim)
        self.cross_attention = nn.MultiheadAttention(
            config.token_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(config.token_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.token_dim, config.mlp_hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden_dim, config.token_dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        query_coordinates: torch.Tensor,
        context: torch.Tensor,
        context_coordinates: torch.Tensor,
        grid_shape: tuple[int, int],
        *,
        profile: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        self_started = _profile_start(queries, profile)
        _, self_assignment = _group_assignment(
            query_coordinates,
            self.config.region_size_cells,
            self.shift_size_cells,
        )
        self_layout = _pack_layout(self_assignment)
        packed_queries = _pack_features(self.self_norm(queries), self_layout)
        self_output, _ = self.self_attention(
            packed_queries,
            packed_queries,
            packed_queries,
            key_padding_mask=~self_layout.valid,
            need_weights=False,
        )
        self_output = self_output.masked_fill(
            ~self_layout.valid[..., None], 0.0
        )
        queries = queries + _unpack_features(self_output, self_layout)
        self_ms = _profile_elapsed_ms(queries, self_started)

        cross_started = _profile_start(queries, profile)
        query_layout, context_layout, context_indices = (
            _context_neighborhood_layout(
                query_coordinates,
                context_coordinates,
                grid_shape,
                self.config.region_size_cells,
                self.config.context_region_radius,
            )
        )
        contexts_per_query = context_layout.counts[
            query_layout.groups[torch.argsort(query_layout.order)]
        ]
        if context_layout.valid.shape[1] > 0:
            packed_query = _pack_features(
                self.cross_query_norm(queries), query_layout
            )
            duplicated_context = self.cross_context_norm(
                context[context_indices]
            )
            packed_context = _pack_features(
                duplicated_context, context_layout
            )
            groups_with_context = context_layout.counts > 0
            cross_output = torch.zeros_like(packed_query)
            attended, _ = self.cross_attention(
                packed_query[groups_with_context],
                packed_context[groups_with_context],
                packed_context[groups_with_context],
                key_padding_mask=~context_layout.valid[groups_with_context],
                need_weights=False,
            )
            cross_output[groups_with_context] = attended.to(
                cross_output.dtype
            )
            cross_output = cross_output.masked_fill(
                ~query_layout.valid[..., None], 0.0
            )
            queries = queries + _unpack_features(cross_output, query_layout)
        cross_ms = _profile_elapsed_ms(queries, cross_started)
        queries = queries + self.mlp(self.mlp_norm(queries))
        return queries, {
            "query_region_counts": self_layout.counts,
            "contexts_per_query": contexts_per_query,
            "self_attention_ms": self_ms,
            "cross_attention_ms": cross_ms,
        }


def _per_sample_distribution(
    values: torch.Tensor,
    sample_ids: torch.Tensor,
    batch_size: int,
    prefix: str,
) -> dict[str, torch.Tensor]:
    result = {}
    for batch_index in range(batch_size):
        selected = values[sample_ids == batch_index].to(torch.float32)
        if not len(selected):
            selected = values.new_zeros(1, dtype=torch.float32)
        result.setdefault(f"{prefix}_minimum", []).append(selected.min())
        result.setdefault(f"{prefix}_mean", []).append(selected.mean())
        result.setdefault(f"{prefix}_median", []).append(selected.median())
        result.setdefault(f"{prefix}_p90", []).append(
            torch.quantile(selected, 0.9)
        )
        result.setdefault(f"{prefix}_maximum", []).append(selected.max())
    return {name: torch.stack(entries) for name, entries in result.items()}


class RepairQueryDecoder(nn.Module):
    def __init__(
        self,
        lidar_channels: int,
        radar_channels: int,
        config: RepairQueryConfig,
    ):
        super().__init__()
        config.validate()
        self.config = config
        self.position_encoder = XYPositionEncoder(config.token_dim)
        self.token_builder = RepairQueryTokenBuilder(
            lidar_channels,
            radar_channels,
            config,
            self.position_encoder,
        )
        self.blocks = nn.ModuleList(
            RepairQueryDecoderBlock(config, index)
            for index in range(config.num_decoder_blocks)
        )
        self.occupancy_head = nn.Linear(config.token_dim, 1)
        self.height_head = nn.Linear(config.token_dim, 1)
        self.density_head = nn.Linear(config.token_dim, 1)

    def forward(
        self,
        lidar: PointPillarsOutput,
        radar: PointPillarsOutput,
        reconstruction_mask: torch.Tensor,
        *,
        radar_enabled: bool,
        profile: bool = False,
    ) -> dict[str, torch.Tensor | dict]:
        token_started = _profile_start(lidar.dense_features, profile)
        tokens = self.token_builder(
            lidar, radar, reconstruction_mask, radar_enabled=radar_enabled
        )
        token_ms = _profile_elapsed_ms(tokens.query_features, token_started)
        queries = tokens.query_features
        block_debug = []
        if len(tokens.query_coordinates):
            for block in self.blocks:
                queries, debug = block(
                    queries,
                    tokens.query_coordinates,
                    tokens.context_features,
                    tokens.context_coordinates,
                    reconstruction_mask.shape[-2:],
                    profile=profile,
                )
                block_debug.append(debug)
        output_started = _profile_start(queries, profile)
        occupancy = self.occupancy_head(queries)
        density = self.density_head(queries)
        height = self.height_head(queries)
        batch_size, _, grid_height, grid_width = reconstruction_mask.shape
        replacement_raw = queries.new_zeros(
            (batch_size, 3, grid_height, grid_width)
        )
        batch_index, rows, cols = tokens.query_coordinates.unbind(dim=1)
        replacement_raw[batch_index, 0, rows, cols] = occupancy[:, 0].to(
            replacement_raw.dtype
        )
        replacement_raw[batch_index, 1, rows, cols] = density[:, 0].to(
            replacement_raw.dtype
        )
        replacement_raw[batch_index, 2, rows, cols] = height[:, 0].to(
            replacement_raw.dtype
        )
        output_ms = _profile_elapsed_ms(replacement_raw, output_started)

        statistics = dict(tokens.statistics)
        if block_debug:
            contexts_per_query = block_debug[-1]["contexts_per_query"]
            statistics.update(
                _per_sample_distribution(
                    contexts_per_query,
                    tokens.query_coordinates[:, 0],
                    batch_size,
                    "context_tokens_per_query",
                )
            )
            normal_regions = regional_group_indices(
                tokens.query_coordinates, self.config.region_size_cells
            )
            region_keys = (
                normal_regions[:, 0]
                * math.ceil(grid_height / self.config.region_size_cells)
                * math.ceil(grid_width / self.config.region_size_cells)
                + normal_regions[:, 1]
                * math.ceil(grid_width / self.config.region_size_cells)
                + normal_regions[:, 2]
            )
            unique_region_keys, region_counts = torch.unique(
                region_keys, sorted=True, return_counts=True
            )
            region_samples = torch.div(
                unique_region_keys,
                math.ceil(grid_height / self.config.region_size_cells)
                * math.ceil(grid_width / self.config.region_size_cells),
                rounding_mode="floor",
            )
            query_region_stats = _per_sample_distribution(
                region_counts,
                region_samples,
                batch_size,
                "repair_queries_per_region",
            )
            query_region_stats.pop("repair_queries_per_region_p90")
            statistics.update(query_region_stats)
        timing = {
            "token_construction_ms": token_ms,
            "query_self_attention_ms": sum(
                float(debug["self_attention_ms"]) for debug in block_debug
            ),
            "cross_attention_ms": sum(
                float(debug["cross_attention_ms"]) for debug in block_debug
            ),
            "output_scatter_ms": output_ms,
        }
        return {
            "replacement_raw": replacement_raw,
            "query_features": queries,
            "query_coordinates": tokens.query_coordinates,
            "context_features": tokens.context_features,
            "context_coordinates": tokens.context_coordinates,
            "trusted_lidar_coordinates": tokens.trusted_lidar_coordinates,
            "radar_context_coordinates": tokens.radar_coordinates,
            "statistics": statistics,
            "timing_ms": timing,
        }
