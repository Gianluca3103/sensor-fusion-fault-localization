"""PointPillarV3: within-pillar distribution and point-attention ablation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .pointpillars import (
    BEVGridGeometry,
    PointPillarsEncoder,
    PointPillarsOutput,
)


@dataclass(frozen=True)
class PointPillarsV3Config:
    enabled: bool = False
    output_channels: int = 64
    max_points_per_pillar: int = 100
    max_pillars: int | None = None
    lidar_use_reflectivity: bool = True
    radar_use_power: bool = True
    radar_use_radial_velocity: bool = True
    use_mean_pool: bool = True
    use_point_residual: bool = True
    point_residual_hidden_channels: int = 64
    initial_residual_scale: float = 0.1

    @property
    def lidar_raw_channels(self) -> int:
        return 4 if self.lidar_use_reflectivity else 3

    @property
    def radar_raw_channels(self) -> int:
        return 3 + int(self.radar_use_power) + int(
            self.radar_use_radial_velocity
        )

    @property
    def lidar_decorated_channels(self) -> int:
        return self.lidar_raw_channels + 5

    @property
    def radar_decorated_channels(self) -> int:
        return self.radar_raw_channels + 5

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("pointpillars_v3.enabled must be boolean")
        if self.output_channels < 1:
            raise ValueError("pointpillars_v3.output_channels must be positive")
        if self.max_points_per_pillar < 1:
            raise ValueError(
                "pointpillars_v3.max_points_per_pillar must be positive"
            )
        if self.max_pillars is not None and self.max_pillars < 1:
            raise ValueError("pointpillars_v3.max_pillars must be positive or null")
        if self.point_residual_hidden_channels < 1:
            raise ValueError(
                "pointpillars_v3.point_residual_hidden_channels must be positive"
            )
        if (
            isinstance(self.initial_residual_scale, bool)
            or not isinstance(self.initial_residual_scale, (int, float))
            or not math.isfinite(float(self.initial_residual_scale))
            or self.initial_residual_scale < 0
        ):
            raise ValueError(
                "pointpillars_v3.initial_residual_scale must be finite and "
                "non-negative"
            )
        for name in (
            "lidar_use_reflectivity",
            "radar_use_power",
            "radar_use_radial_velocity",
            "use_mean_pool",
            "use_point_residual",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"pointpillars_v3.{name} must be boolean")

    def to_dict(self) -> dict:
        return asdict(self)


class PillarFeatureNetV3(nn.Module):
    """Max, mean, and learned point-attention aggregation within each pillar."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        use_mean_pool: bool = True,
        use_point_residual: bool = True,
        point_residual_hidden_channels: int = 64,
        initial_residual_scale: float = 0.1,
        epsilon: float = 1.0e-8,
    ):
        super().__init__()
        if input_channels < 1 or output_channels < 1:
            raise ValueError("input_channels and output_channels must be positive")
        if point_residual_hidden_channels < 1:
            raise ValueError("point_residual_hidden_channels must be positive")
        if (
            not math.isfinite(initial_residual_scale)
            or initial_residual_scale < 0
        ):
            raise ValueError(
                "initial_residual_scale must be finite and non-negative"
            )
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        self.output_channels = int(output_channels)
        self.use_mean_pool = bool(use_mean_pool)
        self.use_point_residual = bool(use_point_residual)
        self.epsilon = float(epsilon)

        # Identical modules to the baseline shared point encoder.
        self.linear = nn.Linear(input_channels, output_channels, bias=False)
        self.normalization = nn.BatchNorm1d(output_channels)
        self.activation = nn.ReLU(inplace=True)

        if self.use_point_residual:
            self.point_residual_mlp = nn.Sequential(
                nn.Linear(input_channels, point_residual_hidden_channels),
                nn.SiLU(),
                nn.Linear(point_residual_hidden_channels, output_channels),
            )
            self.point_score = nn.Linear(output_channels, 1)
        else:
            self.point_residual_mlp = None
            self.point_score = None

        branch_count = 1 + int(self.use_mean_pool) + int(
            self.use_point_residual
        )
        if branch_count > 1:
            self.fusion = nn.Sequential(
                nn.Linear(branch_count * output_channels, output_channels),
                nn.LayerNorm(output_channels),
                nn.SiLU(),
                nn.Linear(output_channels, output_channels),
            )
            self.residual_scale = nn.Parameter(
                torch.tensor(float(initial_residual_scale), dtype=torch.float32)
            )
        else:
            self.fusion = None
            self.register_parameter("residual_scale", None)

    def _encode_points(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.linear(features)
        # BatchNorm cannot estimate variance from a single retained point while
        # training. Its running statistics give the natural finite fallback.
        if self.training and len(projected) == 1:
            projected = F.batch_norm(
                projected,
                self.normalization.running_mean,
                self.normalization.running_var,
                self.normalization.weight,
                self.normalization.bias,
                training=False,
                momentum=0.0,
                eps=self.normalization.eps,
            )
        else:
            projected = self.normalization(projected)
        return self.activation(projected)

    def _max_pool(
        self,
        encoded_points: torch.Tensor,
        point_to_pillar: torch.Tensor,
        pillar_count: int,
    ) -> torch.Tensor:
        pooled = encoded_points.new_full(
            (pillar_count, self.output_channels), -torch.inf
        )
        pooled.scatter_reduce_(
            0,
            point_to_pillar[:, None].expand(-1, self.output_channels),
            encoded_points,
            reduce="amax",
            include_self=True,
        )
        return pooled

    def _mean_pool(
        self,
        encoded_points: torch.Tensor,
        point_to_pillar: torch.Tensor,
        pillar_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        counts = torch.bincount(point_to_pillar, minlength=pillar_count)
        sums = encoded_points.new_zeros((pillar_count, self.output_channels))
        sums.index_add_(0, point_to_pillar, encoded_points)
        means = sums / counts.clamp_min(1).to(encoded_points.dtype).unsqueeze(1)
        return means, counts

    def _segment_softmax(
        self,
        scores: torch.Tensor,
        point_to_pillar: torch.Tensor,
        pillar_count: int,
    ) -> torch.Tensor:
        # AMP may leave ``scores`` in FP16 while ``torch.exp`` promotes its
        # result to FP32.  Accumulate the softmax in FP32 explicitly, then
        # return weights in the input dtype so the following weighted pooling
        # has matching source and destination dtypes.
        accumulation_scores = scores.float()
        maxima = accumulation_scores.new_full((pillar_count, 1), -torch.inf)
        maxima.scatter_reduce_(
            0,
            point_to_pillar[:, None],
            accumulation_scores,
            reduce="amax",
            include_self=True,
        )
        exponentials = torch.exp(
            accumulation_scores - maxima[point_to_pillar]
        )
        denominators = accumulation_scores.new_zeros((pillar_count, 1))
        denominators.index_add_(0, point_to_pillar, exponentials)
        weights = exponentials / denominators[point_to_pillar].clamp_min(
            self.epsilon
        )
        return weights.to(scores.dtype)

    def _attention_pool(
        self,
        features: torch.Tensor,
        point_to_pillar: torch.Tensor,
        pillar_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.point_residual_mlp is not None
        assert self.point_score is not None
        residual_points = self.point_residual_mlp(features)
        attention = self._segment_softmax(
            self.point_score(residual_points), point_to_pillar, pillar_count
        )
        pooled = residual_points.new_zeros((pillar_count, self.output_channels))
        pooled.index_add_(0, point_to_pillar, attention * residual_points)
        return pooled, attention

    def forward(
        self,
        features: torch.Tensor,
        point_to_pillar: torch.Tensor,
        pillar_count: int,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if features.ndim != 2:
            raise ValueError("features must have shape [N,D]")
        if point_to_pillar.shape != (len(features),):
            raise ValueError("point_to_pillar must have shape [N]")
        if pillar_count < 0:
            raise ValueError("pillar_count must be non-negative")
        if pillar_count == 0:
            empty = features.new_empty((0, self.output_channels))
            diagnostics = {
                "max_pooled": empty,
                "mean_pooled": empty,
                "point_residual_pooled": empty,
                "attention_weights": features.new_empty((0, 1)),
                "point_counts": torch.empty(
                    0, dtype=torch.long, device=features.device
                ),
            }
            return (empty, diagnostics) if return_diagnostics else empty

        encoded_points = self._encode_points(features)
        max_pooled = self._max_pool(
            encoded_points, point_to_pillar, pillar_count
        )
        mean_pooled, point_counts = self._mean_pool(
            encoded_points, point_to_pillar, pillar_count
        )

        branches = [max_pooled]
        if self.use_mean_pool:
            branches.append(mean_pooled)
        if self.use_point_residual:
            point_residual_pooled, attention = self._attention_pool(
                features, point_to_pillar, pillar_count
            )
            branches.append(point_residual_pooled)
        else:
            point_residual_pooled = torch.zeros_like(max_pooled)
            attention = features.new_zeros((len(features), 1))

        if self.fusion is None:
            pillar_features = max_pooled
        else:
            fusion_delta = self.fusion(torch.cat(branches, dim=1))
            pillar_features = max_pooled + self.residual_scale.to(
                max_pooled.dtype
            ) * fusion_delta

        if not return_diagnostics:
            return pillar_features
        return pillar_features, {
            "max_pooled": max_pooled,
            "mean_pooled": mean_pooled,
            "point_residual_pooled": point_residual_pooled,
            "attention_weights": attention,
            "point_counts": point_counts,
        }


class PointPillarsEncoderV3(PointPillarsEncoder):
    """Unchanged pillarization/scatter with V3 within-pillar feature pooling."""

    def __init__(
        self,
        geometry: BEVGridGeometry,
        *,
        raw_channels: int,
        output_channels: int,
        max_points_per_pillar: int,
        max_pillars: int | None,
        use_mean_pool: bool = True,
        use_point_residual: bool = True,
        point_residual_hidden_channels: int = 64,
        initial_residual_scale: float = 0.1,
    ):
        super().__init__(
            geometry,
            raw_channels=raw_channels,
            output_channels=output_channels,
            max_points_per_pillar=max_points_per_pillar,
            max_pillars=max_pillars,
        )
        self.feature_net = PillarFeatureNetV3(
            raw_channels + 5,
            output_channels,
            use_mean_pool=use_mean_pool,
            use_point_residual=use_point_residual,
            point_residual_hidden_channels=point_residual_hidden_channels,
            initial_residual_scale=initial_residual_scale,
        )

    @staticmethod
    def _v3_statistics(
        diagnostics: dict[str, torch.Tensor],
        point_to_pillar: torch.Tensor,
        pillar_batches: torch.Tensor,
        batch_size: int,
        residual_scale: torch.Tensor | None,
        use_point_residual: bool,
    ) -> dict[str, torch.Tensor]:
        device = pillar_batches.device
        counts = torch.bincount(pillar_batches, minlength=batch_size)

        def per_sample_mean(values: torch.Tensor) -> torch.Tensor:
            totals = torch.zeros(batch_size, device=device, dtype=torch.float32)
            if len(values):
                totals.index_add_(0, pillar_batches, values.detach().float())
            return totals / counts.clamp_min(1).float()

        def per_sample_max(values: torch.Tensor) -> torch.Tensor:
            maxima = torch.zeros(batch_size, device=device, dtype=torch.float32)
            if len(values):
                maxima.scatter_reduce_(
                    0,
                    pillar_batches,
                    values.detach().float(),
                    reduce="amax",
                    include_self=True,
                )
            return maxima

        attention = diagnostics["attention_weights"]
        point_counts = diagnostics["point_counts"]
        if use_point_residual and len(attention):
            # Autocast can produce FP16 attention while log/exp operations
            # promote intermediate values to FP32.  Diagnostics do not need
            # gradients, so calculate and aggregate them consistently in
            # FP32 instead of mixing source and destination dtypes.
            attention_values = attention.detach().float().squeeze(1)
            entropy_terms = -attention_values * torch.log(
                attention_values.clamp_min(1.0e-8)
            )
            entropy = torch.zeros(
                len(point_counts), device=device, dtype=torch.float32
            )
            entropy.index_add_(0, point_to_pillar, entropy_terms)
            maximum_attention = torch.zeros_like(entropy)
            maximum_attention.scatter_reduce_(
                0,
                point_to_pillar,
                attention_values,
                reduce="amax",
                include_self=True,
            )
            effective_points = torch.exp(entropy)
        else:
            entropy = torch.zeros(len(pillar_batches), device=device)
            maximum_attention = torch.zeros_like(entropy)
            effective_points = torch.zeros_like(entropy)
        max_mean_difference = (
            diagnostics["max_pooled"] - diagnostics["mean_pooled"]
        ).abs().mean(dim=1)
        scale_value = (
            torch.zeros((), device=device)
            if residual_scale is None
            else residual_scale.detach().float()
        )
        return {
            "average_attention_entropy": per_sample_mean(entropy),
            "maximum_attention_weight": per_sample_max(maximum_attention),
            "average_max_mean_feature_difference": per_sample_mean(
                max_mean_difference
            ),
            "average_points_receiving_attention": per_sample_mean(
                effective_points
            ),
            "residual_scale": scale_value.expand(batch_size),
        }

    def forward(
        self,
        point_clouds: Sequence[torch.Tensor],
        *,
        return_sparse: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | PointPillarsOutput:
        if not point_clouds:
            raise ValueError("point_clouds must contain at least one sample")
        (
            features,
            point_to_pillar,
            pillar_batches,
            pillar_rows,
            pillar_cols,
            statistics,
        ) = self._pillarize_batch(point_clouds)
        pillar_features, diagnostics = self.feature_net(
            features,
            point_to_pillar,
            len(pillar_rows),
            return_diagnostics=True,
        )
        statistics.update(
            self._v3_statistics(
                diagnostics,
                point_to_pillar,
                pillar_batches,
                len(point_clouds),
                self.feature_net.residual_scale,
                self.feature_net.use_point_residual,
            )
        )
        dense_features = self.scatter.forward_batch(
            pillar_features,
            pillar_batches,
            pillar_rows,
            pillar_cols,
            len(point_clouds),
        )
        if not return_sparse:
            return dense_features, statistics
        sparse_coordinates = torch.stack(
            (pillar_batches, pillar_rows, pillar_cols), dim=1
        )
        return PointPillarsOutput(
            dense_features=dense_features,
            sparse_features=pillar_features,
            sparse_coordinates=sparse_coordinates,
            statistics=statistics,
        )
