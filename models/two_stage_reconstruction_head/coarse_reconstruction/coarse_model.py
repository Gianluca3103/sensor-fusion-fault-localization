"""VoD PointPillars + HRNet coarse LiDAR reconstruction model."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .coarse_config import CoarseReconstructionConfig
from .hrnet_backbone import HRNetBackbone
from ..PointPillarV2 import PointPillarsEncoderV2, PointPillarsV2Config
from ..PointPillarV3 import PointPillarsEncoderV3, PointPillarsV3Config
from ..pointpillars import BEVGridGeometry, PointPillarsEncoder
from ..reconstruction_crop import ReconstructionCropExtractor
from ..reconstruction_inputs import ReconstructionInputs


class CoarseReplacementHead(nn.Module):
    """Predict occupancy logits, density, and height at full BEV resolution."""

    def __init__(self, in_channels: int, lidar_channels: int = 3):
        super().__init__()
        # Keep the historical attribute name so existing HRNet checkpoints
        # remain loadable after the architecture cleanup.
        self.head = nn.Conv2d(in_channels, lidar_channels, kernel_size=1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.head(tensor)


class CoarseReconstructionModel(nn.Module):
    """Reconstruct selected VoD LiDAR cells from LiDAR and Radar context."""

    def __init__(
        self,
        config: CoarseReconstructionConfig | None = None,
        *,
        grid_geometry: BEVGridGeometry | None = None,
    ):
        super().__init__()
        self.config = config or CoarseReconstructionConfig()
        self.config.validate()
        self.grid_geometry = grid_geometry

        self.lidar_pillar_encoder = None
        self.radar_pillar_encoder = None
        if self.config.pointpillars_enabled:
            if grid_geometry is None:
                raise ValueError(
                    "grid_geometry is required when PointPillars is enabled"
                )
            grid_geometry.validate()
            if (grid_geometry.height, grid_geometry.width) != (320, 320):
                raise ValueError("VoD reconstruction requires a 320x320 BEV grid")
            pointpillars = self.config.pillar_encoder_config
            if isinstance(pointpillars, PointPillarsV3Config):
                encoder_class = PointPillarsEncoderV3
            elif isinstance(pointpillars, PointPillarsV2Config):
                encoder_class = PointPillarsEncoderV2
            else:
                encoder_class = PointPillarsEncoder
            common_arguments = {
                "output_channels": pointpillars.output_channels,
                "max_points_per_pillar": pointpillars.max_points_per_pillar,
                "max_pillars": pointpillars.max_pillars,
            }
            if isinstance(pointpillars, PointPillarsV2Config):
                common_arguments.update(
                    {
                        "neighbor_enabled": pointpillars.neighbor_enabled,
                        "neighbor_radius_m": pointpillars.neighbor_radius_m,
                        "neighbor_max_neighbors": pointpillars.neighbor_max_neighbors,
                        "neighbor_initial_scale": pointpillars.neighbor_initial_scale,
                    }
                )
            elif isinstance(pointpillars, PointPillarsV3Config):
                common_arguments.update(
                    {
                        "use_mean_pool": pointpillars.use_mean_pool,
                        "use_point_residual": pointpillars.use_point_residual,
                        "point_residual_hidden_channels": (
                            pointpillars.point_residual_hidden_channels
                        ),
                        "initial_residual_scale": (
                            pointpillars.initial_residual_scale
                        ),
                    }
                )
            self.lidar_pillar_encoder = encoder_class(
                grid_geometry,
                raw_channels=pointpillars.lidar_raw_channels,
                **common_arguments,
            )
            self.radar_pillar_encoder = encoder_class(
                grid_geometry,
                raw_channels=pointpillars.radar_raw_channels,
                **common_arguments,
            )

        self.hrnet_backbone = HRNetBackbone(
            self.config.local_input_channels,
            self.config.hrnet,
        )
        self.replacement_head = CoarseReplacementHead(
            self.hrnet_backbone.out_channels,
            self.config.target_lidar_channels,
        )
        self.context_crop_extractor = (
            ReconstructionCropExtractor(
                pad_multiple=self.config.context_crop_pad_multiple,
                minimum_size=self.config.minimum_context_crop_size,
            )
            if self.config.minimum_context_crop_size > 0
            else None
        )

    @staticmethod
    def _pad_spatial(
        tensor: torch.Tensor,
        target_height: int,
        target_width: int,
    ) -> torch.Tensor:
        height, width = tensor.shape[-2:]
        return F.pad(tensor, (0, target_width - width, 0, target_height - height))

    def _run_context_buckets(
        self,
        local_input: torch.Tensor,
        crops,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        torch.Tensor,
    ]:
        """Run HRNet in bounded-padding shape buckets for GPU efficiency."""

        exact_shape_pairs = list(
            zip(crops.padded_heights.tolist(), crops.padded_widths.tolist())
        )
        maximum_height, maximum_width = local_input.shape[-2:]
        bucket_multiple = self.config.context_shape_bucket_multiple

        def bucket_dimension(value: int, maximum: int) -> int:
            rounded = ((value + bucket_multiple - 1) // bucket_multiple) * bucket_multiple
            return min(rounded, maximum)

        shape_pairs = [
            (
                bucket_dimension(height, maximum_height),
                bucket_dimension(width, maximum_width),
            )
            for height, width in exact_shape_pairs
        ]
        buckets: dict[tuple[int, int], list[int]] = {}
        for index, shape in enumerate(shape_pairs):
            buckets.setdefault(shape, []).append(index)

        sample_features: list[torch.Tensor | None] = [None] * local_input.shape[0]
        sample_replacements: list[torch.Tensor | None] = [None] * local_input.shape[0]
        sample_debug: dict[str, list[torch.Tensor | None]] = {}
        for (height, width), sample_indices in buckets.items():
            indices = torch.tensor(
                sample_indices, device=local_input.device, dtype=torch.long
            )
            bucket_input = local_input.index_select(0, indices)[..., :height, :width]
            bucket_features, bucket_debug = self.hrnet_backbone(bucket_input)
            bucket_replacement = self.replacement_head(bucket_features)
            for bucket_index, sample_index in enumerate(sample_indices):
                sample_features[sample_index] = bucket_features[bucket_index : bucket_index + 1]
                sample_replacements[sample_index] = bucket_replacement[
                    bucket_index : bucket_index + 1
                ]
                for name, value in bucket_debug.items():
                    sample_debug.setdefault(
                        name, [None] * local_input.shape[0]
                    )[sample_index] = value[bucket_index : bucket_index + 1]

        def combine(values: list[torch.Tensor | None]) -> torch.Tensor:
            present = [value for value in values if value is not None]
            if len(present) != len(values):
                raise RuntimeError("Context bucket output is incomplete")
            target_height = max(value.shape[-2] for value in present)
            target_width = max(value.shape[-1] for value in present)
            return torch.cat(
                [self._pad_spatial(value, target_height, target_width) for value in present],
                dim=0,
            )

        features = combine(sample_features)
        replacements = combine(sample_replacements)
        debug = {name: combine(values) for name, values in sample_debug.items()}
        bucket_shapes = crops.padded_heights.new_tensor(shape_pairs)
        return features, replacements, debug, bucket_shapes

    def _select_lidar_fields(
        self, point_clouds: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        for points in point_clouds:
            if points.ndim != 2 or points.shape[1] != 4:
                raise ValueError(
                    "faulty_lidar_points must contain aligned [x,y,z,reflectivity] rows"
                )
        if self.config.pillar_encoder_config.lidar_use_reflectivity:
            return tuple(point_clouds)
        return tuple(points[:, :3] for points in point_clouds)

    def _select_radar_fields(
        self, point_clouds: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        selected = []
        for points in point_clouds:
            if points.ndim != 2 or points.shape[1] != 5:
                raise ValueError(
                    "radar_points must contain aligned [x,y,z,power,doppler] rows"
                )
            columns = [points[:, :3]]
            if self.config.pillar_encoder_config.radar_use_power:
                columns.append(points[:, 3:4])
            if self.config.pillar_encoder_config.radar_use_radial_velocity:
                columns.append(points[:, 4:5])
            selected.append(torch.cat(columns, dim=1))
        return tuple(selected)

    def _sensor_features(
        self,
        faulty_lidar_bev: torch.Tensor,
        radar_bev: torch.Tensor,
        faulty_lidar_points: Sequence[torch.Tensor] | None,
        radar_points: Sequence[torch.Tensor] | None,
        *,
        radar_enabled: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        if not self.config.pointpillars_enabled:
            radar_features = radar_bev if radar_enabled else torch.zeros_like(radar_bev)
            return faulty_lidar_bev, radar_features, {}, {}

        if faulty_lidar_points is None:
            raise ValueError("faulty_lidar_points are required by PointPillars")
        assert self.lidar_pillar_encoder is not None
        assert self.radar_pillar_encoder is not None
        lidar_features, lidar_statistics = self.lidar_pillar_encoder(
            self._select_lidar_fields(faulty_lidar_points)
        )
        if radar_enabled:
            if radar_points is None:
                raise ValueError("radar_points are required by PointPillars")
            radar_features, radar_statistics = self.radar_pillar_encoder(
                self._select_radar_fields(radar_points)
            )
        else:
            radar_features = lidar_features.new_zeros(
                (
                    lidar_features.shape[0],
                    self.config.radar_channels,
                    lidar_features.shape[2],
                    lidar_features.shape[3],
                )
            )
            radar_statistics = {}
        return lidar_features, radar_features, lidar_statistics, radar_statistics

    def forward(
        self,
        faulty_lidar_bev: torch.Tensor,
        radar_bev: torch.Tensor,
        reconstruction_mask: torch.Tensor,
        healthy_context_mask: torch.Tensor,
        halo_mask: torch.Tensor,
        *,
        faulty_lidar_points: Sequence[torch.Tensor] | None = None,
        radar_points: Sequence[torch.Tensor] | None = None,
        radar_enabled: bool = True,
        shared_inputs: ReconstructionInputs | None = None,
    ) -> dict[str, torch.Tensor]:
        if shared_inputs is None:
            shared_inputs = ReconstructionInputs(
                faulty_lidar_bev=faulty_lidar_bev,
                radar_bev=radar_bev,
                reconstruction_mask=reconstruction_mask,
                healthy_context_mask=healthy_context_mask,
                halo_mask=halo_mask,
                faulty_lidar_points=faulty_lidar_points,
                radar_points=radar_points,
            )
        faulty_lidar_bev = shared_inputs.faulty_lidar_bev
        radar_bev = shared_inputs.radar_bev
        reconstruction_mask = shared_inputs.reconstruction_mask
        healthy_context_mask = shared_inputs.healthy_context_mask
        halo_mask = shared_inputs.effective_halo
        (
            lidar_sensor_bev,
            radar_sensor_bev,
            lidar_pillar_statistics,
            radar_pillar_statistics,
        ) = self._sensor_features(
            faulty_lidar_bev,
            radar_bev,
            shared_inputs.faulty_lidar_points,
            shared_inputs.radar_points,
            radar_enabled=radar_enabled,
        )

        if not self.config.use_halo_context:
            halo_mask = torch.zeros_like(halo_mask)
        active_mask = torch.maximum(reconstruction_mask, halo_mask)
        erased_lidar_bev = (1.0 - reconstruction_mask) * lidar_sensor_bev

        crops = None
        if self.context_crop_extractor is not None:
            crops = self.context_crop_extractor.extract(
                {
                    "lidar": lidar_sensor_bev,
                    "radar": radar_sensor_bev,
                    "raw_radar": (
                        radar_bev if radar_enabled else torch.zeros_like(radar_bev)
                    ),
                    "repair": reconstruction_mask,
                    "healthy": healthy_context_mask,
                    "halo": halo_mask,
                },
                reconstruction_mask,
                halo_mask,
            )
            lidar_input = crops.tensors["lidar"]
            radar_input = crops.tensors["radar"]
            raw_radar_input = crops.tensors["raw_radar"]
            repair_input = crops.tensors["repair"]
            healthy_input = crops.tensors["healthy"]
            halo_input = crops.tensors["halo"]
            real_context = crops.valid_mask * (1.0 - repair_input)
        else:
            lidar_input = lidar_sensor_bev
            radar_input = radar_sensor_bev
            raw_radar_input = (
                radar_bev if radar_enabled else torch.zeros_like(radar_bev)
            )
            repair_input = reconstruction_mask
            healthy_input = healthy_context_mask
            halo_input = halo_mask
            real_context = 1.0 - repair_input

        if self.config.use_healthy_context_mask:
            if crops is not None:
                # The explicit context channel describes every real, observable
                # cell added around repair+halo; repair itself stays erased.
                local_context_mask = real_context
            else:
                local_context_mask = (
                    healthy_input * (1.0 - repair_input)
                    if self.config.use_halo_context
                    else torch.zeros_like(healthy_input)
                )
            mask_channels = (repair_input, local_context_mask)
        else:
            local_context_mask = real_context
            mask_channels = (repair_input,)

        local_lidar_context = real_context * lidar_input
        local_radar_context = (
            crops.valid_mask * radar_input if crops is not None else active_mask * radar_input
        )
        raw_radar_context = (
            crops.valid_mask * raw_radar_input
            if crops is not None
            else active_mask * raw_radar_input
        )
        input_streams = [local_lidar_context, local_radar_context]
        if self.config.include_raw_radar_bev:
            input_streams.append(raw_radar_context)
        input_streams.extend(mask_channels)
        local_input = torch.cat(input_streams, dim=1)
        if crops is not None:
            (
                hrnet_features,
                replacement_crop,
                hrnet_debug,
                context_bucket_shapes,
            ) = self._run_context_buckets(local_input, crops)
        else:
            hrnet_features, hrnet_debug = self.hrnet_backbone(local_input)
            replacement_crop = self.replacement_head(hrnet_features)
            context_bucket_shapes = None
        replacement_raw = (
            crops.paste(replacement_crop) if crops is not None else replacement_crop
        )
        occupancy_logits = replacement_raw[:, 0:1]
        predicted_density = replacement_raw[:, 1:2]
        predicted_height = replacement_raw[:, 2:3]
        replacement_bev = torch.cat(
            (
                torch.sigmoid(occupancy_logits),
                predicted_density,
                predicted_height,
            ),
            dim=1,
        )
        coarse_lidar_bev = (
            (1.0 - reconstruction_mask) * faulty_lidar_bev
            + reconstruction_mask * replacement_bev
        )

        outputs = {
            "erased_lidar_bev": erased_lidar_bev,
            "erased_lidar_features": erased_lidar_bev,
            "lidar_sensor_bev": lidar_sensor_bev,
            "radar_sensor_bev": radar_sensor_bev,
            "radar_raw_bev": radar_bev,
            "replacement_raw": replacement_raw,
            "replacement_bev": replacement_bev,
            "occupancy_logits": occupancy_logits,
            "predicted_density": predicted_density,
            "predicted_height": predicted_height,
            "coarse_lidar_bev": coarse_lidar_bev,
            "reconstruction_mask": reconstruction_mask,
            "healthy_context_mask": healthy_context_mask,
            "halo_mask": halo_mask,
            "active_mask": active_mask,
            "local_context_mask": local_context_mask,
            "local_lidar_context": local_lidar_context,
            "local_radar_active": local_radar_context,
            "local_radar_context": local_radar_context,
            "local_raw_radar_context": raw_radar_context,
            "local_input": local_input,
            "hrnet_features": (
                crops.paste(hrnet_features) if crops is not None else hrnet_features
            ),
            "lidar_pillar_bev": lidar_sensor_bev,
            "radar_pillar_bev": radar_sensor_bev,
            "lidar_pillar_statistics": lidar_pillar_statistics,
            "radar_pillar_statistics": radar_pillar_statistics,
        }
        if crops is not None:
            outputs.update(
                {
                    "context_crop_boxes": crops.boxes,
                    "context_source_boxes": crops.source_boxes,
                    "context_valid_mask": crops.valid_mask,
                    "context_crop_heights": crops.crop_heights,
                    "context_crop_widths": crops.crop_widths,
                    "context_padded_heights": crops.padded_heights,
                    "context_padded_widths": crops.padded_widths,
                    "context_bucket_padded_shapes": context_bucket_shapes,
                    "context_bucket_count": crops.padded_heights.new_tensor(
                        len(set(map(tuple, context_bucket_shapes.tolist())))
                    ),
                    "context_deepest_heights": torch.div(
                        crops.crop_heights + 7, 8, rounding_mode="floor"
                    ),
                    "context_deepest_widths": torch.div(
                        crops.crop_widths + 7, 8, rounding_mode="floor"
                    ),
                    "replacement_crop_raw": replacement_crop,
                    "crop_reconstruction_mask": repair_input,
                    "crop_halo_mask": halo_input,
                }
            )
        outputs.update(hrnet_debug)
        return outputs
