"""Physically consistent planar augmentation for aligned reconstruction samples."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from models.shared_geometric_augmentation import (
    GeometricAugmentationConfig, GeometricTransform, HorizontalFlipConfig,
    ScaleAugmentationConfig, TranslationAugmentationConfig,
    YawAugmentationConfig, forward_xy_matrix, sample_transform,
    transform_points_xy,
)

from .pointpillars import BEVGridGeometry


class ReconstructionGeometricAugmentation:
    """Apply one shared ego-centric XY transform to every sample modality."""

    def __init__(
        self,
        config: GeometricAugmentationConfig,
        geometry: BEVGridGeometry,
    ):
        config.validate()
        geometry.validate()
        self.config = config
        self.geometry = geometry

    def sample_transform(self, *, generator=None) -> GeometricTransform:
        return sample_transform(self.config, generator=generator)

    def _forward_matrix(self, transform: GeometricTransform, dtype, device):
        return forward_xy_matrix(transform, dtype, device)

    def _sampling_grid(self, transform: GeometricTransform, dtype, device):
        geometry = self.geometry
        rows = torch.arange(geometry.height, dtype=dtype, device=device)
        columns = torch.arange(geometry.width, dtype=dtype, device=device)
        x = geometry.x_max - (rows + 0.5) * geometry.pillar_size_x
        y = geometry.y_min + (columns + 0.5) * geometry.pillar_size_y
        destination_x, destination_y = torch.meshgrid(x, y, indexing="ij")
        destination = torch.stack((destination_x, destination_y), dim=-1)
        translation = torch.tensor(
            [transform.translation_x_m, transform.translation_y_m],
            dtype=dtype,
            device=device,
        )
        inverse = torch.linalg.inv(
            self._forward_matrix(transform, dtype, device)
        )
        source = (destination - translation) @ inverse.T
        grid_y = 2.0 * (source[..., 1] - geometry.y_min) / (
            geometry.y_max - geometry.y_min
        ) - 1.0
        grid_x = 1.0 - 2.0 * (source[..., 0] - geometry.x_min) / (
            geometry.x_max - geometry.x_min
        )
        return torch.stack((grid_y, grid_x), dim=-1).unsqueeze(0)

    def warp(
        self,
        tensor: torch.Tensor,
        transform: GeometricTransform,
        *,
        mode: str,
        fill_value: float = 0.0,
        grid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if transform.is_identity:
            return tensor
        squeeze_channel = tensor.ndim == 2
        if squeeze_channel:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3 or tensor.shape[-2:] != (
            self.geometry.height,
            self.geometry.width,
        ):
            raise ValueError(
                "Spatial augmentation expects [C,H,W] aligned to the dataset grid; "
                f"got {tuple(tensor.shape)}"
            )
        original_dtype = tensor.dtype
        source = tensor.to(dtype=torch.float32)
        if fill_value:
            source = source - fill_value
        result = F.grid_sample(
            source.unsqueeze(0),
            (
                grid
                if grid is not None
                else self._sampling_grid(transform, source.dtype, source.device)
            ),
            mode=mode,
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(0)
        if fill_value:
            result = result + fill_value
        result = result.to(
            dtype=original_dtype if original_dtype.is_floating_point else torch.float32
        )
        return result.squeeze(0) if squeeze_channel else result

    def _warp_lidar_bev(
        self,
        tensor: torch.Tensor,
        transform: GeometricTransform,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        occupancy = self.warp(
            tensor[0:1], transform, mode="nearest", grid=grid
        )
        continuous = self.warp(
            tensor[1:3], transform, mode="bilinear", grid=grid
        )
        occupancy = (occupancy >= 0.5).to(dtype=continuous.dtype)
        return torch.cat((occupancy, continuous * occupancy), dim=0)

    def _warp_radar_bev(
        self,
        tensor: torch.Tensor,
        transform: GeometricTransform,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        occupancy = self.warp(
            tensor[0:1], transform, mode="nearest", grid=grid
        )
        continuous = self.warp(
            tensor[1:], transform, mode="bilinear", grid=grid
        )
        occupancy = (occupancy >= 0.5).to(continuous.dtype)
        return torch.cat((occupancy, continuous * occupancy), dim=0)

    def _warp_binary(
        self,
        tensor: torch.Tensor,
        transform: GeometricTransform,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        dtype = tensor.dtype
        warped = self.warp(
            tensor, transform, mode="nearest", grid=grid
        ) >= 0.5
        return warped.to(dtype=dtype)

    def _transform_points(
        self, points: torch.Tensor, transform: GeometricTransform
    ) -> torch.Tensor:
        if transform.is_identity:
            return points
        transformed = transform_points_xy(points, transform)
        geometry = self.geometry
        inside = (
            (transformed[:, 0] >= geometry.x_min)
            & (transformed[:, 0] < geometry.x_max)
            & (transformed[:, 1] >= geometry.y_min)
            & (transformed[:, 1] < geometry.y_max)
        )
        return transformed[inside]

    def apply(
        self,
        item: dict[str, object],
        *,
        transform: GeometricTransform | None = None,
        generator=None,
    ) -> dict[str, object]:
        transform = transform or self.sample_transform(generator=generator)
        if transform.is_identity:
            return item
        output = dict(item)
        reference = output["faulty_bev"]
        grid = self._sampling_grid(
            transform,
            torch.float32,
            reference.device,
        )
        for key in ("clean_bev", "faulty_bev"):
            if key in output:
                output[key] = self._warp_lidar_bev(
                    output[key], transform, grid
                )
        if "lidar_input_bev" in output:
            output["lidar_input_bev"] = self._warp_radar_bev(
                output["lidar_input_bev"], transform, grid
            )
        if "radar_bev" in output:
            output["radar_bev"] = self._warp_radar_bev(
                output["radar_bev"], transform, grid
            )
        for key in (
            "reconstruction_mask",
            "halo_mask",
            "healthy_context_mask",
            "fault_heatmap",
            "valid_support_mask",
        ):
            if key in output:
                output[key] = self._warp_binary(output[key], transform, grid)
        for key in (
            "observability_confidence",
            "observability_ray_count",
            "observability_vertical_coverage",
            "observability_ray_support",
        ):
            if key in output:
                output[key] = self.warp(
                    output[key], transform, mode="bilinear", grid=grid
                )
        if "reliability_map" in output:
            output["reliability_map"] = self.warp(
                output["reliability_map"],
                transform,
                mode="bilinear",
                fill_value=1.0,
                grid=grid,
            )
        for key in (
            "faulty_counts",
            "added_faulty_counts",
            "missing_faulty_counts",
            "moved_faulty_counts",
        ):
            if key in output:
                output[key] = self.warp(
                    output[key], transform, mode="nearest", grid=grid
                )
        for key in ("faulty_lidar_points", "radar_points"):
            if key in output:
                output[key] = self._transform_points(output[key], transform)
        # Supervision must follow the same spatial transform but never enter
        # sensor conditioning or PointPillars feature extraction.
        if 'geometric_reference_points' in output:
            output['geometric_reference_points'] = self._transform_points(
                output['geometric_reference_points'], transform)

        repair = output.get("reconstruction_mask")
        halo = output.get("halo_mask")
        context = output.get("healthy_context_mask")
        if repair is not None and halo is not None:
            output["halo_mask"] = halo * (1 - repair)
        if repair is not None and context is not None:
            context = context * (1 - repair)
            if "halo_mask" in output:
                context = context * output["halo_mask"]
            output["healthy_context_mask"] = context
        output["augmentation_transform"] = transform.to_dict()
        return output
