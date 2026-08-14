"""Physically consistent planar augmentation for aligned reconstruction samples."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import math

import torch
import torch.nn.functional as F

from .pointpillars import BEVGridGeometry


@dataclass(frozen=True)
class HorizontalFlipConfig:
    enabled: bool = True
    probability: float = 0.5


@dataclass(frozen=True)
class TranslationAugmentationConfig:
    enabled: bool = True
    max_x_m: float = 0.5
    max_y_m: float = 0.5


@dataclass(frozen=True)
class YawAugmentationConfig:
    enabled: bool = True
    max_degrees: float = 5.0


@dataclass(frozen=True)
class ScaleAugmentationConfig:
    enabled: bool = True
    min: float = 0.95
    max: float = 1.05


@dataclass(frozen=True)
class GeometricAugmentationConfig:
    enabled: bool = False
    horizontal_flip: HorizontalFlipConfig = field(
        default_factory=HorizontalFlipConfig
    )
    translation: TranslationAugmentationConfig = field(
        default_factory=TranslationAugmentationConfig
    )
    yaw: YawAugmentationConfig = field(default_factory=YawAugmentationConfig)
    scale: ScaleAugmentationConfig = field(default_factory=ScaleAugmentationConfig)

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("augmentation.enabled must be boolean")
        if not isinstance(self.horizontal_flip.enabled, bool):
            raise ValueError("augmentation.horizontal_flip.enabled must be boolean")
        numeric_values = (
            self.horizontal_flip.probability,
            self.translation.max_x_m,
            self.translation.max_y_m,
            self.yaw.max_degrees,
            self.scale.min,
            self.scale.max,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("augmentation values must be finite")
        if not 0.0 <= self.horizontal_flip.probability <= 1.0:
            raise ValueError("horizontal-flip probability must be in [0,1]")
        if not isinstance(self.translation.enabled, bool):
            raise ValueError("augmentation.translation.enabled must be boolean")
        if self.translation.max_x_m < 0.0 or self.translation.max_y_m < 0.0:
            raise ValueError("translation limits must be non-negative")
        if not isinstance(self.yaw.enabled, bool):
            raise ValueError("augmentation.yaw.enabled must be boolean")
        if self.yaw.max_degrees < 0.0:
            raise ValueError("maximum yaw must be non-negative")
        if not isinstance(self.scale.enabled, bool):
            raise ValueError("augmentation.scale.enabled must be boolean")
        if self.scale.min <= 0.0 or self.scale.max < self.scale.min:
            raise ValueError("scale bounds must satisfy 0 < min <= max")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict | None) -> "GeometricAugmentationConfig":
        payload = {} if payload is None else dict(payload)
        allowed = {field.name for field in fields(cls)}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(
                "Unknown augmentation settings: " + ", ".join(sorted(unknown))
            )

        def nested(name, config_type):
            value = payload.get(name, {})
            if not isinstance(value, dict):
                raise ValueError(f"augmentation.{name} must be an object")
            valid = {field.name for field in fields(config_type)}
            extra = set(value) - valid
            if extra:
                raise ValueError(
                    f"Unknown augmentation.{name} settings: "
                    + ", ".join(sorted(extra))
                )
            return config_type(**value)

        config = cls(
            enabled=payload.get("enabled", False),
            horizontal_flip=nested("horizontal_flip", HorizontalFlipConfig),
            translation=nested("translation", TranslationAugmentationConfig),
            yaw=nested("yaw", YawAugmentationConfig),
            scale=nested("scale", ScaleAugmentationConfig),
        )
        config.validate()
        return config


@dataclass(frozen=True)
class GeometricTransform:
    flip_y: bool = False
    scale: float = 1.0
    yaw_radians: float = 0.0
    translation_x_m: float = 0.0
    translation_y_m: float = 0.0

    @property
    def is_identity(self) -> bool:
        return (
            not self.flip_y
            and self.scale == 1.0
            and self.yaw_radians == 0.0
            and self.translation_x_m == 0.0
            and self.translation_y_m == 0.0
        )

    def to_dict(self) -> dict:
        return {
            "flip_y": self.flip_y,
            "scale": self.scale,
            "yaw_degrees": math.degrees(self.yaw_radians),
            "translation_x_m": self.translation_x_m,
            "translation_y_m": self.translation_y_m,
        }


def _uniform(low: float, high: float, generator=None) -> float:
    if low == high:
        return float(low)
    value = torch.rand((), generator=generator).item()
    return float(low + (high - low) * value)


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
        if not self.config.enabled:
            return GeometricTransform()
        flip = self.config.horizontal_flip.enabled and (
            torch.rand((), generator=generator).item()
            < self.config.horizontal_flip.probability
        )
        translation = self.config.translation
        yaw = self.config.yaw
        scale = self.config.scale
        return GeometricTransform(
            flip_y=flip,
            scale=(
                _uniform(scale.min, scale.max, generator)
                if scale.enabled
                else 1.0
            ),
            yaw_radians=(
                math.radians(
                    _uniform(-yaw.max_degrees, yaw.max_degrees, generator)
                )
                if yaw.enabled
                else 0.0
            ),
            translation_x_m=(
                _uniform(-translation.max_x_m, translation.max_x_m, generator)
                if translation.enabled
                else 0.0
            ),
            translation_y_m=(
                _uniform(-translation.max_y_m, translation.max_y_m, generator)
                if translation.enabled
                else 0.0
            ),
        )

    def _forward_matrix(self, transform: GeometricTransform, dtype, device):
        cosine = math.cos(transform.yaw_radians)
        sine = math.sin(transform.yaw_radians)
        flip = -1.0 if transform.flip_y else 1.0
        return torch.tensor(
            [
                [transform.scale * cosine, -transform.scale * sine * flip],
                [transform.scale * sine, transform.scale * cosine * flip],
            ],
            dtype=dtype,
            device=device,
        )

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
        transformed = points.clone()
        matrix = self._forward_matrix(
            transform, transformed.dtype, transformed.device
        )
        translation = transformed.new_tensor(
            [transform.translation_x_m, transform.translation_y_m]
        )
        transformed[:, :2] = transformed[:, :2] @ matrix.T + translation
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
