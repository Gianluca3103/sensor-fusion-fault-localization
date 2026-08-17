"""Configuration for the VoD PointPillars + HRNet coarse reconstructor."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path

from .coarse_loss import (
    CoarseLossConfig,
    ObservabilityWeightingConfig,
)
from .hrnet_backbone import HRNetConfig
from ..fault_selector import FaultSelectorConfig
from ..geometric_augmentation import GeometricAugmentationConfig
from ..pointpillars import PointPillarsConfig


@dataclass(frozen=True)
class CoarseReconstructionConfig:
    """Configuration for the coarse HRNet reconstruction model.

    PointPillars may be disabled only to load a direct-BEV HRNet checkpoint used
    by the untouched diffusion pipeline. New coarse experiments should enable it.
    """

    lidar_channels: int = 3
    radar_channels: int = 4
    target_lidar_channels: int = 3
    use_healthy_context_mask: bool = True
    use_halo_context: bool = True
    pointpillars: PointPillarsConfig = field(default_factory=PointPillarsConfig)
    hrnet: HRNetConfig = field(default_factory=HRNetConfig)

    @property
    def local_input_channels(self) -> int:
        mask_channels = 2 if self.use_healthy_context_mask else 1
        return self.lidar_channels + self.radar_channels + mask_channels

    def validate(self) -> None:
        for name in ("lidar_channels", "radar_channels", "target_lidar_channels"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.target_lidar_channels != 3:
            raise ValueError(
                "The coarse target must contain occupancy, density, and height"
            )
        if not isinstance(self.use_healthy_context_mask, bool):
            raise ValueError("use_healthy_context_mask must be boolean")
        if not isinstance(self.use_halo_context, bool):
            raise ValueError("use_halo_context must be boolean")
        self.pointpillars.validate()
        self.hrnet.validate()
        if self.pointpillars.enabled:
            expected = self.pointpillars.output_channels
            if self.lidar_channels != expected or self.radar_channels != expected:
                raise ValueError(
                    "PointPillars output_channels must match both sensor channel counts"
                )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "CoarseReconstructionConfig":
        """Load serialized HRNet settings used by saved checkpoints."""

        values = dict(payload)
        pointpillars = values.get("pointpillars", {})
        if isinstance(pointpillars, dict):
            values["pointpillars"] = PointPillarsConfig(**pointpillars)
        hrnet = values.get("hrnet", {})
        if isinstance(hrnet, dict):
            hrnet_fields = {item.name for item in fields(HRNetConfig)}
            values["hrnet"] = HRNetConfig(
                **{key: value for key, value in hrnet.items() if key in hrnet_fields}
            )
        supported = {item.name for item in fields(cls)}
        config = cls(**{key: value for key, value in values.items() if key in supported})
        config.validate()
        return config


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError("Coarse configuration must be a JSON file")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Coarse configuration must decode to an object")
    return payload


def _checked_dataclass(section: str, payload: object, cls):
    if not isinstance(payload, dict):
        raise ValueError(f"{section} must be an object")
    valid = {item.name for item in fields(cls)}
    unknown = set(payload) - valid
    if unknown:
        raise ValueError(
            f"Unknown {section} settings: " + ", ".join(sorted(unknown))
        )
    return cls(**payload)


def build_selector_config(payload: dict) -> FaultSelectorConfig:
    config = _checked_dataclass(
        "fault_selector", payload.get("fault_selector", {}), FaultSelectorConfig
    )
    config.validate()
    return config


def build_augmentation_config(payload: dict) -> GeometricAugmentationConfig:
    return GeometricAugmentationConfig.from_dict(payload.get("augmentation", {}))


def build_configs(payload: dict):
    """Build the VoD coarse HRNet model, its loss, and selector."""

    coarse = payload.get("coarse_reconstruction", {})
    if not isinstance(coarse, dict):
        raise ValueError("coarse_reconstruction must be an object")
    if not coarse.get("enabled", True):
        raise ValueError("coarse_reconstruction.enabled must be true")

    pointpillars = _checked_dataclass(
        "pointpillars", payload.get("pointpillars", {}), PointPillarsConfig
    )
    hrnet = _checked_dataclass("hrnet", payload.get("hrnet", {}), HRNetConfig)

    mask_payload = payload.get("masks", {})
    if not isinstance(mask_payload, dict):
        raise ValueError("masks must be an object")
    unknown_masks = set(mask_payload) - {
        "use_healthy_context_mask",
        "use_halo_context",
    }
    if unknown_masks:
        raise ValueError("Unknown masks settings: " + ", ".join(sorted(unknown_masks)))

    lidar_channels = (
        pointpillars.output_channels
        if pointpillars.enabled
        else 3
    )
    radar_channels = (
        pointpillars.output_channels
        if pointpillars.enabled
        else 4
    )
    model_config = CoarseReconstructionConfig(
        lidar_channels=lidar_channels,
        radar_channels=radar_channels,
        use_healthy_context_mask=mask_payload.get("use_healthy_context_mask", True),
        use_halo_context=mask_payload.get("use_halo_context", True),
        pointpillars=pointpillars,
        hrnet=hrnet,
    )

    loss_payload = coarse.get("loss", {})
    if not isinstance(loss_payload, dict):
        raise ValueError("coarse_reconstruction.loss must be an object")
    allowed_loss = {
        "lambda_occupancy",
        "lambda_density",
        "lambda_height",
        "epsilon",
        "positive_occupancy_weight",
        "observability_weighting",
    }
    unknown_loss = set(loss_payload) - allowed_loss
    if unknown_loss:
        raise ValueError(
            "Unknown or obsolete coarse loss settings: "
            + ", ".join(sorted(unknown_loss))
        )
    observability = _checked_dataclass(
        "observability_weighting",
        loss_payload.get("observability_weighting", {}),
        ObservabilityWeightingConfig,
    )
    loss_config = CoarseLossConfig(
        lambda_occupancy=loss_payload.get("lambda_occupancy", 1.0),
        lambda_density=loss_payload.get("lambda_density", 1.0),
        lambda_height=loss_payload.get("lambda_height", 1.0),
        epsilon=loss_payload.get("epsilon", 1.0e-8),
        positive_occupancy_weight=loss_payload.get(
            "positive_occupancy_weight", 1.0
        ),
        observability_weighting=observability,
    )
    selector_config = build_selector_config(payload)
    model_config.validate()
    loss_config.validate()
    return model_config, loss_config, selector_config
