"""Direct faulty-LiDAR/radar PointPillars fusion for VoD detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from collections.abc import Sequence

import torch
from torch import nn

from models.two_stage_reconstruction_head.coarse_reconstruction.hrnet_backbone import (
    HRNetBackbone,
    HRNetConfig,
)
from models.two_stage_reconstruction_head.encoders import _group_count
from models.two_stage_reconstruction_head.pointpillars import (
    BEVGridGeometry,
    PointPillarsConfig,
    PointPillarsEncoder,
)

from .detector import BEVDetectorConfig


def _nested_dataclass(payload: object, cls, section: str):
    if not isinstance(payload, dict):
        raise ValueError(f"{section} must be an object")
    valid = {item.name for item in fields(cls)}
    unknown = set(payload) - valid
    if unknown:
        raise ValueError(
            f"Unknown {section} settings: " + ", ".join(sorted(unknown))
        )
    return cls(**payload)


@dataclass(frozen=True)
class FusionDetectorConfig:
    """Architecture configuration for direct multimodal BEV detection."""

    pointpillars: PointPillarsConfig = field(
        default_factory=lambda: PointPillarsConfig(
            enabled=True,
            output_channels=32,
            max_points_per_pillar=100,
            max_pillars=None,
        )
    )
    hrnet: HRNetConfig = field(
        default_factory=lambda: HRNetConfig(base_channels=32, dropout=0.2)
    )
    detector: BEVDetectorConfig = field(
        default_factory=lambda: BEVDetectorConfig(
            input_channels=32,
            base_channels=64,
            output_stride=2,
            box_regression_channels=8,
        )
    )

    def validate(self) -> None:
        self.pointpillars.validate()
        self.hrnet.validate()
        self.detector.validate()
        if not self.pointpillars.enabled:
            raise ValueError("fusion detector requires PointPillars")
        if self.detector.input_channels != 32:
            raise ValueError(
                "fusion detector detection input_channels must match HRNet's "
                "32 output channels"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "FusionDetectorConfig":
        if not isinstance(payload, dict):
            raise ValueError("fusion_detector must be an object")
        valid = {item.name for item in fields(cls)}
        unknown = set(payload) - valid
        if unknown:
            raise ValueError(
                "Unknown fusion_detector settings: "
                + ", ".join(sorted(unknown))
            )
        defaults = cls()
        config = cls(
            pointpillars=_nested_dataclass(
                payload.get("pointpillars", defaults.pointpillars.to_dict()),
                PointPillarsConfig,
                "fusion_detector.pointpillars",
            ),
            hrnet=_nested_dataclass(
                payload.get("hrnet", defaults.hrnet.to_dict()),
                HRNetConfig,
                "fusion_detector.hrnet",
            ),
            detector=_nested_dataclass(
                payload.get("detector", defaults.detector.to_dict()),
                BEVDetectorConfig,
                "fusion_detector.detector",
            ),
        )
        config.validate()
        return config


class AnchorFreeCenterHead(nn.Module):
    """Center heatmaps and metric 3D box regression on dense BEV features."""

    def __init__(
        self,
        in_channels: int,
        class_count: int,
        config: BEVDetectorConfig,
    ) -> None:
        super().__init__()
        if class_count < 1:
            raise ValueError("class_count must be positive")
        channels = config.base_channels
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(inplace=True),
        ]
        stride = 1
        while stride < config.output_stride:
            layers.extend(
                (
                    nn.Conv2d(
                        channels,
                        channels,
                        3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(_group_count(channels), channels),
                    nn.SiLU(inplace=True),
                )
            )
            stride *= 2
        self.shared = nn.Sequential(*layers)
        self.heatmap = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, class_count, 1),
        )
        self.box = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, config.box_regression_channels, 1),
        )
        nn.init.constant_(self.heatmap[-1].bias, -2.19)

    def forward(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.shared(tensor)
        return {
            "heatmap_logits": self.heatmap(features),
            "box_regression": self.box(features),
        }


class PointPillarsHRNetFusionDetector(nn.Module):
    """Detect objects directly from faulty LiDAR and stacked radar points.

    The two sensors have independent PointPillars weights. Their dense feature
    maps share one geometry and are concatenated before HRNet. No clean LiDAR,
    reconstruction mask, fault selector, or reconstructed pseudo-points enter
    the inference path.
    """

    def __init__(
        self,
        class_names: tuple[str, ...],
        geometry: BEVGridGeometry,
        config: FusionDetectorConfig | None = None,
    ) -> None:
        super().__init__()
        self.class_names = tuple(class_names)
        if not self.class_names:
            raise ValueError("At least one detector class is required")
        geometry.validate()
        self.geometry = geometry
        self.config = config or FusionDetectorConfig()
        self.config.validate()
        pillars = self.config.pointpillars
        common = {
            "output_channels": pillars.output_channels,
            "max_points_per_pillar": pillars.max_points_per_pillar,
            "max_pillars": pillars.max_pillars,
        }
        self.lidar_pillar_encoder = PointPillarsEncoder(
            geometry,
            raw_channels=pillars.lidar_raw_channels,
            **common,
        )
        self.radar_pillar_encoder = PointPillarsEncoder(
            geometry,
            raw_channels=pillars.radar_raw_channels,
            **common,
        )
        self.fusion_backbone = HRNetBackbone(
            2 * pillars.output_channels,
            self.config.hrnet,
        )
        self.detection_head = AnchorFreeCenterHead(
            self.fusion_backbone.out_channels,
            len(self.class_names),
            self.config.detector,
        )

    def _lidar_fields(
        self, point_clouds: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        selected = []
        for points in point_clouds:
            if points.ndim != 2 or points.shape[1] != 4:
                raise ValueError(
                    "faulty_lidar_points must contain [x,y,z,reflectivity] rows"
                )
            selected.append(
                points
                if self.config.pointpillars.lidar_use_reflectivity
                else points[:, :3]
            )
        return tuple(selected)

    def _radar_fields(
        self, point_clouds: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        selected = []
        for points in point_clouds:
            if points.ndim != 2 or points.shape[1] != 5:
                raise ValueError(
                    "radar_points must contain [x,y,z,power,doppler] rows"
                )
            columns = [points[:, :3]]
            if self.config.pointpillars.radar_use_power:
                columns.append(points[:, 3:4])
            if self.config.pointpillars.radar_use_radial_velocity:
                columns.append(points[:, 4:5])
            selected.append(torch.cat(columns, dim=1))
        return tuple(selected)

    def forward(
        self,
        faulty_lidar_points: Sequence[torch.Tensor],
        radar_points: Sequence[torch.Tensor],
        *,
        radar_enabled: bool = True,
        return_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor | dict]:
        lidar_features, lidar_statistics = self.lidar_pillar_encoder(
            self._lidar_fields(faulty_lidar_points)
        )
        if radar_enabled:
            radar_features, radar_statistics = self.radar_pillar_encoder(
                self._radar_fields(radar_points)
            )
        else:
            radar_features = torch.zeros_like(lidar_features)
            radar_statistics = {}
        if lidar_features.shape[-2:] != radar_features.shape[-2:]:
            raise RuntimeError("LiDAR and radar PointPillars grids are not aligned")
        concatenated = torch.cat((lidar_features, radar_features), dim=1)
        fused_features, hrnet_debug = self.fusion_backbone(concatenated)
        outputs = self.detection_head(fused_features)
        if return_diagnostics:
            outputs["diagnostics"] = {
                "lidar_pillar_features": lidar_features,
                "radar_pillar_features": radar_features,
                "concatenated_features": concatenated,
                "fused_features": fused_features,
                "lidar_statistics": lidar_statistics,
                "radar_statistics": radar_statistics,
                "hrnet": hrnet_debug,
            }
        return outputs
