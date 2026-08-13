"""Configuration loading and validation for direct-BEV coarse reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path

from .coarse_loss import CoarseLossConfig, ObservabilityWeightingConfig
from ..fault_selector import FaultSelectorConfig
from ..pointpillars import PointPillarsConfig
from .sst_backbone import SSTConfig
from .repair_query import RepairQueryConfig
from .hrnet_backbone import HRNetConfig
from .range_aware_radar import RangeAwareRadarConfig


@dataclass(frozen=True)
class CoarseReconstructionConfig:
    backbone: str = "unet"
    lidar_channels: int = 3
    radar_channels: int = 4
    target_lidar_channels: int = 3
    unet_base_channels: int = 16
    unet_depth: int = 4
    dropout: float = 0.025
    use_healthy_context_mask: bool = True
    global_base_channels: int = 16
    global_channel_multipliers: tuple[int, ...] = (1, 2, 4, 8, 16)
    attention_dim: int = 128
    num_heads: int = 4
    attention_dropout: float = 0.025
    pointpillars: PointPillarsConfig = field(default_factory=PointPillarsConfig)
    sst: SSTConfig = field(default_factory=SSTConfig)
    repair_query: RepairQueryConfig = field(default_factory=RepairQueryConfig)
    hrnet: HRNetConfig = field(default_factory=HRNetConfig)
    range_aware_radar: RangeAwareRadarConfig = field(
        default_factory=RangeAwareRadarConfig
    )

    @property
    def local_input_channels(self) -> int:
        mask_channels = 2 if self.use_healthy_context_mask else 1
        radar_channels = (
            self.range_aware_radar.output_channels
            if self.range_aware_radar.enabled
            else self.radar_channels
        )
        return self.lidar_channels + radar_channels + mask_channels

    def validate(self) -> None:
        if self.backbone not in {"unet", "sst", "repair_query", "hrnet"}:
            raise ValueError(
                "model.backbone must be 'unet', 'sst', 'repair_query', or 'hrnet'"
            )
        integer_values = {
            "lidar_channels": self.lidar_channels,
            "radar_channels": self.radar_channels,
            "target_lidar_channels": self.target_lidar_channels,
            "unet_base_channels": self.unet_base_channels,
            "unet_depth": self.unet_depth,
            "global_base_channels": self.global_base_channels,
            "attention_dim": self.attention_dim,
            "num_heads": self.num_heads,
        }
        for name, value in integer_values.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.target_lidar_channels != 3:
            raise ValueError(
                "The coarse reconstruction target remains the existing "
                "three-channel LiDAR BEV"
            )
        if not self.global_channel_multipliers or any(
            value < 1 for value in self.global_channel_multipliers
        ):
            raise ValueError("global_channel_multipliers must contain positive values")
        if self.attention_dim % self.num_heads:
            raise ValueError("attention_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if not isinstance(self.use_healthy_context_mask, bool):
            raise ValueError("use_healthy_context_mask must be boolean")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0,1)")
        self.pointpillars.validate()
        self.sst.validate()
        self.repair_query.validate()
        self.hrnet.validate()
        self.range_aware_radar.validate()
        if self.range_aware_radar.enabled and self.backbone != "hrnet":
            raise ValueError(
                "range-aware Radar aggregation is isolated to the HRNet backbone"
            )
        if self.range_aware_radar.enabled and self.pointpillars.enabled:
            raise ValueError(
                "range-aware Radar aggregation requires handcrafted Radar BEVs"
            )
        if (
            self.range_aware_radar.enabled
            and self.hrnet.radar_context_layers
        ):
            raise ValueError(
                "Use either fixed RF Radar context or range-aware Radar, not both"
            )
        if (
            self.backbone in {"sst", "repair_query"}
            and not self.pointpillars.enabled
        ):
            raise ValueError(
                f"The {self.backbone} backbone requires PointPillars inputs"
            )
        expected_lidar = (
            self.pointpillars.output_channels
            if self.pointpillars.enabled
            else 3
        )
        expected_radar = (
            self.pointpillars.output_channels
            if self.pointpillars.enabled
            else 4
        )
        if self.lidar_channels != expected_lidar:
            raise ValueError(
                "lidar_channels does not match the selected sensor representation: "
                f"expected {expected_lidar}, got {self.lidar_channels}"
            )
        if self.radar_channels != expected_radar:
            raise ValueError(
                "radar_channels does not match the selected sensor representation: "
                f"expected {expected_radar}, got {self.radar_channels}"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "CoarseReconstructionConfig":
        values = dict(payload)
        if "global_channel_multipliers" in values:
            values["global_channel_multipliers"] = tuple(
                values["global_channel_multipliers"]
            )
        pointpillars = values.get("pointpillars", {})
        if isinstance(pointpillars, dict):
            values["pointpillars"] = PointPillarsConfig(**pointpillars)
        sst = values.get("sst", {})
        if isinstance(sst, dict):
            values["sst"] = SSTConfig(**sst)
        repair_query = values.get("repair_query", {})
        if isinstance(repair_query, dict):
            values["repair_query"] = RepairQueryConfig(**repair_query)
        hrnet = values.get("hrnet", {})
        if isinstance(hrnet, dict):
            values["hrnet"] = HRNetConfig(**hrnet)
        range_aware_radar = values.get("range_aware_radar", {})
        if isinstance(range_aware_radar, dict):
            values["range_aware_radar"] = RangeAwareRadarConfig(
                **range_aware_radar
            )
        return cls(**values)


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError("Coarse configuration must be a JSON file")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Coarse configuration must decode to a mapping")
    return payload


def build_selector_config(payload: dict) -> FaultSelectorConfig:
    selector_payload = payload.get("fault_selector", {})
    valid_fields = {field.name for field in fields(FaultSelectorConfig)}
    unknown = set(selector_payload) - valid_fields
    if unknown:
        raise ValueError("Unknown fault_selector settings: " + ", ".join(sorted(unknown)))
    config = FaultSelectorConfig(**selector_payload)
    config.validate()
    return config


def build_configs(payload: dict):
    coarse = payload.get("coarse_reconstruction", {})
    unet = coarse.get("unet", {})
    global_context = coarse.get("global_context", {})
    if not coarse.get("enabled", True):
        raise ValueError("coarse_reconstruction.enabled must be true for this trainer")
    if unet.get("normalization", "group_norm") != "group_norm":
        raise ValueError("The coarse U-Net currently supports group_norm only")
    if unet.get("activation", "silu") != "silu":
        raise ValueError("The coarse U-Net currently supports silu activation only")
    defaults = CoarseReconstructionConfig()
    model_payload = payload.get("model", {})
    if not isinstance(model_payload, dict):
        raise ValueError("model must be an object")
    unknown_model_fields = set(model_payload) - {"backbone"}
    if unknown_model_fields:
        raise ValueError(
            "Unknown model settings: " + ", ".join(sorted(unknown_model_fields))
        )
    backbone = model_payload.get("backbone", defaults.backbone)
    sst_payload = payload.get("sst", {})
    if not isinstance(sst_payload, dict):
        raise ValueError("sst must be an object")
    valid_sst_fields = {field.name for field in fields(SSTConfig)}
    unknown_sst_fields = set(sst_payload) - valid_sst_fields
    if unknown_sst_fields:
        raise ValueError(
            "Unknown sst settings: " + ", ".join(sorted(unknown_sst_fields))
        )
    sst = SSTConfig(**sst_payload)
    repair_query_payload = payload.get("repair_query", {})
    if not isinstance(repair_query_payload, dict):
        raise ValueError("repair_query must be an object")
    valid_repair_query_fields = {
        field.name for field in fields(RepairQueryConfig)
    }
    unknown_repair_query_fields = (
        set(repair_query_payload) - valid_repair_query_fields
    )
    if unknown_repair_query_fields:
        raise ValueError(
            "Unknown repair_query settings: "
            + ", ".join(sorted(unknown_repair_query_fields))
        )
    repair_query = RepairQueryConfig(**repair_query_payload)
    hrnet_payload = payload.get("hrnet", {})
    if not isinstance(hrnet_payload, dict):
        raise ValueError("hrnet must be an object")
    valid_hrnet_fields = {field.name for field in fields(HRNetConfig)}
    unknown_hrnet_fields = set(hrnet_payload) - valid_hrnet_fields
    if unknown_hrnet_fields:
        raise ValueError(
            "Unknown hrnet settings: "
            + ", ".join(sorted(unknown_hrnet_fields))
        )
    hrnet = HRNetConfig(**hrnet_payload)
    range_aware_payload = payload.get("range_aware_radar", {})
    if not isinstance(range_aware_payload, dict):
        raise ValueError("range_aware_radar must be an object")
    valid_range_aware_fields = {
        field.name for field in fields(RangeAwareRadarConfig)
    }
    unknown_range_aware_fields = (
        set(range_aware_payload) - valid_range_aware_fields
    )
    if unknown_range_aware_fields:
        raise ValueError(
            "Unknown range_aware_radar settings: "
            + ", ".join(sorted(unknown_range_aware_fields))
        )
    range_aware_radar = RangeAwareRadarConfig(**range_aware_payload)
    pointpillars_payload = payload.get("pointpillars", {})
    if not isinstance(pointpillars_payload, dict):
        raise ValueError("pointpillars must be an object")
    valid_pointpillars_fields = {
        field.name for field in fields(PointPillarsConfig)
    }
    unknown_pointpillars_fields = (
        set(pointpillars_payload) - valid_pointpillars_fields
    )
    if unknown_pointpillars_fields:
        raise ValueError(
            "Unknown pointpillars settings: "
            + ", ".join(sorted(unknown_pointpillars_fields))
        )
    pointpillars = PointPillarsConfig(**pointpillars_payload)
    pointpillars.validate()
    lidar_channels = pointpillars.output_channels if pointpillars.enabled else 3
    radar_channels = pointpillars.output_channels if pointpillars.enabled else 4
    model_config = CoarseReconstructionConfig(
        backbone=backbone,
        lidar_channels=lidar_channels,
        radar_channels=radar_channels,
        unet_base_channels=unet.get(
            "base_channels", defaults.unet_base_channels
        ),
        unet_depth=unet.get("depth", defaults.unet_depth),
        dropout=unet.get("dropout", defaults.dropout),
        use_healthy_context_mask=unet.get(
            "use_healthy_context_mask", defaults.use_healthy_context_mask
        ),
        global_base_channels=global_context.get(
            "base_channels", defaults.global_base_channels
        ),
        global_channel_multipliers=tuple(
            global_context.get(
                "channel_multipliers", defaults.global_channel_multipliers
            )
        ),
        attention_dim=global_context.get(
            "attention_dim", defaults.attention_dim
        ),
        num_heads=global_context.get("num_heads", defaults.num_heads),
        attention_dropout=global_context.get(
            "attention_dropout", defaults.attention_dropout
        ),
        pointpillars=pointpillars,
        sst=sst,
        repair_query=repair_query,
        hrnet=hrnet,
        range_aware_radar=range_aware_radar,
    )
    loss_payload = coarse.get("loss", {})
    allowed_loss_fields = {
        "lambda_occupancy",
        "lambda_density",
        "lambda_height",
        "epsilon",
        "observability_weighting",
    }
    unknown_loss_fields = set(loss_payload) - allowed_loss_fields
    if unknown_loss_fields:
        raise ValueError(
            "Unknown or obsolete coarse loss settings: "
            + ", ".join(sorted(unknown_loss_fields))
        )
    observability_payload = loss_payload.get("observability_weighting", {})
    if not isinstance(observability_payload, dict):
        raise ValueError("coarse loss observability_weighting must be an object")
    unknown_observability_fields = set(observability_payload) - {
        "enabled",
        "min_empty_weight",
    }
    if unknown_observability_fields:
        raise ValueError(
            "Unknown observability_weighting settings: "
            + ", ".join(sorted(unknown_observability_fields))
        )
    loss_config = CoarseLossConfig(
        lambda_occupancy=loss_payload.get("lambda_occupancy", 1.0),
        lambda_density=loss_payload.get("lambda_density", 1.0),
        lambda_height=loss_payload.get("lambda_height", 1.0),
        epsilon=loss_payload.get("epsilon", 1.0e-8),
        observability_weighting=ObservabilityWeightingConfig(
            enabled=observability_payload.get("enabled", False),
            min_empty_weight=observability_payload.get(
                "min_empty_weight", 0.1
            ),
        ),
    )
    selector_config = build_selector_config(payload)
    model_config.validate()
    loss_config.validate()
    return model_config, loss_config, selector_config
