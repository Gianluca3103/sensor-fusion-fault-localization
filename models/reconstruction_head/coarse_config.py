"""Configuration loading and validation for direct-BEV coarse reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

from .coarse_loss import CoarseLossConfig
from .fault_selector import FaultSelectorConfig


@dataclass(frozen=True)
class CoarseReconstructionConfig:
    lidar_channels: int = 3
    radar_channels: int = 4
    unet_base_channels: int = 16
    unet_depth: int = 5
    dropout: float = 0.0
    global_base_channels: int = 16
    global_channel_multipliers: tuple[int, ...] = (1, 2, 4, 8, 16)
    attention_dim: int = 128
    num_heads: int = 4
    attention_dropout: float = 0.0

    @property
    def local_input_channels(self) -> int:
        return self.lidar_channels + self.radar_channels + 2

    def validate(self) -> None:
        integer_values = {
            "lidar_channels": self.lidar_channels,
            "radar_channels": self.radar_channels,
            "unet_base_channels": self.unet_base_channels,
            "unet_depth": self.unet_depth,
            "global_base_channels": self.global_base_channels,
            "attention_dim": self.attention_dim,
            "num_heads": self.num_heads,
        }
        for name, value in integer_values.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if not self.global_channel_multipliers or any(
            value < 1 for value in self.global_channel_multipliers
        ):
            raise ValueError("global_channel_multipliers must contain positive values")
        if self.attention_dim % self.num_heads:
            raise ValueError("attention_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0,1)")

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = json.load(handle)
        elif suffix in {".yaml", ".yml"}:
            import yaml

            payload = yaml.safe_load(handle)
        else:
            raise ValueError(f"Unsupported configuration format: {suffix}")
    if not isinstance(payload, dict):
        raise ValueError("Coarse configuration must decode to a mapping")
    return payload


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
    model_config = CoarseReconstructionConfig(
        unet_base_channels=unet.get(
            "base_channels", defaults.unet_base_channels
        ),
        unet_depth=unet.get("depth", defaults.unet_depth),
        dropout=unet.get("dropout", defaults.dropout),
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
    )
    loss_payload = coarse.get("loss", {})
    loss_config = CoarseLossConfig(
        reconstruction_loss_type=loss_payload.get(
            "reconstruction_loss_type", "smooth_l1"
        ),
        lambda_reconstruction=loss_payload.get("lambda_reconstruction", 1.0),
        epsilon=loss_payload.get("epsilon", 1.0e-8),
    )
    selector_payload = payload.get("fault_selector", {})
    valid_selector_fields = {field.name for field in fields(FaultSelectorConfig)}
    unknown = set(selector_payload) - valid_selector_fields
    if unknown:
        raise ValueError("Unknown fault_selector settings: " + ", ".join(sorted(unknown)))
    selector_config = FaultSelectorConfig(**selector_payload)
    model_config.validate()
    loss_config.validate()
    selector_config.validate()
    return model_config, loss_config, selector_config
