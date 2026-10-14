"""Validated settings for the bottleneck-only Stage-II fusion experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from models.radar_lidar_stage2_residual.config import ResidualStage2Config


@dataclass(frozen=True)
class BottleneckDiffusionConfig:
    residual: ResidualStage2Config = ResidualStage2Config()
    architecture: str = "bottleneck_diffusion_v2"
    channels: tuple[int, ...] = (24, 48, 96, 128)
    spatial_strides: tuple[int, ...] = (2, 2, 2)
    blocks_per_level: int = 1
    timestep_dim: int = 128
    fusion_mode: str = "cross_attention"  # or concat
    attention_heads: int = 4
    attention_radius_m: float = 6.0
    attention_max_neighbors: int = 16
    timesteps: int = 50
    beta_schedule: str = "cosine"  # or linear
    sampling_method: str = "ddpm"  # or ddim
    ddim_eta: float = 0.0
    diffusion_weight: float = 1.0
    occupancy_weight: float = 0.25
    geometry_weight: float = 0.25
    preservation_weight: float = 0.25
    sample_seed: int = 1701

    def __post_init__(self) -> None:
        if self.architecture != "bottleneck_diffusion_v2":
            raise ValueError("Expected the bottleneck Stage-II architecture")
        if len(self.channels) < 2 or any(ch < 4 for ch in self.channels):
            raise ValueError("The sparse U-Net needs at least two positive channel levels")
        if self.channels[0] != self.residual.channels[0]:
            raise ValueError("The U-Net input width must match the existing fine conditioner")
        if len(self.spatial_strides) != len(self.channels) - 1 or any(s < 2 for s in self.spatial_strides):
            raise ValueError("Specify one downsampling stride per U-Net transition")
        if self.blocks_per_level < 1 or self.timestep_dim < 8 or self.timestep_dim % 2:
            raise ValueError("Invalid residual-block count or timestep width")
        if self.fusion_mode not in {"cross_attention", "concat"}:
            raise ValueError("fusion_mode must be cross_attention or concat")
        if self.attention_heads < 1 or self.channels[-1] % self.attention_heads:
            raise ValueError("Bottleneck width must be divisible by attention heads")
        if self.attention_radius_m <= 0 or self.attention_max_neighbors < 1:
            raise ValueError("Bottleneck attention needs a positive radius and neighbor count")
        if self.timesteps < 2 or self.beta_schedule not in {"cosine", "linear"}:
            raise ValueError("Invalid diffusion schedule")
        if self.sampling_method not in {"ddpm", "ddim"} or not 0 <= self.ddim_eta <= 1:
            raise ValueError("Invalid sampling method or DDIM eta")
        weights = (self.diffusion_weight, self.occupancy_weight,
                   self.geometry_weight, self.preservation_weight)
        if any(not math.isfinite(w) or w < 0 for w in weights) or self.diffusion_weight <= 0:
            raise ValueError("Loss weights must be finite, nonnegative, and include diffusion")
        if self.residual.max_candidate_sites is not None:
            raise ValueError("The candidate support must not be capped")
        if self.residual.candidate_method != "edge_aware" or self.residual.expansion_zyx != (0, 0, 0):
            raise ValueError("Only Stage-I edge-aware interpolation may expand support")

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "BottleneckDiffusionConfig":
        values = dict(values)
        nested = dict(values.pop("residual"))
        for key in ("channels", "query_radii_m", "expansion_zyx"):
            if key in nested:
                nested[key] = tuple(nested[key])
        for key in ("channels", "spatial_strides"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(residual=ResidualStage2Config(**nested), **values)

    @classmethod
    def from_json(cls, path: str | Path) -> "BottleneckDiffusionConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
