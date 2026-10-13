"""Separate diffusion settings; deterministic Stage II remains unchanged."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from models.radar_lidar_stage2_residual.config import ResidualStage2Config


@dataclass(frozen=True)
class DiffusionConfig:
    residual: ResidualStage2Config = ResidualStage2Config()
    diffusion_enabled: bool = True
    timesteps: int = 16
    occupancy_denoise_weight: float = 1.0
    offset_denoise_weight: float = 1.0
    classification_weight: float = 0.5
    sample_seed: int = 1701

    def __post_init__(self) -> None:
        if not self.diffusion_enabled:
            raise ValueError("This module always runs diffusion")
        if self.residual.diffusion_enabled:
            raise ValueError("The nested residual settings describe conditioning only")
        if self.residual.max_candidate_sites is not None:
            raise ValueError("Sparse diffusion cannot cap candidate sites or generated additions")
        if self.timesteps < 2:
            raise ValueError("Diffusion requires at least two timesteps")
        if min(self.occupancy_denoise_weight, self.offset_denoise_weight,
               self.classification_weight) < 0:
            raise ValueError("Loss weights must be nonnegative")
        if self.occupancy_denoise_weight + self.offset_denoise_weight == 0:
            raise ValueError("At least one denoising loss is required")

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "DiffusionConfig":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        residual = values.pop("residual")
        for key in ("channels", "query_radii_m", "expansion_zyx"):
            if key in residual:
                residual[key] = tuple(residual[key])
        return cls(residual=ResidualStage2Config(**residual), **values)
