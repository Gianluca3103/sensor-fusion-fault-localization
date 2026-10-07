"""Configuration for deterministic radar-only sparse reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Stage2Config:
    channels: tuple[int, int, int, int] = (24, 48, 96, 128)
    conditioning_dim: int = 32
    query_radii_m: tuple[float, float, float, float] = (0.6, 1.2, 2.4, 4.8)
    max_neighbors: int = 8
    confidence_threshold: float = 0.25
    expansion_zyx: tuple[int, int, int] = (1, 1, 1)
    max_candidate_sites: int | None = None  # Optional legacy point-proposal limit; learned regions are uncapped.
    occupancy_threshold: float = 0.5
    occupancy_weight: float = 1.0
    offset_weight: float = 1.0
    positive_weight: float = 2.0
    free_ray_tolerance_m: float = 0.15
    checkpoint_metric: str = "geom_f1_0.2m"
    diffusion_enabled: bool = False
    stage1_frozen: bool = True
    point_geometry_loss_enabled: bool = False

    def __post_init__(self) -> None:
        if len(self.channels) != 4 or min(self.channels) < 4:
            raise ValueError("Stage II requires four positive channel widths")
        if len(self.query_radii_m) != 4 or min(self.query_radii_m) <= 0:
            raise ValueError("Four positive physical query radii are required")
        if self.max_neighbors < 1 or self.conditioning_dim < 4:
            raise ValueError("Invalid conditioning capacity")
        if not 0 <= self.confidence_threshold <= 1 or not 0 <= self.occupancy_threshold <= 1:
            raise ValueError("Thresholds must be in [0,1]")
        if (self.max_candidate_sites is not None and self.max_candidate_sites < 1) or self.positive_weight <= 0:
            raise ValueError("Invalid candidate cap or positive weight")
        if min(self.occupancy_weight, self.offset_weight) < 0 or self.free_ray_tolerance_m <= 0:
            raise ValueError("Loss weights must be nonnegative and free-ray tolerance positive")
        if self.checkpoint_metric not in {"geom_f1_0.2m", "geom_f1_0.5m", "occupancy_iou"}:
            raise ValueError("Unsupported checkpoint metric")
        if self.diffusion_enabled or not self.stage1_frozen or self.point_geometry_loss_enabled:
            raise NotImplementedError("Diffusion, Stage-I fine-tuning, and point loss are disabled in V1")

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "Stage2Config":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("channels", "query_radii_m", "expansion_zyx"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)
