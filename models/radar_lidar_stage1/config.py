"""Physical geometry and separately ablatable Stage-1 model settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class VoxelGrid:
    minimum_xyz: tuple[float, float, float] = (-128.0, -128.0, -32.0)
    maximum_xyz: tuple[float, float, float] = (128.0, 128.0, 32.0)
    size_xyz: tuple[float, float, float] = (0.2, 0.2, 0.25)

    def __post_init__(self) -> None:
        for low, high, step in zip(self.minimum_xyz, self.maximum_xyz, self.size_xyz):
            cells = (high - low) / step
            if not all(math.isfinite(v) for v in (low, high, step)) or step <= 0 or cells <= 0 or abs(cells-round(cells)) > 1e-5:
                raise ValueError("Grid limits must contain a whole, positive number of finite voxels")

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        x, y, z = (round((high-low)/step) for low, high, step in zip(self.minimum_xyz, self.maximum_xyz, self.size_xyz))
        return z, y, x

    def scale_shape(self, stride: int) -> tuple[int, int, int]:
        if stride < 1 or any(n % stride for n in self.shape_zyx):
            raise ValueError("Scale stride must divide all grid dimensions")
        return tuple(n // stride for n in self.shape_zyx)

    def centers_xyz(self, coords_bzyx, stride: int = 1):
        """Return physical centers of sparse cells, preserving input device."""
        import torch
        xyz = coords_bzyx[:, [3, 2, 1]].to(torch.float32)
        origin = xyz.new_tensor(self.minimum_xyz)
        size = xyz.new_tensor(self.size_xyz) * stride
        return origin + (xyz + 0.5) * size


@dataclass(frozen=True)
class Stage1Config:
    grid: VoxelGrid = VoxelGrid()
    channels: tuple[int, ...] = (32, 64, 128, 256)
    attention_radii_m: tuple[float, ...] = (0.75, 1.5, 3.0, 6.0)
    max_neighbors: int = 24
    attention_dim: int = 64
    growth_scales: tuple[int, ...] = (4,)
    # The defaults are approximately one voxel diagonal at each stride.
    # They define geometric proxies, not verified shared reflector identities.
    positive_radii_m: tuple[float, ...] = (0.4, 0.8, 1.6, 3.2)
    corr_scale_weights: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    temperature: float = 0.1
    negative_strategy: str = "nearest"  # nearest (hard local) or random_local
    num_negatives: int = 16
    correspondence_weight: float = 1.0
    geometric_weight: float = 0.2
    confidence_weight: float = 0.1
    occupancy_weight: float = 1.0
    local_geometry_weight: float = 1.0
    confidence_sigma_m: float = 0.5
    geometry_eval_tolerance_m: float = 0.2
    # Zero preserves the original checkpoint architecture. The surface config
    # enables multiple radar-conditioned LiDAR locations per fine radar site.
    surface_proposals_per_site: int = 0
    surface_radius_m: float = 1.5
    surface_target_neighbors: int = 24
    surface_geometry_weight: float = 0.0
    surface_confidence_weight: float = 0.0
    surface_match_sigma_m: float = 0.2
    surface_context_radii_m: tuple[float, ...] = (1.0, 2.0, 4.0, 6.0)
    surface_context_neighbors: int = 16

    def __post_init__(self) -> None:
        if not 1 <= len(self.channels) <= 4 or min(self.channels) < 4:
            raise ValueError("Use 1–4 scales with at least four channels")
        if any(b <= a for a,b in zip(self.channels[:-1],self.channels[1:])):
            raise ValueError("Channel widths must increase across scales")
        if len(self.attention_radii_m) < len(self.channels) or any(x <= 0 for x in self.attention_radii_m[:len(self.channels)]):
            raise ValueError("A positive physical radius is required for each scale")
        if len(self.positive_radii_m) < len(self.channels) or any(x <= 0 or x >= self.attention_radii_m[i] for i,x in enumerate(self.positive_radii_m[:len(self.channels)])):
            raise ValueError("Each positive radius must be positive and smaller than its local search radius")
        if len(self.corr_scale_weights) < len(self.channels) or any(x < 0 for x in self.corr_scale_weights[:len(self.channels)]):
            raise ValueError("Each correspondence scale weight must be nonnegative")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Contrastive temperature must be positive")
        if self.negative_strategy not in ("nearest", "random_local") or self.num_negatives < 1:
            raise ValueError("Use nearest or random_local with at least one negative")
        if self.max_neighbors < 1 or self.attention_dim < 4:
            raise ValueError("Attention capacity must be positive")
        if len(set(self.growth_scales)) != len(self.growth_scales) or any(x < 1 or x > len(self.channels) for x in self.growth_scales):
            raise ValueError("Growth scales must be distinct valid one-based scale indices")
        if any(not math.isfinite(x) or x < 0 for x in (self.correspondence_weight, self.geometric_weight, self.confidence_weight, self.occupancy_weight, self.local_geometry_weight)):
            raise ValueError("Loss weights must be nonnegative")
        if self.correspondence_weight+self.geometric_weight+self.confidence_weight+self.surface_geometry_weight+self.surface_confidence_weight == 0:
            raise ValueError("At least one implemented training objective must be enabled")
        if self.confidence_sigma_m <= 0 or self.geometry_eval_tolerance_m <= 0:
            raise ValueError("Confidence sigma and geometry evaluation tolerance must be positive meters")
        if (self.surface_proposals_per_site < 0 or self.surface_target_neighbors < 1
                or self.surface_context_neighbors < 1
                or self.surface_radius_m <= 0 or self.surface_match_sigma_m <= 0
                or min(self.surface_geometry_weight,self.surface_confidence_weight) < 0):
            raise ValueError("Invalid surface proposal settings")
        if len(self.surface_context_radii_m) < len(self.channels) or any(
                radius <= 0 for radius in self.surface_context_radii_m[:len(self.channels)]):
            raise ValueError("Surface context needs a positive physical radius per scale")
        if self.surface_proposals_per_site == 0 and (self.surface_geometry_weight or self.surface_confidence_weight):
            raise ValueError("Surface losses require surface proposals")
        if self.surface_proposals_per_site and self.surface_geometry_weight == 0:
            raise ValueError("Surface proposals need a geometric training objective")
        self.grid.scale_shape(2 ** (len(self.channels)-1))

    def as_dict(self) -> dict:
        return asdict(self)
