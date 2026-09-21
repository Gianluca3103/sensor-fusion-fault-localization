"""Standalone import surface for deterministic 3D voxel preprocessing.

The implementation is colocated with the reconstruction data adapters, but
this package deliberately bypasses their PyTorch-heavy package initializer so
statistics, caching, visualization, and tests remain NumPy-only.
"""

from pathlib import Path

__path__ = [
    str(
        Path(__file__).resolve().parents[1]
        / "models"
        / "two_stage_reconstruction_head"
        / "voxelization"
    )
]

from .config import (  # noqa: E402
    ModalityVoxelConfig,
    VoxelGridConfig,
    VoxelizationConfig,
    load_voxelization_config,
)
from .hard_voxelizer import HardVoxelizer, VoxelizedPointCloud  # noqa: E402
from .temporal_radar_filter import (  # noqa: E402
    VoxelTemporalConsistencyConfig,
    filter_temporally_consistent_radar_voxels,
)
from .spatial_radar_filter import (  # noqa: E402
    SpatialRadarVoxelFilterConfig,
    filter_spatially_isolated_radar_voxels,
)

__all__ = (
    "HardVoxelizer",
    "ModalityVoxelConfig",
    "VoxelGridConfig",
    "VoxelizationConfig",
    "VoxelizedPointCloud",
    "VoxelTemporalConsistencyConfig",
    "filter_temporally_consistent_radar_voxels",
    "SpatialRadarVoxelFilterConfig",
    "filter_spatially_isolated_radar_voxels",
    "load_voxelization_config",
)
