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

__all__ = (
    "HardVoxelizer",
    "ModalityVoxelConfig",
    "VoxelGridConfig",
    "VoxelizationConfig",
    "VoxelizedPointCloud",
    "load_voxelization_config",
)
