"""Deterministic, non-learned 3D voxel preprocessing utilities.

Physical point coordinates always use ``[x, y, z]``.  Sparse voxel
coordinates always use ``[z_index, y_index, x_index]`` (``zyx``), matching
the convention expected by common sparse-convolution libraries.
"""

from .config import (
    ModalityVoxelConfig,
    VoxelGridConfig,
    VoxelizationConfig,
    load_voxelization_config,
)
from .hard_voxelizer import HardVoxelizer, VoxelizedPointCloud

__all__ = (
    "HardVoxelizer",
    "ModalityVoxelConfig",
    "VoxelGridConfig",
    "VoxelizationConfig",
    "VoxelizedPointCloud",
    "load_voxelization_config",
)
