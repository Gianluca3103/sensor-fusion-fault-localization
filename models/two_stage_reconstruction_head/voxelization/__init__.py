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
from .temporal_radar_filter import (
    VoxelTemporalConsistencyConfig,
    filter_temporally_consistent_radar_voxels,
)
from .spatial_radar_filter import (
    SpatialRadarVoxelFilterConfig,
    filter_spatially_isolated_radar_voxels,
)
from .fault_targets import VoxelFaultTargets, build_voxel_fault_targets
from .fault_selector_3d import (
    OracleFaultComponent3D,
    OracleFaultSelection3D,
    OracleFaultSelector3DConfig,
    select_oracle_fault_regions_3d,
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
    "VoxelFaultTargets",
    "build_voxel_fault_targets",
    "OracleFaultComponent3D",
    "OracleFaultSelection3D",
    "OracleFaultSelector3DConfig",
    "select_oracle_fault_regions_3d",
)
