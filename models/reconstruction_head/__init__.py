"""Building blocks for the second-stage BEV reconstruction pipeline."""

from .fault_selector import (
    FaultBlob,
    FaultSelection,
    FaultSelector,
    FaultSelectorConfig,
)
from .fault_selector_simplified import (
    FaultSelectorSimplified,
    SimplifiedFaultComponent,
    SimplifiedFaultSelection,
    SimplifiedFaultSelectorConfig,
)
from .fault_selector_cache import (
    InvalidSelectorCacheError,
    build_selector_cache_entry,
    load_selector_inputs,
    load_selector_cache,
    selector_cache_path,
    selector_cache_root,
)
from .coarse_reconstruction.coarse_model import (
    AbsolutePositionEncoder,
    BottleneckFusionBlock,
    CoarseReplacementHead,
    CoarseReconstructionModel,
    GlobalFusionBlock,
    GlobalLidarEncoder,
    GlobalRadarEncoder,
    LocalToGlobalCrossAttention,
    LocalUNetDecoder,
    LocalUNetEncoder,
)
from .coarse_reconstruction.coarse_config import (
    CoarseReconstructionConfig,
    build_configs,
    build_selector_config,
    load_config,
)
from .coarse_reconstruction.coarse_loss import (
    CoarseLossConfig,
    MaskedBEVReconstructionLoss,
    ObservabilityWeightingConfig,
    coarse_reconstruction_metrics,
    occupancy_bce_weights,
)
from .coarse_dataset import CoarseReconstructionDataset, load_bev_triplet
from .diffusion_process.diffusion_process import (
    BEVChannelNormalization,
    DiffusionProcessConfig,
    GaussianNoiseSchedule,
    MaskedEpsilonMSELoss,
    residual_target,
)
from .diffusion_process.residual_diffusion import (
    DiffusionDownBlock,
    DiffusionUpBlock,
    MaskedResidualDiffusion,
    ResidualDiffusionUNet,
    ResidualDiffusionUNetConfig,
    SinusoidalTimeEmbedding,
    TimeConditionedResidualBlock,
)
from .diffusion_process.diffusion_pipeline import (
    FrozenCoarseDiffusionPipeline,
    ResidualDiffusionSampler,
    load_frozen_coarse_model,
    validate_diffusion_checkpoint_compatibility,
)
from .diffusion_process.diffusion_metrics import (
    bev_occupancy,
    occupancy_metrics,
    per_channel_continuous_metrics,
    reconstruction_stage_metrics,
)

__all__ = [
    "BEVChannelNormalization",
    "AbsolutePositionEncoder",
    "BottleneckFusionBlock",
    "CoarseReplacementHead",
    "CoarseReconstructionDataset",
    "CoarseLossConfig",
    "CoarseReconstructionConfig",
    "CoarseReconstructionModel",
    "DiffusionDownBlock",
    "DiffusionProcessConfig",
    "DiffusionUpBlock",
    "GlobalFusionBlock",
    "GlobalLidarEncoder",
    "GlobalRadarEncoder",
    "GaussianNoiseSchedule",
    "LocalToGlobalCrossAttention",
    "LocalUNetDecoder",
    "LocalUNetEncoder",
    "MaskedBEVReconstructionLoss",
    "ObservabilityWeightingConfig",
    "MaskedEpsilonMSELoss",
    "MaskedResidualDiffusion",
    "FaultBlob",
    "FaultSelection",
    "FaultSelector",
    "FaultSelectorConfig",
    "FaultSelectorSimplified",
    "SimplifiedFaultComponent",
    "SimplifiedFaultSelection",
    "SimplifiedFaultSelectorConfig",
    "InvalidSelectorCacheError",
    "ResidualDiffusionSampler",
    "ResidualDiffusionUNet",
    "ResidualDiffusionUNetConfig",
    "SinusoidalTimeEmbedding",
    "TimeConditionedResidualBlock",
    "FrozenCoarseDiffusionPipeline",
    "load_bev_triplet",
    "coarse_reconstruction_metrics",
    "build_configs",
    "build_selector_config",
    "build_selector_cache_entry",
    "load_config",
    "occupancy_bce_weights",
    "bev_occupancy",
    "load_frozen_coarse_model",
    "load_selector_cache",
    "load_selector_inputs",
    "validate_diffusion_checkpoint_compatibility",
    "occupancy_metrics",
    "per_channel_continuous_metrics",
    "reconstruction_stage_metrics",
    "residual_target",
    "selector_cache_path",
    "selector_cache_root",
]
