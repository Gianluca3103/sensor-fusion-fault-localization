"""VoD coarse reconstruction and residual-diffusion components."""

from .fault_selector import FaultBlob, FaultSelection, FaultSelector, FaultSelectorConfig
from .fault_selector_cache import (
    InvalidSelectorCacheError,
    build_selector_cache_entry,
    load_selector_cache,
    load_selector_inputs,
    selector_cache_path,
    selector_cache_root,
)
from .coarse_dataset import (
    CoarseReconstructionDataset,
    coarse_reconstruction_collate,
    load_bev_grid_geometry,
    load_bev_triplet,
)
from .geometric_augmentation import (
    GeometricAugmentationConfig,
    GeometricTransform,
    HorizontalFlipConfig,
    ReconstructionGeometricAugmentation,
    ScaleAugmentationConfig,
    TranslationAugmentationConfig,
    YawAugmentationConfig,
)
from .pointpillars import (
    BEVGridGeometry,
    PillarFeatureNet,
    PillarScatter,
    Pillarizer,
    PointPillarsConfig,
    PointPillarsEncoder,
    PointPillarsOutput,
)
from .PointPillarV2 import (
    NeighborAwarePillarEnhancer,
    PointPillarsEncoderV2,
    PointPillarsV2Config,
)
from .PointPillarV3 import (
    PillarFeatureNetV3,
    PointPillarsEncoderV3,
    PointPillarsV3Config,
)
from .coarse_reconstruction.coarse_config import (
    CoarseReconstructionConfig,
    build_augmentation_config,
    build_configs,
    build_selector_config,
    load_config,
)
from .coarse_reconstruction.coarse_loss import (
    CoarseLossConfig,
    MaskedBEVReconstructionLoss,
    ObservabilityWeightingConfig,
    OccupancyLossConfig,
    coarse_reconstruction_metrics,
    coarse_reconstruction_range_metrics,
    occupancy_bce_weights,
    tolerance_radius_cells,
)
from .coarse_reconstruction.coarse_model import (
    CoarseReplacementHead,
    CoarseReconstructionModel,
)
from .coarse_reconstruction.hrnet_backbone import (
    HRNetBackbone,
    HRNetConfig,
    HRNetFusion,
    HRNetModule,
    HRNetResidualBlock,
    HRNetTransition,
)
from .diffusion_process.diffusion_process import (
    BEVChannelNormalization,
    DiffusionProcessConfig,
    GaussianNoiseSchedule,
    MaskedFlowMSELoss,
    ResidualChannelNormalization,
    residual_target,
)
from .diffusion_process.local_diffusion import (
    DiffusionRefinementBlock,
    FineDiffusionConfig,
    FineDiffusionRefiner,
    GlobalFaultyLidarEncoder,
    AuxiliaryConditionEncoder,
    LocalResidualDiffusionTransformer,
    MaskedExactReconstructionLoss,
    MaskedNoDegradationLoss,
    ReconstructionCropBatch,
    ReconstructionCropExtractor,
    SinusoidalTimeEmbedding,
    WindowAttention2d,
    fine_diffusion_architecture_metadata,
    validate_fine_diffusion_checkpoint_compatibility,
)
from .diffusion_process.residual_statistics import (
    ResidualStatisticsAccumulator,
    estimate_training_residual_statistics,
)
from .diffusion_process.diffusion_pipeline import (
    FrozenCoarseFineDiffusionPipeline,
    load_frozen_coarse_model,
)
from .diffusion_process.diffusion_metrics import (
    bev_occupancy,
    occupancy_metrics,
    per_channel_continuous_metrics,
    reconstruction_stage_metrics,
)

__all__ = [name for name in globals() if not name.startswith("_")]
