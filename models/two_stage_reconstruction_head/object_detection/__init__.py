"""Frozen-detector evaluation for reconstructed View-of-Delft BEVs."""

from .annotations import (
    DEFAULT_VOD_CLASSES,
    RotatedBEVBox,
    VODAnnotationLoader,
)
from .detector import (
    BEVDetectorConfig,
    LightweightBEVDetector,
    decode_detections,
    detector_loss,
    make_detection_targets,
)
from .geometry import rotated_box_iou, rotated_nms
from .fusion_detector import (
    AnchorFreeCenterHead,
    FusionDetectorConfig,
    PointPillarsHRNetFusionDetector,
)
from .metrics import evaluate_detection_conditions

__all__ = [name for name in globals() if not name.startswith("_")]
