"""Official KITTI-format View-of-Delft labels in the reconstruction BEV frame."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math

import numpy as np

from Fault_Localization_Model.vod_dataset import load_vod_lidar_to_camera
from models.two_stage_reconstruction_head.pointpillars import BEVGridGeometry


DEFAULT_VOD_CLASSES = ("Car", "Pedestrian", "Cyclist")


@dataclass(frozen=True)
class RotatedBEVBox:
    """Full metric BEV box in the LiDAR coordinate frame."""

    class_name: str
    x: float
    y: float
    length: float
    width: float
    yaw: float
    z: float = 0.0
    height: float = 0.0
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


def _public_root(vod_root: str | Path) -> Path:
    root = Path(vod_root)
    nested = root / "view_of_delft_PUBLIC"
    return nested if nested.is_dir() else root


def _wrap_yaw(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


class VODAnnotationLoader:
    """Load VoD labels and transform boxes from camera to LiDAR coordinates.

    VoD follows the KITTI row convention: ``class truncation occlusion alpha
    bbox h w l x y z rotation_y [score]``. Locations are bottom centres in
    the camera frame. Only the three official VoD benchmark classes are used
    by default; aliases can be supplied explicitly without changing parsing.
    """

    def __init__(
        self,
        vod_root: str | Path,
        geometry: BEVGridGeometry,
        *,
        label_root: str | Path | None = None,
        classes: tuple[str, ...] = DEFAULT_VOD_CLASSES,
        class_aliases: dict[str, str] | None = None,
    ) -> None:
        self.public_root = _public_root(vod_root)
        self.label_root = (
            Path(label_root)
            if label_root is not None
            else self.public_root / "lidar" / "training" / "label_2"
        )
        self.calibration_root = self.public_root / "lidar" / "training" / "calib"
        self.geometry = geometry
        self.geometry.validate()
        self.classes = tuple(classes)
        if not self.classes or len(set(self.classes)) != len(self.classes):
            raise ValueError("classes must be a non-empty unique sequence")
        self.class_aliases = dict(class_aliases or {})

    def _canonical_class(self, raw_name: str) -> str | None:
        name = self.class_aliases.get(raw_name, raw_name)
        return name if name in self.classes else None

    def annotation_path(self, frame_id: str | int) -> Path:
        return self.label_root / f"{int(frame_id):05d}.txt"

    def calibration_path(self, frame_id: str | int) -> Path:
        return self.calibration_root / f"{int(frame_id):05d}.txt"

    def load(self, frame_id: str | int) -> list[RotatedBEVBox]:
        label_path = self.annotation_path(frame_id)
        calibration_path = self.calibration_path(frame_id)
        if not label_path.is_file():
            raise FileNotFoundError(
                f"VoD annotation is missing for frame {int(frame_id):05d}: "
                f"{label_path}. The public VoD test split has no labels; provide "
                "--label-root only when separate authorized test annotations exist."
            )
        if not calibration_path.is_file():
            raise FileNotFoundError(f"VoD calibration is missing: {calibration_path}")

        camera_from_lidar = load_vod_lidar_to_camera(calibration_path)
        lidar_from_camera = np.linalg.inv(camera_from_lidar)
        rotation = lidar_from_camera[:3, :3]
        boxes: list[RotatedBEVBox] = []
        for line_number, raw_line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            fields = raw_line.split()
            if not fields:
                continue
            class_name = self._canonical_class(fields[0])
            if class_name is None:
                continue
            if len(fields) < 15:
                raise ValueError(
                    f"Malformed VoD label {label_path}:{line_number}; expected "
                    "at least 15 KITTI fields"
                )
            values = np.asarray([float(value) for value in fields[1:15]], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite VoD label at {label_path}:{line_number}")
            height, width, length = values[7:10]
            camera_bottom = values[10:13]
            rotation_y = float(values[13])
            if min(height, width, length) <= 0.0:
                continue

            camera_center = camera_bottom.copy()
            camera_center[1] -= height / 2.0
            homogeneous = np.append(camera_center, 1.0)
            lidar_center = (lidar_from_camera @ homogeneous)[:3]
            camera_heading = np.asarray(
                [math.cos(rotation_y), 0.0, -math.sin(rotation_y)],
                dtype=np.float64,
            )
            lidar_heading = rotation @ camera_heading
            yaw = _wrap_yaw(math.atan2(lidar_heading[1], lidar_heading[0]))
            x, y, z = (float(value) for value in lidar_center)
            if not (
                self.geometry.x_min <= x < self.geometry.x_max
                and self.geometry.y_min <= y < self.geometry.y_max
            ):
                continue
            boxes.append(
                RotatedBEVBox(
                    class_name=class_name,
                    x=x,
                    y=y,
                    z=z,
                    yaw=yaw,
                    length=float(length),
                    width=float(width),
                    height=float(height),
                )
            )
        return boxes

    def validate_split(self, frame_ids: list[str]) -> None:
        missing = [frame_id for frame_id in frame_ids if not self.annotation_path(frame_id).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)}/{len(frame_ids)} frames have no VoD annotations; "
                f"first missing frame is {missing[0]}."
            )
