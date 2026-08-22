"""Export View-of-Delft through OpenPCDet's official ``CustomDataset`` contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from Fault_Localization_Model.vod_dataset.vod_io import (
    load_vod_lidar,
    load_vod_split_ids,
    resolve_vod_public_root,
)
from models.two_stage_reconstruction_head.object_detection.annotations import (
    DEFAULT_VOD_CLASSES,
    VODAnnotationLoader,
)
from models.two_stage_reconstruction_head.pointpillars import BEVGridGeometry


@dataclass(frozen=True)
class VODCustomDatasetLayout:
    root: Path
    points: Path
    labels: Path
    image_sets: Path


def _layout(output_root: str | Path) -> VODCustomDatasetLayout:
    root = Path(output_root)
    return VODCustomDatasetLayout(
        root=root,
        points=root / "points",
        labels=root / "labels",
        image_sets=root / "ImageSets",
    )


def _geometry() -> BEVGridGeometry:
    return BEVGridGeometry(
        x_min=0.0,
        x_max=64.0,
        y_min=-32.0,
        y_max=32.0,
        height=320,
        width=320,
    )


def _label_line(box) -> str:
    # Official OpenPCDet CustomDataset order is x y z dx dy dz heading class.
    return (
        f"{box.x:.8f} {box.y:.8f} {box.z:.8f} "
        f"{box.length:.8f} {box.width:.8f} {box.height:.8f} "
        f"{box.yaw:.8f} {box.class_name}\n"
    )


def export_vod_custom_dataset(
    vod_root: str | Path,
    output_root: str | Path,
    *,
    splits: tuple[str, ...] = ("train", "val", "test"),
    overwrite: bool = False,
    label_root: str | Path | None = None,
) -> dict:
    """Materialize clean VoD for upstream ``pcdet.datasets.CustomDataset``.

    The exported fourth point feature is the measured VoD reflectivity.  Test
    labels are deliberately omitted when the public release provides none.
    """

    public_root = resolve_vod_public_root(vod_root)
    layout = _layout(output_root)
    for directory in (layout.points, layout.labels, layout.image_sets):
        directory.mkdir(parents=True, exist_ok=True)

    annotations = VODAnnotationLoader(
        public_root,
        _geometry(),
        label_root=label_root,
        classes=DEFAULT_VOD_CLASSES,
    )
    summary: dict[str, dict[str, int | bool]] = {}
    seen: set[str] = set()
    for split in splits:
        frame_ids = load_vod_split_ids(public_root, split)
        labeled = all(annotations.annotation_path(frame_id).is_file() for frame_id in frame_ids)
        if split in {"train", "val"} and not labeled:
            annotations.validate_split(frame_ids)
        (layout.image_sets / f"{split}.txt").write_text(
            "".join(f"{frame_id}\n" for frame_id in frame_ids),
            encoding="utf-8",
        )
        written_points = 0
        written_labels = 0
        for frame_id in frame_ids:
            source = public_root / "lidar" / "training" / "velodyne" / f"{frame_id}.bin"
            if not source.is_file():
                raise FileNotFoundError(f"VoD LiDAR frame is missing: {source}")
            point_path = layout.points / f"{frame_id}.npy"
            if overwrite or not point_path.is_file():
                points = load_vod_lidar(source).astype(np.float32, copy=False)
                np.save(point_path, points, allow_pickle=False)
                written_points += 1
            label_path = layout.labels / f"{frame_id}.txt"
            if annotations.annotation_path(frame_id).is_file():
                if overwrite or not label_path.is_file():
                    label_path.write_text(
                        "".join(_label_line(box) for box in annotations.load(frame_id)),
                        encoding="utf-8",
                    )
                    written_labels += 1
            elif label_path.exists():
                raise RuntimeError(
                    f"Refusing stale labels for unlabeled VoD {split} frame {frame_id}: "
                    f"{label_path}"
                )
            seen.add(frame_id)
        summary[split] = {
            "frames": len(frame_ids),
            "labels_available": labeled,
            "points_written": written_points,
            "labels_written": written_labels,
        }

    manifest = {
        "format": "OpenPCDet CustomDataset",
        "point_features": ["x", "y", "z", "reflectivity"],
        "box_format": ["x", "y", "z", "dx", "dy", "dz", "heading"],
        "classes": list(DEFAULT_VOD_CLASSES),
        "source": str(public_root.resolve()),
        "unique_frames": len(seen),
        "splits": summary,
    }
    (layout.root / "vod_export_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
