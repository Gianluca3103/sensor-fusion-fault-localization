"""Evaluate K-Radar object footprints against coarse reconstruction masks.

This command intentionally does not run an object detector. Coarse
reconstruction produces a three-channel BEV raster, not a LiDAR point cloud,
and this repository currently has no detector that consumes that raster.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from Fault_Localization_Model.io_utils import atomic_write_json, write_csv_rows
from Fault_Localization_Model.kradar_dataset import (
    load_kradar_annotations,
    resolve_kradar_label_path,
)
from Fault_Localization_Model.sample_utils import load_sample_metadata
from PFS.training_utils import _split_paths

from .coarse_reconstruction.coarse_config import (
    build_selector_config,
    load_config,
)
from .fault_selector_cache import load_selector_cache, selector_cache_path
from .object_evaluation import (
    BEVGeometry,
    object_mask_overlap,
    oriented_box_corners_xy,
    summarize_object_overlaps,
)


DETECTOR_UNAVAILABLE_REASON = (
    "Coarse reconstruction returns a 3-channel BEV raster "
    "(occupancy, log density, robust upper height), not a LiDAR point cloud. "
    "No compatible frozen object detector exists in this repository, and no "
    "BEV-to-point-cloud conversion was invented for evaluation."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--kradar-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/coarse_reconstruction.json"),
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--revised-label-root", type=Path)
    parser.add_argument(
        "--label-version",
        choices=("auto", "v1_0", "v1_1", "v2_0", "v2_1"),
        default="auto",
    )
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--visualize-samples", type=int, default=10)
    parser.add_argument(
        "--affected-fraction-bins",
        type=float,
        nargs="+",
        default=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
    )
    return parser.parse_args()


def _metric_to_image(
    points: np.ndarray,
    geometry: BEVGeometry,
) -> np.ndarray:
    columns = (
        (points[:, 1] - geometry.y_range[0]) / geometry.y_resolution
    )
    rows = (
        (geometry.x_range[1] - points[:, 0]) / geometry.x_resolution
    )
    return np.column_stack((columns, rows))


def _visualize(
    destination: Path,
    clean_rgb: np.ndarray,
    reconstruction_mask: np.ndarray,
    overlaps,
    geometry: BEVGeometry,
    title: str,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor="black")
    mask_overlay = np.zeros((*reconstruction_mask.shape, 4), dtype=np.float32)
    mask_overlay[..., 0] = 1.0
    mask_overlay[..., 3] = reconstruction_mask.astype(np.float32) * 0.35
    for axis, include_mask in zip(axes, (False, True)):
        axis.imshow(clean_rgb)
        if include_mask:
            axis.imshow(mask_overlay, interpolation="nearest")
        for overlap in overlaps:
            corners = _metric_to_image(
                oriented_box_corners_xy(overlap.annotation), geometry
            )
            color = "lime" if overlap.any_overlap else "cyan"
            axis.add_patch(
                Polygon(
                    corners,
                    closed=True,
                    fill=False,
                    edgecolor=color,
                    linewidth=1.3,
                )
            )
            center = _metric_to_image(
                np.asarray(((overlap.annotation.x, overlap.annotation.y),)),
                geometry,
            )[0]
            axis.text(
                center[0],
                center[1],
                f"{overlap.annotation.class_name}\n{overlap.affected_fraction:.0%}",
                color=color,
                fontsize=6,
            )
        axis.set_title(
            "GT oriented boxes" if not include_mask else "GT + reconstruction mask",
            color="white",
        )
        axis.set_xlim(0, geometry.width)
        axis.set_ylim(geometry.height, 0)
        axis.axis("off")
    figure.suptitle(title, color="white")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination,
        dpi=150,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


def _record(
    *,
    sample_path: Path,
    label_path: Path,
    metadata: dict,
    overlap,
) -> dict:
    annotation = overlap.annotation
    box = {
        "x": annotation.x,
        "y": annotation.y,
        "z": annotation.z,
        "yaw": annotation.yaw,
        "length": annotation.length,
        "width": annotation.width,
        "height": annotation.height,
    }
    return {
        "sequence_id": str(metadata["sequence"]),
        "frame_id": str(metadata["lidar_index"]),
        "class": annotation.class_name,
        "box": box,
        "any_overlap": overlap.any_overlap,
        "affected_fraction": overlap.affected_fraction,
        "overlap_area_m2": overlap.overlap_area_m2,
        "clean_detected": None,
        "corrupted_detected": None,
        "reconstructed_detected": None,
        "clean_iou": None,
        "corrupted_iou": None,
        "reconstructed_iou": None,
        "recovered": None,
        "fault": str(metadata.get("fault", "")),
        "severity": metadata.get("severity"),
        "sample_path": str(sample_path),
        "label_path": str(label_path),
    }


def main() -> None:
    args = parse_args()
    if args.visualize_samples < 0:
        raise ValueError("--visualize-samples cannot be negative")
    fraction_bins = tuple(args.affected_fraction_bins)
    selector_config = build_selector_config(load_config(args.config))
    sample_paths = _split_paths(
        args.data_root,
        args.split,
        args.limit_samples,
        args.seed,
    )

    records = []
    visualized = 0
    for index, sample_path in enumerate(sample_paths, 1):
        metadata = load_sample_metadata(sample_path)
        cache = load_selector_cache(
            selector_cache_path(sample_path, args.data_root),
            selector_config,
        )
        reconstruction_mask = cache["reconstruction_mask"].astype(bool)
        geometry = BEVGeometry.from_metadata(metadata, reconstruction_mask.shape)
        label_path = resolve_kradar_label_path(
            args.kradar_root,
            metadata,
            revised_label_root=args.revised_label_root,
        )
        annotations = load_kradar_annotations(
            label_path,
            version=args.label_version,
        )
        overlaps = tuple(
            object_mask_overlap(annotation, reconstruction_mask, geometry)
            for annotation in annotations
        )
        records.extend(
            _record(
                sample_path=sample_path,
                label_path=label_path,
                metadata=metadata,
                overlap=overlap,
            )
            for overlap in overlaps
        )

        if visualized < args.visualize_samples:
            with np.load(sample_path, allow_pickle=False) as sample:
                clean_rgb = np.asarray(sample["clean_rgb"], dtype=np.uint8)
            _visualize(
                args.output_root
                / "visualizations"
                / f"{index:05d}_{sample_path.stem}.png",
                clean_rgb,
                reconstruction_mask,
                overlaps,
                geometry,
                f"Sequence {metadata['sequence']} | LiDAR {metadata['lidar_index']}",
            )
            visualized += 1
        if index % 100 == 0 or index == len(sample_paths):
            print(
                f"Evaluated {index}/{len(sample_paths)} samples; "
                f"object rows={len(records)}",
                flush=True,
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_root / "objects.json", records)
    csv_rows = []
    for record in records:
        row = dict(record)
        row["box"] = json.dumps(row["box"], separators=(",", ":"))
        csv_rows.append(row)
    write_csv_rows(args.output_root / "objects.csv", csv_rows)

    overlap_summary = summarize_object_overlaps(
        records,
        fraction_bins=fraction_bins,
    )
    summary = {
        "split": args.split,
        "samples": len(sample_paths),
        "label_version": args.label_version,
        "revised_label_root": (
            str(args.revised_label_root) if args.revised_label_root else None
        ),
        "overlap_evaluation": overlap_summary,
        "primary_metric_population": "objects with any_overlap=true",
        "detector_evaluation": {
            "status": "unavailable",
            "reason": DETECTOR_UNAVAILABLE_REASON,
            "object_loss_rate": None,
            "object_recovery_rate": None,
            "object_preservation_rate": None,
        },
    }
    atomic_write_json(args.output_root / "summary.json", summary)
    print(f"Saved object overlap evaluation to {args.output_root}")
    print(f"Detector evaluation unavailable: {DETECTOR_UNAVAILABLE_REASON}")


if __name__ == "__main__":
    main()
