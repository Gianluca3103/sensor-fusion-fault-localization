"""Evaluate one frozen OpenPCDet PV-RCNN++ on selected conditions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Fault_Localization_Model.vod_dataset import load_vod_split_ids
from models.two_stage_reconstruction_head.object_detection.annotations import (
    DEFAULT_VOD_CLASSES,
    RotatedBEVBox,
    VODAnnotationLoader,
)
from models.two_stage_reconstruction_head.object_detection.geometry import box_corners
from models.two_stage_reconstruction_head.object_detection.metrics import (
    evaluate_detection_conditions,
)
from models.two_stage_reconstruction_head.pointpillars import BEVGridGeometry
from pcdet_integration.openpcdet_eval import (
    add_openpcdet_to_path,
    create_custom_infos,
    evaluate_checkpoint_on_condition,
    load_openpcdet_config,
)


ALL_CONDITIONS = ("clean", "faulty", "coarse", "fine")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpcdet-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--detector-checkpoint", required=True, type=Path)
    parser.add_argument("--conditions-root", required=True, type=Path)
    parser.add_argument("--vod-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--label-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--visualize-samples", type=int, default=50)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=ALL_CONDITIONS,
        default=ALL_CONDITIONS,
    )
    args = parser.parse_args()
    if args.split == "test" and args.label_root is None:
        raise ValueError(
            "Public VoD test labels are unavailable. Supply an authorized "
            "--label-root or evaluate --split val."
        )
    return args


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _prediction_boxes(annotation: dict) -> list[RotatedBEVBox]:
    boxes = annotation.get("boxes_lidar")
    if boxes is None:
        raise KeyError(
            "OpenPCDet result.pkl lacks boxes_lidar; use the upstream "
            "CustomDataset prediction exporter from the pinned revision"
        )
    names = np.asarray(annotation["name"])
    scores = np.asarray(annotation["score"], dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64)
    output = []
    for name, score, box in zip(names, scores, boxes):
        class_name = "Car" if str(name) == "Vehicle" else str(name)
        if class_name not in DEFAULT_VOD_CLASSES:
            continue
        output.append(
            RotatedBEVBox(
                class_name=class_name,
                x=float(box[0]), y=float(box[1]), z=float(box[2]),
                length=float(box[3]), width=float(box[4]), height=float(box[5]),
                yaw=float(box[6]), confidence=float(score),
            )
        )
    return output


def _ap_summary(raw: dict) -> dict:
    categories = {"bev": {}, "3d": {}}
    for key, value in raw.items():
        if not isinstance(value, (int, float, np.number)):
            continue
        lower = key.lower()
        kind = "3d" if "3d" in lower else "bev" if "bev" in lower else None
        if kind is None:
            continue
        categories[kind][key] = float(value)
    summary = {"raw": raw}
    for kind, values in categories.items():
        moderate_r40 = [
            value for key, value in values.items()
            if "moderate" in key.lower() and "r40" in key.lower()
        ]
        selected = moderate_r40 or list(values.values())
        summary[f"{kind}_map"] = float(np.mean(selected)) if selected else None
        summary[f"{kind}_ap"] = values
    return summary


def _plot_box(axis, box: RotatedBEVBox, color: str, linewidth: float) -> None:
    corners = box_corners(box)
    closed = np.vstack((corners, corners[0]))
    axis.plot(closed[:, 1], closed[:, 0], color=color, linewidth=linewidth)


def _visualize(
    destination: Path,
    frame_id: str,
    roots: dict[str, Path],
    gt: list[RotatedBEVBox],
    predictions: dict[str, list[RotatedBEVBox]],
    conditions: tuple[str, ...],
) -> None:
    figure, axes = plt.subplots(
        1, len(conditions), figsize=(5 * len(conditions), 6), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    for axis, condition in zip(axes, conditions):
        points = np.load(roots[condition] / "points" / f"{frame_id}.npy")
        axis.scatter(points[:, 1], points[:, 0], s=0.15, c="0.65", rasterized=True)
        for box in gt:
            _plot_box(axis, box, "lime", 1.2)
        for box in predictions[condition]:
            _plot_box(axis, box, "red", 1.0)
        axis.set_title(condition)
        axis.set_xlim(-32, 32)
        axis.set_ylim(0, 64)
        axis.set_aspect("equal")
        axis.set_xlabel("y [m]")
    axes[0].set_ylabel("x [m]")
    figure.suptitle(f"VoD {frame_id} | GT green | frozen PV-RCNN++ red")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    if not 0 < args.iou_threshold <= 1:
        raise ValueError("iou-threshold must be in (0, 1]")
    add_openpcdet_to_path(args.openpcdet_root)
    cfg = load_openpcdet_config(args.openpcdet_root, args.config)
    conditions = tuple(dict.fromkeys(args.conditions))
    required = {"clean", "faulty", "coarse"}
    missing = sorted(required - set(conditions))
    if missing:
        raise ValueError("Detection comparison requires: " + ", ".join(missing))
    roots = {condition: args.conditions_root / condition for condition in conditions}
    for condition, root in roots.items():
        if not (root / "points").is_dir():
            raise FileNotFoundError(f"Missing {condition} points: {root / 'points'}")
        create_custom_infos(
            args.openpcdet_root,
            cfg.DATA_CONFIG,
            list(DEFAULT_VOD_CLASSES),
            root,
            args.split,
            workers=args.workers,
        )

    official = {}
    condition_predictions = {}
    model = None
    for condition in conditions:
        metrics, predictions, model = evaluate_checkpoint_on_condition(
            args.openpcdet_root,
            cfg,
            args.detector_checkpoint,
            roots[condition],
            args.output_root / "official" / condition,
            split=args.split,
            batch_size=args.batch_size,
            workers=args.workers,
            model=model,
        )
        official[condition] = _ap_summary(metrics)
        condition_predictions[condition] = predictions

    frame_ids = load_vod_split_ids(args.vod_root, args.split)
    lengths = {condition: len(values) for condition, values in condition_predictions.items()}
    if any(length != len(frame_ids) for length in lengths.values()):
        raise RuntimeError(
            f"Prediction/frame count mismatch: frames={len(frame_ids)}, predictions={lengths}"
        )
    geometry = BEVGridGeometry(0.0, 64.0, -32.0, 32.0, 320, 320)
    annotation_loader = VODAnnotationLoader(
        args.vod_root, geometry, label_root=args.label_root
    )
    records = []
    prediction_rows = []
    for frame_index, frame_id in enumerate(frame_ids):
        gt = annotation_loader.load(frame_id)
        by_condition = {}
        for condition in conditions:
            boxes = _prediction_boxes(condition_predictions[condition][frame_index])
            by_condition[condition] = boxes
            for prediction_index, box in enumerate(boxes):
                prediction_rows.append(
                    {
                        "frame_id": frame_id,
                        "condition": condition,
                        "prediction_index": prediction_index,
                        **box.to_dict(),
                    }
                )
        records.append(
            {"frame_id": frame_id, "ground_truth": gt, "predictions": by_condition}
        )
        if frame_index < args.visualize_samples:
            _visualize(
                args.output_root / "visualizations" / f"{frame_index:03d}_{frame_id}.png",
                frame_id,
                roots,
                gt,
                by_condition,
                conditions,
            )

    matching, frame_rows, object_rows = evaluate_detection_conditions(
        records, DEFAULT_VOD_CLASSES, args.iou_threshold
    )
    official_deltas = {}
    for kind in ("bev_map", "3d_map"):
        values = {condition: official[condition][kind] for condition in conditions}
        if all(value is not None for value in values.values()):
            deltas = {"coarse_minus_faulty": values["coarse"] - values["faulty"]}
            if "fine" in values:
                deltas.update(
                    fine_minus_faulty=values["fine"] - values["faulty"],
                    fine_minus_coarse=values["fine"] - values["coarse"],
                )
            official_deltas[kind] = deltas
    summary = {
        "detector": "official OpenPCDet PVRCNNPlusPlus",
        "detector_checkpoint": str(args.detector_checkpoint.resolve()),
        "detector_frozen": True,
        "split": args.split,
        "frames": len(frame_ids),
        "same_score_nms_matching_for_all_conditions": True,
        "official_openpcdet_ap": official,
        "official_ap_improvement": official_deltas,
        "classwise_rotated_bev_matching": matching,
    }
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "frame_metrics.json", frame_rows)
    _write_csv(args.output_root / "frame_metrics.csv", frame_rows)
    _write_json(args.output_root / "per_object_matching.json", object_rows)
    _write_csv(args.output_root / "per_object_matching.csv", object_rows)
    _write_json(args.output_root / "predictions.json", prediction_rows)
    _write_csv(args.output_root / "predictions.csv", prediction_rows)
    per_class_rows = []
    for condition, values in matching["conditions"].items():
        for class_name, class_values in values["per_class"].items():
            per_class_rows.append(
                {"condition": condition, "class": class_name, **class_values}
            )
    _write_csv(args.output_root / "per_class_metrics.csv", per_class_rows)
    _write_json(args.output_root / "per_class_metrics.json", per_class_rows)
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
