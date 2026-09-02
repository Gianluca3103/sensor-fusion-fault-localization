"""Evaluate a frozen faulty-LiDAR/radar fusion detector on one VoD split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from Fault_Localization_Model.io_utils import atomic_write_json, write_csv_rows
from Fault_Localization_Model.sample_utils import load_sample_metadata
from models.Fault_Localization.training_utils import (
    _split_paths,
    resolve_device,
    seed_everything,
)
from models.two_stage_reconstruction_head.coarse_dataset import (
    load_bev_grid_geometry,
)
from models.two_stage_reconstruction_head.pointpillars import BEVGridGeometry

from .annotations import VODAnnotationLoader
from .detector import BEVDetectorConfig, decode_detections
from .fusion_detector import FusionDetectorConfig, PointPillarsHRNetFusionDetector
from .metrics import match_frame
from .train_fusion_detector import (
    FaultyFusionDetectionDataset,
    _move_points,
    _single_condition_metrics,
    fusion_detection_collate,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--radar-root", required=True, type=Path)
    parser.add_argument("--vod-root", required=True, type=Path)
    parser.add_argument("--label-root", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--include-lidar-only-ablation",
        action="store_true",
        help="Also zero the radar feature map using the same frozen weights.",
    )
    return parser.parse_args()


def _load_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "model_state_dict",
        "fusion_detector_config",
        "class_names",
        "grid_geometry",
    }
    if missing := required - set(checkpoint):
        raise ValueError(
            "Fusion checkpoint is missing: " + ", ".join(sorted(missing))
        )
    config = FusionDetectorConfig.from_dict(checkpoint["fusion_detector_config"])
    geometry = BEVGridGeometry(**checkpoint["grid_geometry"])
    model = PointPillarsHRNetFusionDetector(
        tuple(checkpoint["class_names"]), geometry, config
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    return model, geometry, checkpoint


def _prediction_rows(frame_id, condition, boxes, ground_truth, threshold):
    status = match_frame(boxes, ground_truth, threshold)["prediction_status"]
    return [
        {
            "frame_id": frame_id,
            "condition": condition,
            "prediction_index": index,
            **box.to_dict(),
            "matched": status[index][0],
            "matched_gt_index": status[index][1],
            "matched_iou": status[index][2],
        }
        for index, box in enumerate(boxes)
    ]


def main() -> None:
    args = _parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    model, geometry, checkpoint = _load_model(args.checkpoint, device)
    sample_paths = _split_paths(
        args.data_root, args.split, args.limit_samples, args.seed
    )
    actual_geometry = load_bev_grid_geometry(sample_paths[0])
    if actual_geometry.to_dict() != geometry.to_dict():
        raise ValueError("Checkpoint and evaluation BEV geometries differ")
    annotations = VODAnnotationLoader(
        args.vod_root,
        geometry,
        label_root=args.label_root,
        classes=model.class_names,
    )
    frame_ids = [
        str(load_sample_metadata(path)["frame_id"]) for path in sample_paths
    ]
    annotations.validate_split(frame_ids)
    dataset = FaultyFusionDetectionDataset(
        sample_paths, args.radar_root, annotations
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=fusion_detection_collate,
    )
    conditions = ["fusion"]
    if args.include_lidar_only_ablation:
        conditions.append("lidar_only")
    records = {condition: [] for condition in conditions}
    prediction_rows = []
    use_amp = device.type == "cuda" and not args.no_amp
    completed = 0
    with torch.inference_mode():
        for batch in loader:
            lidar = _move_points(batch["faulty_lidar_points"], device)
            radar = _move_points(batch["radar_points"], device)
            decoded = {}
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                for condition in conditions:
                    outputs = model(
                        lidar,
                        radar,
                        radar_enabled=condition == "fusion",
                    )
                    decoded[condition] = decode_detections(
                        outputs,
                        model.class_names,
                        geometry,
                        model.config.detector,
                    )
            for index, frame_id in enumerate(batch["frame_id"]):
                ground_truth = batch["boxes"][index]
                for condition in conditions:
                    predictions = decoded[condition][index]
                    records[condition].append(
                        {
                            "frame_id": frame_id,
                            "ground_truth": ground_truth,
                            "predictions": predictions,
                        }
                    )
                    prediction_rows.extend(
                        _prediction_rows(
                            frame_id,
                            condition,
                            predictions,
                            ground_truth,
                            model.config.detector.match_iou_threshold,
                        )
                    )
            completed += len(batch["frame_id"])
            print(f"Evaluated {completed}/{len(dataset)} frames", flush=True)
    condition_metrics = {
        condition: _single_condition_metrics(
            records[condition],
            model.class_names,
            model.config.detector.match_iou_threshold,
        )
        for condition in conditions
    }
    summary = {
        "split": args.split,
        "frames": len(dataset),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "class_names": list(model.class_names),
        "conditions": condition_metrics,
        "frozen_detector": True,
        "inputs": ["faulty_lidar_points", "radar_points"],
        "uses_clean_lidar_as_input": False,
        "uses_fault_selector": False,
        "uses_reconstruction": False,
        "test_annotations_used_for_training_or_model_selection": False,
    }
    if "lidar_only" in condition_metrics:
        summary["fusion_minus_lidar_only_map"] = (
            condition_metrics["fusion"]["map"]
            - condition_metrics["lidar_only"]["map"]
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_root / "summary.json", summary)
    write_csv_rows(args.output_root / "predictions.csv", prediction_rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
