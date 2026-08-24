"""Evaluate clean, faulty, and oracle-repaired post-scatter features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from Fault_Localization_Model.io_utils import atomic_write_json, write_csv_rows
from Fault_Localization_Model.sample_utils import load_sample_metadata
from models.Fault_Localization.training_utils import _split_paths, resolve_device, seed_everything
from models.two_stage_reconstruction_head.coarse_dataset import load_bev_grid_geometry
from models.two_stage_reconstruction_head.coarse_reconstruction.pointpillar_feature_reconstruction import (
    CoarsePointPillarFeatureReconstructor,
    PointPillarFeatureCacheDataset,
    PointPillarFeatureReconstructionConfig,
)
from models.two_stage_reconstruction_head.object_detection.annotations import (
    DEFAULT_VOD_CLASSES,
    VODAnnotationLoader,
)
from models.two_stage_reconstruction_head.object_detection.detector import (
    BEVDetectorConfig,
    LightweightBEVDetector,
    decode_detections,
)
from models.two_stage_reconstruction_head.object_detection.metrics import (
    evaluate_detection_conditions,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--feature-cache-root", required=True, type=Path)
    parser.add_argument("--vod-root", required=True, type=Path)
    parser.add_argument("--detector-checkpoint", required=True, type=Path)
    parser.add_argument("--reconstruction-checkpoint", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--label-root", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _collate(batch: list[dict]) -> dict:
    tensor_keys = (
        "clean_features",
        "faulty_features",
        "feature_repair_mask",
        "radar_features",
        "feature_halo_mask",
    )
    return {
        **{key: torch.stack([item[key] for item in batch]) for key in tensor_keys},
        "sample_path": [item["sample_path"] for item in batch],
    }


def main() -> None:
    args = _parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    paths = _split_paths(args.data_root, args.split, args.limit_samples, args.seed)
    dataset = PointPillarFeatureCacheDataset(
        paths, args.feature_cache_root, args.data_root
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=_collate,
    )
    checkpoint = torch.load(
        args.detector_checkpoint, map_location=device, weights_only=False
    )
    detector_config = BEVDetectorConfig(**checkpoint["detector_config"])
    model = LightweightBEVDetector(
        tuple(checkpoint["class_names"]), detector_config
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    reconstructor = None
    if args.reconstruction_checkpoint is not None:
        reconstruction_checkpoint = torch.load(
            args.reconstruction_checkpoint,
            map_location=device,
            weights_only=False,
        )
        reconstruction_config = PointPillarFeatureReconstructionConfig.from_dict(
            reconstruction_checkpoint["model_config"]
        )
        reconstructor = CoarsePointPillarFeatureReconstructor(
            reconstruction_config
        ).to(device)
        reconstructor.load_state_dict(reconstruction_checkpoint["model_state_dict"])
        reconstructor.eval()
    geometry = load_bev_grid_geometry(paths[0])
    annotations = VODAnnotationLoader(
        args.vod_root, geometry, label_root=args.label_root
    )
    conditions = (
        ("clean", "faulty", "oracle", "coarse")
        if reconstructor is not None
        else ("clean", "faulty", "oracle")
    )
    records = []
    with torch.inference_mode():
        for batch in loader:
            clean = batch["clean_features"].to(device, non_blocking=True)
            faulty = batch["faulty_features"].to(device, non_blocking=True)
            repair = batch["feature_repair_mask"].to(device, non_blocking=True)
            oracle = repair * clean + (1.0 - repair) * faulty
            coarse = None
            if reconstructor is not None:
                coarse = reconstructor(
                    faulty,
                    batch["radar_features"].to(device, non_blocking=True),
                    repair,
                    batch["feature_halo_mask"].to(device, non_blocking=True),
                )["coarse_features"]
            predictions = {}
            condition_tensors = [
                ("clean", clean),
                ("faulty", faulty),
                ("oracle", oracle),
            ]
            if coarse is not None:
                condition_tensors.append(("coarse", coarse))
            for condition, tensor in condition_tensors:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    output = model(tensor)
                predictions[condition] = decode_detections(
                    output, model.class_names, geometry, detector_config
                )
            for index, sample_path in enumerate(batch["sample_path"]):
                frame_id = str(load_sample_metadata(sample_path)["frame_id"])
                records.append(
                    {
                        "frame_id": frame_id,
                        "ground_truth": annotations.load(frame_id),
                        "predictions": {
                            condition: predictions[condition][index]
                            for condition in conditions
                        },
                    }
                )

    summary, frame_rows, object_rows = evaluate_detection_conditions(
        records, DEFAULT_VOD_CLASSES, detector_config.match_iou_threshold
    )
    metrics = summary["conditions"]
    summary["oracle_improvement"] = {
        name: metrics["oracle"][name] - metrics["faulty"][name]
        for name in ("map", "precision", "recall", "f1", "mean_matched_iou")
    }
    summary.update(
        detector_checkpoint=str(args.detector_checkpoint.resolve()),
        detector_frozen=True,
        split=args.split,
        frames=len(records),
        interface="post_pillar_scatter_dense_features",
        reconstruction_checkpoint=(
            str(args.reconstruction_checkpoint.resolve())
            if args.reconstruction_checkpoint is not None
            else None
        ),
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_root / "summary.json", summary)
    atomic_write_json(args.output_root / "per_object_matching.json", object_rows)
    write_csv_rows(args.output_root / "frame_metrics.csv", frame_rows)
    write_csv_rows(args.output_root / "per_object_matching.csv", object_rows)
    print("\nPOST-SCATTER FEATURE ORACLE")
    print(f"{'Condition':<10} {'mAP':>9} {'Precision':>11} {'Recall':>9} {'F1':>9}")
    print("-" * 52)
    for condition in conditions:
        values = metrics[condition]
        print(
            f"{condition:<10} {100*values['map']:>8.2f}% "
            f"{100*values['precision']:>10.2f}% "
            f"{100*values['recall']:>8.2f}% {100*values['f1']:>8.2f}%"
        )
    print("Oracle - faulty:", json.dumps(summary["oracle_improvement"], indent=2))


if __name__ == "__main__":
    main()
