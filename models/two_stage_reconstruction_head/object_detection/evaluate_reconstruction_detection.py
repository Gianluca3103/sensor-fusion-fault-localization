"""Evaluate one frozen BEV detector on clean, faulty, coarse, and fine LiDAR."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
from torch.utils.data import DataLoader

from Fault_Localization_Model.io_utils import atomic_write_json, write_csv_rows
from Fault_Localization_Model.sample_utils import load_sample_metadata
from models.Fault_Localization.training_utils import _split_paths, resolve_device, seed_everything
from models.two_stage_reconstruction_head import (
    BEVChannelNormalization,
    CoarseReconstructionDataset,
    FineDiffusionRefiner,
    FrozenCoarseFineDiffusionPipeline,
    ResidualChannelNormalization,
    coarse_reconstruction_collate,
    load_frozen_coarse_model,
    validate_fine_diffusion_checkpoint_compatibility,
)
from models.two_stage_reconstruction_head.coarse_reconstruction.evaluate_coarse_by_fault import (
    _load_selector_config,
    _move_batch,
)
from models.two_stage_reconstruction_head.diffusion_process.evaluate_fine_diffusion_by_fault import (
    _diffusion_config_from_checkpoint,
    _normalizer_from_fine_config,
)
from .annotations import VODAnnotationLoader
from .detector import BEVDetectorConfig, LightweightBEVDetector, decode_detections
from .geometry import box_corners
from .metrics import evaluate_detection_conditions, match_frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-checkpoint", required=True, type=Path)
    parser.add_argument("--coarse-checkpoint", required=True, type=Path)
    parser.add_argument("--fine-checkpoint", required=True, type=Path)
    parser.add_argument("--fine-config", type=Path)
    parser.add_argument("--selector-config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--radar-root", required=True, type=Path)
    parser.add_argument("--vod-root", required=True, type=Path)
    parser.add_argument("--label-root", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--visualize-samples", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def _load_pipeline(args, device):
    fine_checkpoint = torch.load(args.fine_checkpoint, map_location="cpu", weights_only=False)
    diffusion_config = _diffusion_config_from_checkpoint(fine_checkpoint["diffusion_config"])
    validate_fine_diffusion_checkpoint_compatibility(fine_checkpoint, diffusion_config)
    bev_normalizer = _normalizer_from_fine_config(args.fine_config, diffusion_config)
    if "bev_normalization" in fine_checkpoint:
        metadata = fine_checkpoint["bev_normalization"]
        bev_normalizer = BEVChannelNormalization(
            means=metadata["means"], stds=metadata["stds"],
            epsilon=float(metadata["epsilon"]), source=metadata.get("source", "checkpoint"),
        )
    residual = fine_checkpoint.get("residual_normalization")
    if residual is None:
        raise ValueError("Fine checkpoint lacks residual_normalization")
    residual_normalizer = ResidualChannelNormalization(
        residual.get("raw_channel_stds", residual.get("channel_stds")),
        minimum_std=float(residual.get("minimum_std", diffusion_config.minimum_residual_std)),
        source=residual.get("source", "checkpoint"),
    )
    coarse, _ = load_frozen_coarse_model(args.coarse_checkpoint, device, allow_pointpillars=True)
    diffusion = FineDiffusionRefiner(diffusion_config, bev_normalizer, residual_normalizer).to(device)
    diffusion.load_state_dict(fine_checkpoint["diffusion_state_dict"], strict=True)
    pipeline = FrozenCoarseFineDiffusionPipeline(coarse, diffusion).to(device).eval()
    return pipeline, diffusion_config


def _load_detector(path: Path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model_state_dict", "detector_config", "class_names", "grid_geometry"}
    if missing := required - set(checkpoint):
        raise ValueError("Detector checkpoint is missing: " + ", ".join(sorted(missing)))
    config = BEVDetectorConfig(**checkpoint["detector_config"])
    model = LightweightBEVDetector(tuple(checkpoint["class_names"]), config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.eval().requires_grad_(False), config, checkpoint


def _grid_corners(box, geometry):
    corners = box_corners(box)
    columns = (corners[:, 1] - geometry.y_min) / geometry.pillar_size_y
    rows = geometry.height - 1 - (corners[:, 0] - geometry.x_min) / geometry.pillar_size_x
    return list(zip(columns, rows))


def _visualize(path: Path, bevs: dict, gt, predictions: dict, geometry) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="black")
    colors = {"Car": "lime", "Pedestrian": "orange", "Cyclist": "magenta"}
    for axis, condition in zip(axes, ("clean", "faulty", "coarse", "fine")):
        axis.imshow(bevs[condition][0], cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
        for box in gt:
            axis.add_patch(
                Polygon(_grid_corners(box, geometry), closed=True, fill=False, edgecolor="cyan", linewidth=1.4)
            )
        for box in predictions[condition]:
            axis.add_patch(
                Polygon(
                    _grid_corners(box, geometry), closed=True, fill=False,
                    edgecolor=colors.get(box.class_name, "yellow"), linewidth=1.0,
                )
            )
        axis.set_title(f"{condition.title()}\nGT cyan; predictions colored", color="white")
        axis.axis("off")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.visualize_samples < 0:
        raise ValueError("invalid batch/worker/visualization count")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    detector, detector_config, detector_checkpoint = _load_detector(args.detector_checkpoint, device)
    pipeline, diffusion_config = _load_pipeline(args, device)
    selector = _load_selector_config(args.selector_config)
    sample_paths = _split_paths(args.data_root, args.split, args.limit_samples, args.seed)
    frame_ids = [str(load_sample_metadata(path)["frame_id"]) for path in sample_paths]
    dataset = CoarseReconstructionDataset(
        sample_paths, args.radar_root, data_root=args.data_root,
        selector_config=selector, use_pointpillars=pipeline.coarse_model.config.pointpillars_enabled,
    )
    geometry = dataset.grid_geometry
    if detector_checkpoint["grid_geometry"] != geometry.to_dict():
        raise ValueError("Detector and reconstruction BEV geometries differ")
    annotations = VODAnnotationLoader(
        args.vod_root, geometry, label_root=args.label_root,
        classes=tuple(detector.class_names),
    )
    annotations.validate_split(frame_ids)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
        collate_fn=coarse_reconstruction_collate,
    )
    use_amp = device.type == "cuda" and not args.no_amp
    sampling_steps = args.sampling_steps or diffusion_config.sampling_steps
    records = []
    prediction_rows = []
    visualized = 0
    completed = 0
    args.output_root.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for batch in loader:
            inputs = _move_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                coarse, _ = pipeline.coarse_forward(
                    inputs["faulty_bev"], inputs["radar_bev"], inputs["reconstruction_mask"],
                    inputs["healthy_context_mask"], inputs["halo_mask"],
                    faulty_lidar_points=inputs.get("faulty_lidar_points"),
                    radar_points=inputs.get("radar_points"),
                )
                fine = pipeline.sample(
                    inputs["faulty_bev"], inputs["radar_bev"], inputs["reconstruction_mask"],
                    inputs["healthy_context_mask"], inputs["halo_mask"],
                    coarse_lidar_bev=coarse, sampling_steps=sampling_steps,
                )["final_lidar_bev"]
                condition_bevs = {
                    "clean": inputs["clean_bev"], "faulty": inputs["faulty_bev"],
                    "coarse": coarse, "fine": fine,
                }
                condition_predictions = {
                    condition: decode_detections(detector(bev), detector.class_names, geometry, detector_config)
                    for condition, bev in condition_bevs.items()
                }
            for index, sample_path in enumerate(batch["sample_path"]):
                metadata = load_sample_metadata(sample_path)
                frame_id = str(metadata["frame_id"])
                gt = annotations.load(frame_id)
                predictions = {
                    condition: condition_predictions[condition][index]
                    for condition in condition_bevs
                }
                records.append({"frame_id": frame_id, "ground_truth": gt, "predictions": predictions})
                for condition, boxes in predictions.items():
                    status = match_frame(boxes, gt, detector_config.match_iou_threshold)["prediction_status"]
                    for prediction_index, box in enumerate(boxes):
                        matched, gt_index, iou = status[prediction_index]
                        prediction_rows.append(
                            {
                                "frame_id": frame_id, "condition": condition,
                                "prediction_index": prediction_index, **box.to_dict(),
                                "matched": matched, "matched_gt_index": gt_index, "matched_iou": iou,
                            }
                        )
                if visualized < args.visualize_samples:
                    _visualize(
                        args.output_root / "visualizations" / f"{visualized:03d}_{int(frame_id):05d}.png",
                        {key: value[index].float().cpu().numpy() for key, value in condition_bevs.items()},
                        gt, predictions, geometry,
                    )
                    visualized += 1
            completed += len(batch["sample_path"])
            print(f"Evaluated {completed}/{len(dataset)} frames", flush=True)
    summary, frame_rows, object_rows = evaluate_detection_conditions(
        records, detector.class_names, detector_config.match_iou_threshold
    )
    summary.update(
        {
            "split": args.split,
            "frames": len(records),
            "classes": list(detector.class_names),
            "detector_checkpoint": str(args.detector_checkpoint),
            "coarse_checkpoint": str(args.coarse_checkpoint),
            "fine_checkpoint": str(args.fine_checkpoint),
            "frozen_detector": True,
            "test_annotations_used_for_training_or_selection": False,
        }
    )
    atomic_write_json(args.output_root / "summary.json", summary)
    write_csv_rows(args.output_root / "frame_metrics.csv", frame_rows)
    write_csv_rows(args.output_root / "predictions.csv", prediction_rows)
    write_csv_rows(args.output_root / "object_recovery.csv", object_rows)
    atomic_write_json(args.output_root / "frame_metrics.json", frame_rows)
    atomic_write_json(args.output_root / "predictions.json", prediction_rows)
    atomic_write_json(args.output_root / "object_recovery.json", object_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
