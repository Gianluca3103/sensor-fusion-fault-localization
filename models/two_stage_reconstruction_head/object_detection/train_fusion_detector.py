"""Train direct faulty-LiDAR/radar PointPillars fusion on VoD boxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from Fault_Localization_Model.io_utils import (
    atomic_torch_save,
    atomic_write_json,
    write_csv_rows,
)
from Fault_Localization_Model.sample_utils import load_sample_metadata
from models.Fault_Localization.training_utils import (
    _split_paths,
    resolve_device,
    seed_everything,
)
from models.two_stage_reconstruction_head.coarse_dataset import (
    load_bev_grid_geometry,
    load_bev_triplet,
)

from .annotations import DEFAULT_VOD_CLASSES, VODAnnotationLoader
from .detector import decode_detections, detector_loss, make_detection_targets
from .fusion_detector import FusionDetectorConfig, PointPillarsHRNetFusionDetector
from .metrics import evaluate_detection_conditions


class FaultyFusionDetectionDataset(Dataset):
    """Faulty LiDAR and radar inputs paired only with annotation targets."""

    def __init__(
        self,
        paths: list[Path],
        radar_root: Path,
        annotations: VODAnnotationLoader | None,
        clean_lidar_root: Path | None = None,
    ) -> None:
        self.paths = tuple(paths)
        self.radar_root = radar_root
        self.annotations = annotations
        self.clean_lidar_root = clean_lidar_root

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        path = self.paths[index]
        item = load_bev_triplet(
            path,
            self.radar_root,
            include_pointpillars_inputs=True,
        )
        metadata = load_sample_metadata(path)
        frame_id = str(metadata["frame_id"])
        result = {
            "faulty_lidar_points": item["faulty_lidar_points"],
            "radar_points": item["radar_points"],
            "frame_id": frame_id,
            "sample_path": str(path),
            "boxes": self.annotations.load(frame_id) if self.annotations else [],
        }
        if self.clean_lidar_root is not None:
            from Fault_Localization_Model.vod_dataset import load_vod_lidar

            lidar_path = self.clean_lidar_root / f"{int(frame_id):05d}.bin"
            if not lidar_path.is_file():
                raise FileNotFoundError(f"Clean VoD LiDAR is missing: {lidar_path}")
            result["clean_lidar_points"] = torch.from_numpy(
                load_vod_lidar(lidar_path).astype(np.float32, copy=False)
            )
        return result


def fusion_detection_collate(batch: list[dict]) -> dict:
    result = {
        "faulty_lidar_points": tuple(
            item["faulty_lidar_points"] for item in batch
        ),
        "radar_points": tuple(item["radar_points"] for item in batch),
        "frame_id": [item["frame_id"] for item in batch],
        "sample_path": [item["sample_path"] for item in batch],
        "boxes": [item["boxes"] for item in batch],
    }
    if "clean_lidar_points" in batch[0]:
        result["clean_lidar_points"] = tuple(
            item["clean_lidar_points"] for item in batch
        )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--radar-root", required=True, type=Path)
    parser.add_argument("--vod-root", required=True, type=Path)
    parser.add_argument("--label-root", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=5.0e-3)
    parser.add_argument("--limit-train-samples", type=int)
    parser.add_argument("--limit-val-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def _load_config(path: Path) -> FusionDetectorConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    unknown = set(payload) - {"fusion_detector"}
    if unknown:
        raise ValueError(
            "Unknown top-level fusion configuration sections: "
            + ", ".join(sorted(unknown))
        )
    return FusionDetectorConfig.from_dict(payload.get("fusion_detector", {}))


def _move_points(
    point_clouds: tuple[torch.Tensor, ...], device: torch.device
) -> tuple[torch.Tensor, ...]:
    return tuple(
        points.to(device=device, dtype=torch.float32, non_blocking=True)
        for points in point_clouds
    )


def _single_condition_metrics(records, class_names, iou_threshold):
    # Reuse the repository's exact AP/matching implementation. Both aliases
    # intentionally contain the same predictions; only the returned condition
    # is exposed by this single-condition trainer.
    aliased = [
        {
            "frame_id": record["frame_id"],
            "ground_truth": record["ground_truth"],
            "predictions": {
                "clean": record["predictions"],
                "faulty": record["predictions"],
            },
        }
        for record in records
    ]
    summary, _frame_rows, _object_rows = evaluate_detection_conditions(
        aliased, class_names, iou_threshold
    )
    return summary["conditions"]["faulty"]


@torch.inference_mode()
def _validate(model, loader, geometry, device, use_amp):
    model.eval()
    losses = []
    records = []
    detector_config = model.config.detector
    for batch in loader:
        lidar = _move_points(batch["faulty_lidar_points"], device)
        radar = _move_points(batch["radar_points"], device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            outputs = model(lidar, radar)
            targets = make_detection_targets(
                batch["boxes"],
                model.class_names,
                geometry,
                outputs["heatmap_logits"].shape[-2:],
                detector_config.output_stride,
                device=device,
                box_regression_channels=(
                    detector_config.box_regression_channels
                ),
            )
            losses.append(float(detector_loss(outputs, targets)["loss"]))
        predictions = decode_detections(
            outputs, model.class_names, geometry, detector_config
        )
        records.extend(
            {
                "frame_id": frame_id,
                "ground_truth": boxes,
                "predictions": predicted,
            }
            for frame_id, boxes, predicted in zip(
                batch["frame_id"], batch["boxes"], predictions
            )
        )
    metrics = _single_condition_metrics(
        records,
        model.class_names,
        detector_config.match_iou_threshold,
    )
    return float(np.mean(losses)), metrics


def main() -> None:
    args = _parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("epochs/batch size must be positive and workers non-negative")
    validation_batch_size = args.validation_batch_size or args.batch_size
    if validation_batch_size < 1:
        raise ValueError("validation batch size must be positive")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    config = _load_config(args.config)
    train_paths = _split_paths(
        args.data_root, "train", args.limit_train_samples, args.seed
    )
    val_paths = _split_paths(
        args.data_root, "val", args.limit_val_samples, args.seed
    )
    geometry = load_bev_grid_geometry(train_paths[0])
    if load_bev_grid_geometry(val_paths[0]).to_dict() != geometry.to_dict():
        raise ValueError("Training and validation BEV geometries differ")
    annotations = VODAnnotationLoader(
        args.vod_root,
        geometry,
        label_root=args.label_root,
        classes=DEFAULT_VOD_CLASSES,
    )
    train_frame_ids = [
        str(load_sample_metadata(path)["frame_id"]) for path in train_paths
    ]
    val_frame_ids = [
        str(load_sample_metadata(path)["frame_id"]) for path in val_paths
    ]
    annotations.validate_split(train_frame_ids)
    annotations.validate_split(val_frame_ids)
    train_dataset = FaultyFusionDetectionDataset(
        train_paths, args.radar_root, annotations
    )
    val_dataset = FaultyFusionDetectionDataset(
        val_paths, args.radar_root, annotations
    )
    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "collate_fn": fusion_detection_collate,
    }
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=validation_batch_size,
        shuffle=False,
        **loader_options,
    )
    model = PointPillarsHRNetFusionDetector(
        DEFAULT_VOD_CLASSES, geometry, config
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and not args.no_amp
    )
    use_amp = scaler.is_enabled()
    args.output_root.mkdir(parents=True, exist_ok=True)
    history = []
    best_map = -1.0
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Training samples: {len(train_dataset)}; validation: {len(val_dataset)}; "
        f"parameters: {parameter_count:,}; direct LiDAR-radar fusion: True",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "heatmap_loss": 0.0, "regression_loss": 0.0}
        progress = tqdm(
            train_loader,
            desc=f"fusion epoch {epoch:03d}/{args.epochs:03d}",
            dynamic_ncols=True,
        )
        for batch in progress:
            lidar = _move_points(batch["faulty_lidar_points"], device)
            radar = _move_points(batch["radar_points"], device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                outputs = model(lidar, radar)
                targets = make_detection_targets(
                    batch["boxes"],
                    model.class_names,
                    geometry,
                    outputs["heatmap_logits"].shape[-2:],
                    config.detector.output_stride,
                    device=device,
                    box_regression_channels=(
                        config.detector.box_regression_channels
                    ),
                )
                losses = detector_loss(outputs, targets)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            for key in totals:
                totals[key] += float(losses[key].detach())
            progress.set_postfix(loss=f"{float(losses['loss']):.4f}")
        validation_loss, validation_metrics = _validate(
            model, val_loader, geometry, device, use_amp
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            **{
                f"train/{key}": value / max(len(train_loader), 1)
                for key, value in totals.items()
            },
            "val/loss": validation_loss,
            "val/map": validation_metrics["map"],
            "val/precision": validation_metrics["precision"],
            "val/recall": validation_metrics["recall"],
            "val/f1": validation_metrics["f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}: train/loss={row['train/loss']:.5f} "
            f"val/loss={validation_loss:.5f} "
            f"val/mAP={100 * validation_metrics['map']:.2f}% "
            f"P={100 * validation_metrics['precision']:.2f}% "
            f"R={100 * validation_metrics['recall']:.2f}% "
            f"F1={100 * validation_metrics['f1']:.2f}%",
            flush=True,
        )
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "fusion_detector_config": config.to_dict(),
            "class_names": model.class_names,
            "grid_geometry": geometry.to_dict(),
            "epoch": epoch,
            "validation_metrics": validation_metrics,
            "training_split": "train_faulty_lidar_plus_radar",
            "model_selection_split": "val_faulty_lidar_plus_radar",
            "inference_inputs": ["faulty_lidar_points", "radar_points"],
            "uses_clean_lidar_as_input": False,
            "uses_fault_selector": False,
            "uses_reconstruction": False,
        }
        atomic_torch_save(checkpoint, args.output_root / "last_checkpoint.pt")
        if validation_metrics["map"] > best_map:
            best_map = validation_metrics["map"]
            atomic_torch_save(checkpoint, args.output_root / "best_model.pt")
        write_csv_rows(args.output_root / "history.csv", history)
    atomic_write_json(
        args.output_root / "training_summary.json",
        {
            "best_validation_map": best_map,
            "classes": list(model.class_names),
            "fusion_detector_config": config.to_dict(),
            "grid_geometry": geometry.to_dict(),
            "train_samples": len(train_dataset),
            "validation_samples": len(val_dataset),
            "test_annotations_used": False,
            "uses_clean_lidar_as_input": False,
            "uses_fault_selector": False,
            "uses_reconstruction": False,
        },
    )


if __name__ == "__main__":
    main()
