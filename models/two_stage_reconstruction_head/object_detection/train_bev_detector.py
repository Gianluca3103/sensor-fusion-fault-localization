"""Train a downstream detector exclusively on clean VoD training tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from Fault_Localization_Model.io_utils import atomic_torch_save, atomic_write_json, write_csv_rows
from Fault_Localization_Model.sample_utils import load_sample_metadata
from models.Fault_Localization.training_utils import _split_paths, resolve_device, seed_everything
from models.two_stage_reconstruction_head.coarse_dataset import load_bev_grid_geometry
from .annotations import DEFAULT_VOD_CLASSES, VODAnnotationLoader
from .detector import (
    BEVDetectorConfig,
    LightweightBEVDetector,
    decode_detections,
    detector_loss,
    make_detection_targets,
)
from .metrics import evaluate_detection_conditions


class CleanDetectionDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        annotations: VODAnnotationLoader,
        *,
        data_root: Path,
        pointpillar_feature_cache_root: Path | None = None,
    ):
        self.paths = tuple(paths)
        self.annotations = annotations
        self.data_root = data_root
        self.pointpillar_feature_cache_root = pointpillar_feature_cache_root

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        path = self.paths[index]
        with np.load(path, allow_pickle=False) as sample:
            metadata = json.loads(str(sample["metadata_json"]))
            if self.pointpillar_feature_cache_root is None:
                clean = np.asarray(sample["clean_rgb"], dtype=np.float32).transpose(2, 0, 1) / 255.0
            else:
                cache_path = (
                    self.pointpillar_feature_cache_root
                    / path.relative_to(self.data_root)
                )
                if not cache_path.is_file():
                    raise FileNotFoundError(
                        f"Missing PointPillars feature cache: {cache_path}"
                    )
                with np.load(cache_path, allow_pickle=False) as cached:
                    clean = np.asarray(cached["clean_features"], dtype=np.float32)
        frame_id = str(metadata["frame_id"])
        return {
            "clean_bev": torch.from_numpy(clean),
            "frame_id": frame_id,
            "boxes": self.annotations.load(frame_id),
        }


def _collate(batch: list[dict]) -> dict:
    return {
        "clean_bev": torch.stack([item["clean_bev"] for item in batch]),
        "frame_id": [item["frame_id"] for item in batch],
        "boxes": [item["boxes"] for item in batch],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--vod-root", required=True, type=Path)
    parser.add_argument("--label-root", type=Path)
    parser.add_argument("--pointpillar-feature-cache-root", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=5.0e-3)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--output-stride", type=int, default=2)
    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.1)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train-samples", type=int)
    parser.add_argument("--limit-val-samples", type=int)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def _frame_ids(paths: list[Path]) -> list[str]:
    return [str(load_sample_metadata(path)["frame_id"]) for path in paths]


@torch.inference_mode()
def _validate(model, loader, annotations, geometry, config, device, use_amp):
    model.eval()
    records = []
    losses = []
    for batch in loader:
        bev = batch["clean_bev"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(bev)
            targets = make_detection_targets(
                batch["boxes"], model.class_names, geometry, outputs["heatmap_logits"].shape[-2:],
                config.output_stride, device=device,
            )
            losses.append(float(detector_loss(outputs, targets)["loss"]))
        predictions = decode_detections(outputs, model.class_names, geometry, config)
        for frame_id, gt, predicted in zip(batch["frame_id"], batch["boxes"], predictions):
            records.append(
                {
                    "frame_id": frame_id,
                    "ground_truth": gt,
                    "predictions": {condition: predicted for condition in ("clean", "faulty", "coarse", "fine")},
                }
            )
    summary, _, _ = evaluate_detection_conditions(records, model.class_names, config.match_iou_threshold)
    return float(np.mean(losses)), summary["conditions"]["clean"]


def main() -> None:
    args = _parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("epochs/batch size must be positive and workers non-negative")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    train_paths = _split_paths(args.data_root, "train", args.limit_train_samples, args.seed)
    val_paths = _split_paths(args.data_root, "val", args.limit_val_samples, args.seed)
    geometry = load_bev_grid_geometry(train_paths[0])
    annotations = VODAnnotationLoader(args.vod_root, geometry, label_root=args.label_root)
    annotations.validate_split(_frame_ids(train_paths))
    annotations.validate_split(_frame_ids(val_paths))
    dataset_options = {
        "data_root": args.data_root,
        "pointpillar_feature_cache_root": args.pointpillar_feature_cache_root,
    }
    train_dataset = CleanDetectionDataset(train_paths, annotations, **dataset_options)
    val_dataset = CleanDetectionDataset(val_paths, annotations, **dataset_options)
    loader_options = dict(
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=_collate,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator, **loader_options
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **loader_options)
    config = BEVDetectorConfig(
        input_channels=int(train_dataset[0]["clean_bev"].shape[0]),
        base_channels=args.base_channels,
        output_stride=args.output_stride,
        score_threshold=args.score_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        match_iou_threshold=args.match_iou_threshold,
    )
    model = LightweightBEVDetector(DEFAULT_VOD_CLASSES, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not args.no_amp)
    use_amp = scaler.is_enabled()
    args.output_root.mkdir(parents=True, exist_ok=True)
    history = []
    best_map = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "heatmap_loss": 0.0, "regression_loss": 0.0}
        for batch in tqdm(train_loader, desc=f"detector epoch {epoch:03d}/{args.epochs:03d}", dynamic_ncols=True):
            bev = batch["clean_bev"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(bev)
                targets = make_detection_targets(
                    batch["boxes"], model.class_names, geometry,
                    outputs["heatmap_logits"].shape[-2:], config.output_stride, device=device,
                )
                losses = detector_loss(outputs, targets)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            for key in totals:
                totals[key] += float(losses[key].detach())
        val_loss, val_metrics = _validate(model, val_loader, annotations, geometry, config, device, use_amp)
        row = {
            "epoch": epoch,
            **{f"train/{key}": value / len(train_loader) for key, value in totals.items()},
            "val/loss": val_loss,
            "val/map": val_metrics["map"],
            "val/precision": val_metrics["precision"],
            "val/recall": val_metrics["recall"],
            "val/f1": val_metrics["f1"],
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}: train/loss={row['train/loss']:.5f} "
            f"val/loss={val_loss:.5f} val/mAP={100*val_metrics['map']:.2f}% "
            f"P={100*val_metrics['precision']:.2f}% R={100*val_metrics['recall']:.2f}% "
            f"F1={100*val_metrics['f1']:.2f}%"
        )
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "detector_config": config.to_dict(),
            "class_names": model.class_names,
            "grid_geometry": geometry.to_dict(),
            "epoch": epoch,
            "validation_metrics": val_metrics,
            "training_split": "train_clean_only",
            "model_selection_split": "val_clean_only",
            "input_representation": (
                "post_pillar_scatter_features"
                if args.pointpillar_feature_cache_root is not None
                else "physical_lidar_bev"
            ),
        }
        atomic_torch_save(checkpoint, args.output_root / "last_checkpoint.pt")
        if val_metrics["map"] > best_map:
            best_map = val_metrics["map"]
            atomic_torch_save(checkpoint, args.output_root / "best_model.pt")
        write_csv_rows(args.output_root / "history.csv", history)
    atomic_write_json(
        args.output_root / "training_summary.json",
        {
            "best_validation_map": best_map,
            "classes": list(model.class_names),
            "detector_config": config.to_dict(),
            "grid_geometry": geometry.to_dict(),
            "train_samples": len(train_dataset),
            "validation_samples": len(val_dataset),
            "test_annotations_used": False,
            "input_representation": (
                "post_pillar_scatter_features"
                if args.pointpillar_feature_cache_root is not None
                else "physical_lidar_bev"
            ),
        },
    )


if __name__ == "__main__":
    main()
