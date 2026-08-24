"""Train residual Stage-I reconstruction on cached post-scatter features."""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from Fault_Localization_Model.io_utils import atomic_torch_save, atomic_write_json, write_csv_rows
from models.Fault_Localization.training_utils import _split_paths, resolve_device, seed_everything
from models.two_stage_reconstruction_head.coarse_reconstruction.hrnet_backbone import HRNetConfig
from models.two_stage_reconstruction_head.coarse_reconstruction.pointpillar_feature_reconstruction import (
    CoarsePointPillarFeatureReconstructor,
    PointPillarFeatureCacheDataset,
    PointPillarFeatureReconstructionConfig,
    pointpillar_feature_reconstruction_loss,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--feature-cache-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "coarse_pointpillar_feature_reconstruction_vod.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--limit-train-samples", type=int)
    parser.add_argument("--limit-val-samples", type=int)
    return parser.parse_args()


def _load_config(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    feature_values = dict(payload["pointpillar_feature_reconstruction"])
    hrnet_values = payload.get("hrnet", {})
    valid_hrnet = {item.name for item in fields(HRNetConfig)}
    feature_values["hrnet"] = HRNetConfig(
        **{key: value for key, value in hrnet_values.items() if key in valid_hrnet}
    )
    config = PointPillarFeatureReconstructionConfig(**feature_values)
    config.validate()
    return payload, config


def _collate(batch: list[dict]) -> dict:
    keys = (
        "clean_features",
        "faulty_features",
        "radar_features",
    )
    return {
        **{key: torch.stack([item[key] for item in batch]) for key in keys},
        "sample_path": [item["sample_path"] for item in batch],
    }


def _feature_metrics(coarse, clean, faulty, epsilon):
    faulty_losses = F.smooth_l1_loss(faulty, clean, reduction="none")
    coarse_losses = F.smooth_l1_loss(coarse, clean, reduction="none")
    changed = (
        (clean - faulty).abs().amax(dim=1, keepdim=True) > epsilon
    ).to(clean.dtype)
    faulty_error = faulty_losses.mean()
    coarse_error = coarse_losses.mean()
    faulty_changed_error = (
        faulty_losses * changed.expand_as(clean)
    ).sum() / changed.expand_as(clean).sum().clamp_min(1.0)
    coarse_changed_error = (
        coarse_losses * changed.expand_as(clean)
    ).sum() / changed.expand_as(clean).sum().clamp_min(1.0)
    cosine_support = (
        clean.float().square().sum(dim=1, keepdim=True) > 1.0e-12
    ).to(clean.dtype)
    cosine_cells = cosine_support.sum().clamp_min(1.0)
    faulty_cos = (
        F.cosine_similarity(faulty.float(), clean.float(), dim=1)[:, None]
        * cosine_support
    ).sum() / cosine_cells
    coarse_cos = (
        F.cosine_similarity(coarse.float(), clean.float(), dim=1)[:, None]
        * cosine_support
    ).sum() / cosine_cells
    improvement = 100.0 * (
        faulty_changed_error - coarse_changed_error
    ) / faulty_changed_error.clamp_min(1.0e-12)
    return (
        faulty_error,
        coarse_error,
        faulty_changed_error,
        coarse_changed_error,
        faulty_cos,
        coarse_cos,
        improvement,
    )


def _run_epoch(model, loader, device, scaler, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {}
    batches = 0
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for batch in tqdm(loader, desc="train" if training else "val", dynamic_ncols=True):
            tensors = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=scaler.is_enabled(),
            ):
                output = model(
                    tensors["faulty_features"],
                    tensors["radar_features"],
                )
                losses = pointpillar_feature_reconstruction_loss(
                    output,
                    tensors["clean_features"],
                    tensors["faulty_features"],
                    model.config,
                )
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError("Feature reconstruction loss is non-finite")
            if training:
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            metrics = _feature_metrics(
                output["coarse_features"],
                tensors["clean_features"],
                tensors["faulty_features"],
                model.config.faulty_cell_epsilon,
            )
            values = {
                **{key: float(value.detach()) for key, value in losses.items()},
                "faulty_smooth_l1": float(metrics[0]),
                "coarse_smooth_l1": float(metrics[1]),
                "faulty_changed_smooth_l1": float(metrics[2]),
                "coarse_changed_smooth_l1": float(metrics[3]),
                "faulty_cosine": float(metrics[4]),
                "coarse_cosine": float(metrics[5]),
                "changed_error_improvement_percent": float(metrics[6]),
            }
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + value
            batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def main() -> None:
    args = _parse_args()
    payload, config = _load_config(args.config)
    manifest_path = args.feature_cache_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing feature-cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lidar_channels, height, width = (
        int(value) for value in manifest["lidar_shape"]
    )
    radar_channels, radar_height, radar_width = (
        int(value) for value in manifest["radar_shape"]
    )
    if (radar_height, radar_width) != (height, width):
        raise ValueError("Cached LiDAR and radar feature grids do not align")
    config = replace(
        config,
        lidar_feature_channels=lidar_channels,
        radar_feature_channels=radar_channels,
        lidar_feature_height=height,
        lidar_feature_width=width,
    )
    config.validate()
    training = payload["training"]
    epochs = args.epochs or int(training["epochs"])
    batch_size = args.batch_size or int(training["batch_size"])
    workers = args.num_workers if args.num_workers is not None else int(training["num_workers"])
    seed_everything(int(training["seed"]))
    device = resolve_device(args.device)
    train_paths = _split_paths(args.data_root, "train", args.limit_train_samples, int(training["seed"]))
    val_paths = _split_paths(args.data_root, "val", args.limit_val_samples, int(training["seed"]))
    train_dataset = PointPillarFeatureCacheDataset(train_paths, args.feature_cache_root, args.data_root)
    val_dataset = PointPillarFeatureCacheDataset(val_paths, args.feature_cache_root, args.data_root)
    options = dict(
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=_collate,
    )
    generator = torch.Generator().manual_seed(int(training["seed"]))
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **options)
    val_loader = DataLoader(val_dataset, shuffle=False, **options)
    model = CoarsePointPillarFeatureReconstructor(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and bool(training["mixed_precision"]),
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    history = []
    best_error = float("inf")
    for epoch in range(1, epochs + 1):
        train_stats = _run_epoch(model, train_loader, device, scaler, optimizer)
        val_stats = _run_epoch(model, val_loader, device, scaler)
        row = {
            "epoch": epoch,
            **{f"train/{key}": value for key, value in train_stats.items()},
            **{f"val/{key}": value for key, value in val_stats.items()},
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}: train/loss={train_stats['loss']:.6f} "
            f"val/loss={val_stats['loss']:.6f} | SmoothL1 "
            f"full {val_stats['faulty_smooth_l1']:.6f} -> "
            f"{val_stats['coarse_smooth_l1']:.6f} | changed "
            f"{val_stats['faulty_changed_smooth_l1']:.6f} -> "
            f"{val_stats['coarse_changed_smooth_l1']:.6f} "
            f"({val_stats['changed_error_improvement_percent']:+.2f}%) | cosine "
            f"{val_stats['faulty_cosine']:.4f} -> {val_stats['coarse_cosine']:.4f} | "
            f"changed_cells={100*val_stats['changed_cell_fraction']:.2f}%"
        )
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": config.to_dict(),
            "training": training,
            "validation": val_stats,
            "feature_cache_root": str(args.feature_cache_root.resolve()),
            "interface": "post_pillar_scatter_dense_features",
        }
        atomic_torch_save(checkpoint, args.output_root / "last_checkpoint.pt")
        if val_stats["loss"] < best_error:
            best_error = val_stats["loss"]
            atomic_torch_save(checkpoint, args.output_root / "best_model.pt")
        write_csv_rows(args.output_root / "history.csv", history)
    atomic_write_json(
        args.output_root / "training_summary.json",
        {
            "best_validation_loss": best_error,
            "model_config": config.to_dict(),
            "train_samples": len(train_dataset),
            "validation_samples": len(val_dataset),
            "fault_selector_used": False,
            "full_grid_write_access": True,
            "model_inputs": ["faulty_lidar_postscatter", "radar_postscatter"],
            "training_target": "clean_lidar_postscatter",
        },
    )


if __name__ == "__main__":
    main()
