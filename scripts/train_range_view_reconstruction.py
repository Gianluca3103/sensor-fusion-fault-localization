"""Train deterministic, original-preserving range-view reconstruction."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.two_stage_reconstruction_head.range_view.data import RangeViewDataset
from models.two_stage_reconstruction_head.range_view.evaluation import evaluate_range_model
from models.two_stage_reconstruction_head.range_view.geometry import RangeGeometry
from models.two_stage_reconstruction_head.range_view.loss import RangeLossConfig, range_edit_loss
from models.two_stage_reconstruction_head.range_view.merge import MergeConfig
from models.two_stage_reconstruction_head.range_view.model import RangeModelConfig, RangeViewReconstructor


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--radar-root", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True,
                        help="Sensor beam elevations, azimuth bins and physical range bounds JSON")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fault-map-root", type=Path)
    parser.add_argument("--use-fault-map-conditioning", action="store_true")
    parser.add_argument("--no-radar", action="store_true")
    parser.add_argument("--include-rear", action="store_true",
                        help="Experimental full-azimuth mode; default uses x >= 0 LiDAR/radar only")
    parser.add_argument("--allow-original-deletion", action="store_true")
    parser.add_argument("--delete-threshold", type=float, default=0.999)
    parser.add_argument("--add-threshold", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-channels", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lambda-add", type=float, default=1)
    parser.add_argument("--lambda-range", type=float, default=1)
    parser.add_argument("--lambda-delete", type=float, default=1)
    parser.add_argument("--lambda-geometry", type=float, default=0)
    parser.add_argument("--lambda-free-space", type=float, default=0.1)
    parser.add_argument("--add-positive-weight", type=float, default=10)
    parser.add_argument("--false-delete-penalty", type=float, default=30)
    parser.add_argument("--missed-delete-penalty", type=float, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.use_fault_map_conditioning and args.fault_map_root is None:
        parser.error("--use-fault-map-conditioning requires independent --fault-map-root")
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or args.learning_rate <= 0:
        parser.error("invalid training settings")
    return args


def _paths(root: Path, split: str, limit: int | None) -> list[Path]:
    result = sorted((root / split).glob("*.npz"))
    if limit is not None:
        result = result[:limit]
    if not result:
        raise FileNotFoundError(f"No reconstruction artifacts in {root / split}")
    return result


def main() -> None:
    args = _arguments()
    geometry = RangeGeometry.from_json(args.geometry)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_config = RangeModelConfig(
        hidden_channels=args.hidden_channels, min_range_m=geometry.min_range_m,
        max_range_m=geometry.max_range_m, use_radar=not args.no_radar,
        use_fault_map_conditioning=args.use_fault_map_conditioning,
        circular_azimuth=geometry.azimuth_span_rad >= 2 * np.pi - 1e-8,
    )
    loss_config = RangeLossConfig(
        lambda_add=args.lambda_add, lambda_range=args.lambda_range,
        lambda_delete=args.lambda_delete, lambda_geometry=args.lambda_geometry,
        lambda_free_space=args.lambda_free_space,
        add_positive_weight=args.add_positive_weight,
        false_delete_penalty=args.false_delete_penalty,
        missed_delete_penalty=args.missed_delete_penalty,
    )
    merge_config = MergeConfig(
        allow_original_deletion=args.allow_original_deletion,
        delete_threshold=args.delete_threshold, add_threshold=args.add_threshold,
        forward_only=not args.include_rear,
    )
    train_paths = _paths(args.data_root, "train", args.train_limit)
    val_paths = _paths(args.data_root, "val", args.val_limit)
    fault_root = args.fault_map_root if args.use_fault_map_conditioning else None
    dataset = RangeViewDataset(train_paths, args.radar_root, geometry, fault_map_root=fault_root,
                               forward_only=merge_config.forward_only)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = RangeViewReconstructor(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "resolved_config.json").write_text(json.dumps({
        "representation": "range_view", "reconstruction_mode": "additive",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "geometry": asdict(geometry), "model": asdict(model_config),
        "loss": asdict(loss_config), "merge": asdict(merge_config),
    }, indent=2), encoding="utf-8")
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.perf_counter()
        totals: dict[str, float] = {}
        batches = 0
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            prediction = model(batch["features"])
            losses = range_edit_loss(prediction, batch, loss_config)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach())
            batches += 1
            if batches % 25 == 0:
                print(f"epoch={epoch} batch={batches}/{len(loader)} loss={totals['loss']/batches:.4f}", flush=True)
        validation = evaluate_range_model(
            model, val_paths, args.radar_root, geometry, device=device,
            merge_config=merge_config, fault_map_root=fault_root,
            output_path=args.output_root / f"val_epoch_{epoch}.json",
            visualization_root=args.output_root / "visualizations" / f"epoch_{epoch}",
            visualization_limit=3,
        )
        record = {"epoch": epoch, "seconds": time.perf_counter() - start,
                  "train": {key: value / batches for key, value in totals.items()},
                  "val_overall": validation["overall_macro"],
                  "val_by_fault": validation["by_fault_macro"]}
        with (args.output_root / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "model_config": asdict(model_config), "geometry": asdict(geometry),
                    "loss_config": asdict(loss_config), "merge_config": asdict(merge_config),
                    "representation": "range_view"}, args.output_root / "last_checkpoint.pt")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
