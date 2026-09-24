"""Train deterministic, original-preserving range-view reconstruction."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.two_stage_reconstruction_head.range_view.data import RangeViewDataset
from models.two_stage_reconstruction_head.range_view.evaluation import evaluate_range_model
from models.two_stage_reconstruction_head.range_view.geometry import RangeGeometry
from models.two_stage_reconstruction_head.range_view.loss import RangeLossConfig, range_edit_loss
from models.two_stage_reconstruction_head.range_view.merge import MergeConfig
from models.two_stage_reconstruction_head.range_view.model import RangeModelConfig, RangeViewReconstructor


SUMMARY_FIELDS = (
    "epoch", "seconds", "train_loss", "train_add_loss", "train_range_loss",
    "train_delete_loss", "train_free_space_loss", "val_faulty_f1_at_0_2m",
    "val_reconstructed_precision_at_0_2m", "val_reconstructed_recall_at_0_2m",
    "val_reconstructed_f1_at_0_2m", "val_reconstructed_iou_at_0_2m",
    "val_net_f1_improvement", "val_addition_precision", "val_addition_recall",
    "val_generated_hallucination_rate", "val_generated_count",
)
FAULT_FIELDS = (
    "epoch", "fault", "count", "faulty_f1_at_0_2m",
    "reconstructed_f1_at_0_2m", "net_f1_improvement",
    "addition_precision", "addition_recall", "generated_count",
)


def _summary_rows(record: dict) -> tuple[dict, list[dict]]:
    train = record["train"]
    overall = record["val_overall"]
    summary = {
        "epoch": record["epoch"], "seconds": record["seconds"],
        "train_loss": train["loss"], "train_add_loss": train["add_loss"],
        "train_range_loss": train["range_loss"],
        "train_delete_loss": train["delete_loss"],
        "train_free_space_loss": train["free_space_loss"],
        "val_faulty_f1_at_0_2m": overall["faulty_f1_at_0.2m"],
        "val_reconstructed_precision_at_0_2m": overall["reconstructed_precision_at_0.2m"],
        "val_reconstructed_recall_at_0_2m": overall["reconstructed_recall_at_0.2m"],
        "val_reconstructed_f1_at_0_2m": overall["reconstructed_f1_at_0.2m"],
        "val_reconstructed_iou_at_0_2m": overall["reconstructed_iou_at_0.2m"],
        "val_net_f1_improvement": overall["net_f1_improvement"],
        "val_addition_precision": overall["addition_precision"],
        "val_addition_recall": overall["addition_recall"],
        "val_generated_hallucination_rate": overall["generated_hallucination_rate"],
        "val_generated_count": overall["generated_count"],
    }
    faults = [
        {
            "epoch": record["epoch"], "fault": fault, "count": metrics["count"],
            "faulty_f1_at_0_2m": metrics["faulty_f1_at_0.2m"],
            "reconstructed_f1_at_0_2m": metrics["reconstructed_f1_at_0.2m"],
            "net_f1_improvement": metrics["net_f1_improvement"],
            "addition_precision": metrics["addition_precision"],
            "addition_recall": metrics["addition_recall"],
            "generated_count": metrics["generated_count"],
        }
        for fault, metrics in sorted(record["val_by_fault"].items())
    ]
    return summary, faults


def _append_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    if not rows:
        return
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def _write_progress(path: Path, *, epoch: int, phase: str, completed: int,
                    total: int, loss: float | None = None) -> None:
    payload = {"epoch": epoch, "phase": phase, "completed_batches": completed,
               "total_batches": total, "running_loss": loss}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(path)


def _validation_paths(paths: list[Path], progress_path: Path, epoch: int):
    for index, path in enumerate(paths, start=1):
        yield path
        if index % 5 == 0 or index == len(paths):
            _write_progress(progress_path, epoch=epoch, phase="validation",
                            completed=index, total=len(paths))


def _format_epoch_summary(summary: dict, faults: list[dict], total_epochs: int) -> str:
    lines = [
        f"Epoch {summary['epoch']}/{total_epochs} | {summary['seconds']:.1f}s",
        "  Train loss: "
        f"{summary['train_loss']:.4f} (add {summary['train_add_loss']:.4f}, "
        f"range {summary['train_range_loss']:.4f}, "
        f"delete {summary['train_delete_loss']:.4f})",
        "  Val F1 @ 0.2m: "
        f"{summary['val_reconstructed_f1_at_0_2m']:.4f} "
        f"(faulty {summary['val_faulty_f1_at_0_2m']:.4f}, "
        f"change {summary['val_net_f1_improvement']:+.4f}); "
        f"precision {summary['val_reconstructed_precision_at_0_2m']:.4f}, "
        f"recall {summary['val_reconstructed_recall_at_0_2m']:.4f}",
        "  Additions: "
        f"precision {summary['val_addition_precision']:.4f}, "
        f"recall {summary['val_addition_recall']:.4f}, "
        f"generated {summary['val_generated_count']:,.0f} per sample",
    ]
    for fault in faults:
        lines.append(
            f"  {fault['fault']} (n={fault['count']}): "
            f"F1 {fault['reconstructed_f1_at_0_2m']:.4f} "
            f"(faulty {fault['faulty_f1_at_0_2m']:.4f}, "
            f"change {fault['net_f1_improvement']:+.4f})"
        )
    return "\n".join(lines)


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
    progress_path = args.output_root / "progress.json"
    show_bars = sys.stderr.isatty()
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.perf_counter()
        totals: dict[str, float] = {}
        batches = 0
        _write_progress(progress_path, epoch=epoch, phase="train", completed=0,
                        total=len(loader))
        with tqdm(loader, desc=f"Epoch {epoch}/{args.epochs} train", unit="batch",
                  dynamic_ncols=True, mininterval=1.0, disable=not show_bars) as train_bar:
            for batch in train_bar:
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
                running_loss = totals["loss"] / batches
                if show_bars and (batches % 5 == 0 or batches == len(loader)):
                    train_bar.set_postfix(loss=f"{running_loss:.4f}")
                if batches % 25 == 0 or batches == len(loader):
                    _write_progress(progress_path, epoch=epoch, phase="train",
                                    completed=batches, total=len(loader), loss=running_loss)
        _write_progress(progress_path, epoch=epoch, phase="validation", completed=0,
                        total=len(val_paths))
        with tqdm(_validation_paths(val_paths, progress_path, epoch), total=len(val_paths),
                  desc=f"Epoch {epoch}/{args.epochs} val", unit="sample",
                  dynamic_ncols=True, mininterval=1.0, disable=not show_bars) as val_bar:
            validation = evaluate_range_model(
                model, val_bar, args.radar_root, geometry, device=device,
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
        summary, faults = _summary_rows(record)
        _append_csv(args.output_root / "summary.csv", SUMMARY_FIELDS, [summary])
        _append_csv(args.output_root / "fault_summary.csv", FAULT_FIELDS, faults)
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "model_config": asdict(model_config), "geometry": asdict(geometry),
                    "loss_config": asdict(loss_config), "merge_config": asdict(merge_config),
                    "representation": "range_view"}, args.output_root / "last_checkpoint.pt")
        _write_progress(progress_path, epoch=epoch, phase="complete", completed=len(loader),
                        total=len(loader), loss=summary["train_loss"])
        print(_format_epoch_summary(summary, faults, args.epochs), flush=True)


if __name__ == "__main__":
    main()
