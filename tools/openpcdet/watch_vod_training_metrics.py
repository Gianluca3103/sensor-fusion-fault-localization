"""Evaluate new VoD OpenPCDet checkpoints and print a compact metric table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time


EPOCH_PATTERN = re.compile(r"checkpoint_epoch_(\d+)\.pth$")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpcdet-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    return parser.parse_args()


def _epoch(path: Path) -> int:
    match = EPOCH_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Not an epoch checkpoint: {path}")
    return int(match.group(1))


def _metric_values(metrics: dict, kind: str) -> list[float]:
    return [
        float(value)
        for key, value in metrics.items()
        if kind in key.lower()
        and "moderate" in key.lower()
        and "r40" in key.lower()
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    ]


def _mean_metric(metrics: dict, kind: str) -> float:
    values = _metric_values(metrics, kind)
    return sum(values) / len(values) if values else float("nan")


def _class_metric(metrics: dict, class_name: str, kind: str) -> float:
    values = [
        float(value)
        for key, value in metrics.items()
        if class_name.lower() in key.lower()
        and kind in key.lower()
        and "moderate" in key.lower()
        and "r40" in key.lower()
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    ]
    return sum(values) / len(values) if values else float("nan")


def _row(epoch: int, metrics: dict) -> dict[str, float | int]:
    return {
        "epoch": epoch,
        "bev_map_r40": _mean_metric(metrics, "bev"),
        "map_3d_r40": _mean_metric(metrics, "3d"),
        "car_3d_ap_r40": _class_metric(metrics, "car", "3d"),
        "pedestrian_3d_ap_r40": _class_metric(metrics, "pedestrian", "3d"),
        "cyclist_3d_ap_r40": _class_metric(metrics, "cyclist", "3d"),
    }


def _completed_epochs(csv_path: Path) -> set[int]:
    if not csv_path.is_file():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return {int(row["epoch"]) for row in csv.DictReader(handle)}


def _append_csv(csv_path: Path, row: dict[str, float | int]) -> None:
    exists = csv_path.is_file()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _checkpoint_is_stable(path: Path, previous_sizes: dict[Path, int]) -> bool:
    size = path.stat().st_size
    previous = previous_sizes.get(path)
    previous_sizes[path] = size
    return size > 0 and previous == size


def _evaluate(args: argparse.Namespace, checkpoint: Path, epoch: int) -> dict:
    epoch_root = args.output_root / f"epoch_{epoch:03d}"
    epoch_root.mkdir(parents=True, exist_ok=True)
    checkpoint_link = epoch_root / checkpoint.name
    if not checkpoint_link.exists():
        checkpoint_link.symlink_to(checkpoint.resolve())

    command = [
        sys.executable,
        str(Path(__file__).with_name("train_vod_pvrcnnpp.py")),
        "--openpcdet-root",
        str(args.openpcdet_root),
        "--config",
        str(args.config),
        "--data-root",
        str(args.data_root),
        "--output-root",
        str(epoch_root),
        "--checkpoint-dir",
        str(epoch_root),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--epochs",
        str(args.epochs),
        "--extra-tag",
        f"metric_epoch_{epoch:03d}",
        "--skip-training",
    ]
    log_path = epoch_root / "validation.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    summary_path = epoch_root / "checkpoint_selection.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary["checkpoints"][0]["metrics"]


def main() -> None:
    args = _arguments()
    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "metrics_by_epoch.csv"
    completed = _completed_epochs(csv_path)
    previous_sizes: dict[Path, int] = {}

    print(
        f"{'Epoch':>5} | {'BEV mAP':>8} | {'3D mAP':>8} | "
        f"{'Car 3D':>8} | {'Ped 3D':>8} | {'Cyc 3D':>8}",
        flush=True,
    )
    print("-" * 66, flush=True)

    while len(completed) < args.epochs:
        checkpoints = sorted(
            args.checkpoint_dir.glob("checkpoint_epoch_*.pth"), key=_epoch
        )
        pending = [path for path in checkpoints if _epoch(path) not in completed]
        progressed = False
        for checkpoint in pending:
            epoch = _epoch(checkpoint)
            if not _checkpoint_is_stable(checkpoint, previous_sizes):
                continue
            metrics = _evaluate(args, checkpoint, epoch)
            row = _row(epoch, metrics)
            _append_csv(csv_path, row)
            completed.add(epoch)
            progressed = True
            print(
                f"{epoch:5d} | {row['bev_map_r40']:8.3f} | "
                f"{row['map_3d_r40']:8.3f} | {row['car_3d_ap_r40']:8.3f} | "
                f"{row['pedestrian_3d_ap_r40']:8.3f} | "
                f"{row['cyclist_3d_ap_r40']:8.3f}",
                flush=True,
            )
        if len(completed) >= args.epochs:
            break
        if not progressed:
            time.sleep(args.poll_seconds)

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    best = max(rows, key=lambda item: float(item["map_3d_r40"]))
    print(
        f"Best epoch: {best['epoch']} | BEV mAP {float(best['bev_map_r40']):.3f}% "
        f"| 3D mAP {float(best['map_3d_r40']):.3f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
