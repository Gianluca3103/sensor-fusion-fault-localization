"""Generate random HeRCULES raw-fault and 3D-box inspection samples.

This is a resumable diagnostic orchestrator, not a training-data generator.
It cycles a deterministic fault plan over unique randomly selected frames and
produces the same provenance-filtered artifacts, 3D ground truth, oracle
selector, and LiDAR-only non-overlapping bounding-box view used for VoD.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import random
import subprocess
import sys

from Fault_Localization_Model.fault_injector import parse_fault_plan
from Fault_Localization_Model.hercules_dataset import discover_hercules_frames
from Fault_Localization_Model.io_utils import atomic_write_json


DEFAULT_FAULT_PLAN = ("fog_sim:4", "fog_sim:5", "fov_filter:1", "total_loss:1")


def _selected_indices(frame_count: int, sample_count: int, seed: int) -> list[int]:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if sample_count > frame_count:
        raise ValueError(
            f"Requested {sample_count} unique frames from only {frame_count} available frames"
        )
    return random.Random(seed).sample(range(frame_count), sample_count)


def _run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(
            f"Diagnostic stage failed with exit code {result.returncode}; see {log_path}"
        )


def _process_sample(
    *,
    sample_index: int,
    frame_index: int,
    frame,
    fault: str,
    severity: int,
    injection_seed: int,
    args: argparse.Namespace,
    config: Path,
    raw_root: Path,
    ground_truth_root: Path,
    selector_root: Path,
    view_root: Path,
    log_root: Path,
) -> dict:
    """Generate independent diagnostics for one frame/fault pair.

    Each task writes to a distinct directory and its own log, so tasks can be
    scheduled concurrently without changing selection, fault seeds, or output
    bytes for an individual sample.
    """
    scene = frame.lidar_path.parent.parent.parent.name
    sample_name = f"{scene}_{frame.lidar_path.stem}_{fault}_s{severity}"
    raw_artifact = raw_root / sample_name / "raw_lidar_fault.npz"
    ground_truth = ground_truth_root / sample_name / "fault_ground_truth_3d.npz"
    selector = selector_root / sample_name / "oracle_selector_3d.npz"
    view_directory = view_root / sample_name
    final_image = view_directory / "lidar_removed_points_crop_boxes_3d.png"
    projection_image = view_directory / "lidar_removed_points_crop_boxes_xyz_projections.png"
    log_path = log_root / f"{sample_index:03d}_{sample_name}.log"
    status = (
        "cached"
        if final_image.is_file() and projection_image.is_file() and not args.force
        else "created"
    )
    if status == "created":
        common_manifest = (
            ["--split-manifest", str(args.split_manifest.resolve())]
            if args.split_manifest is not None
            else []
        )
        _run(
            [
                sys.executable, "-u", "-m", "scripts.visualize_hercules_raw_lidar_fault",
                "--hercules-root", str(args.hercules_root.resolve()),
                "--split", args.split,
                "--frame-index", str(frame_index),
                "--fault", fault,
                "--severity", str(severity),
                "--seed", str(injection_seed),
                "--config", str(config),
                "--max-display-points", str(args.max_display_points),
                "--output-root", str(raw_root.resolve()),
                *common_manifest,
            ],
            log_path,
        )
        _run(
            [
                sys.executable, "-u", "-m", "scripts.visualize_3d_fault_ground_truth",
                "--artifact", str(raw_artifact.resolve()),
                "--config", str(config),
                "--movement-tolerance-m", "0.05",
                "--feature-tolerance", "0.0001",
                "--height-slices", "12",
                "--output-root", str((ground_truth_root / sample_name).resolve()),
            ],
            log_path,
        )
        _run(
            [
                sys.executable, "-u", "-m", "scripts.build_visualize_oracle_3d_fault_selector",
                "--ground-truth", str(ground_truth.resolve()),
                "--config", str(config),
                "--halo-m", str(args.halo_m),
                "--grouping-radius-m", str(args.grouping_radius_m),
                "--connectivity", "26",
                "--min-component-voxels", "1",
                "--min-crop-shape-zyx", "4", "10", "10",
                "--height-slices", "12",
                "--output-root", str((selector_root / sample_name).resolve()),
            ],
            log_path,
        )
        _run(
            [
                sys.executable, "-u", "-m", "scripts.visualize_lidar_fault_boxes_3d",
                "--raw-artifact", str(raw_artifact.resolve()),
                "--selector", str(selector.resolve()),
                "--config", str(config),
                "--box-type", "crop",
                "--max-boxes", str(args.max_boxes),
                "--max-lidar-points", str(args.max_display_points),
                "--max-removed-points", str(args.max_display_points),
                "--seed", str(injection_seed),
                "--output-root", str(view_directory.resolve()),
            ],
            log_path,
        )

    selector_summary_path = selector_root / sample_name / "summary.json"
    view_summary_path = view_directory / "lidar_fault_boxes_summary.json"
    selector_summary = json.loads(selector_summary_path.read_text(encoding="utf-8"))
    view_summary = json.loads(view_summary_path.read_text(encoding="utf-8"))
    return {
        "sample_index": sample_index,
        "frame_index": frame_index,
        "scene": scene,
        "timestamp": frame.lidar_path.stem,
        "fault": fault,
        "severity": severity,
        "injection_seed": injection_seed,
        "status": status,
        "removed_points_in_grid": view_summary["removed_clean_lidar_points_in_grid"],
        "repair_voxels": selector_summary["repair_core_voxels"],
        "component_boxes": view_summary["component_boxes_before_merge"],
        "final_nonoverlap_boxes": view_summary["boxes_displayed"],
        "image": str(final_image),
        "projection_image": str(projection_image),
        "raw_artifact": str(raw_artifact),
        "selector_artifact": str(selector),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hercules-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--fault-plan", nargs="+", default=list(DEFAULT_FAULT_PLAN))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/voxelization_3d.json")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--halo-m", type=float, default=1.0)
    parser.add_argument("--grouping-radius-m", type=float, default=0.4)
    parser.add_argument("--max-boxes", type=int, choices=(1, 2), default=2)
    parser.add_argument("--max-display-points", type=int, default=100000)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Independent sample jobs to run concurrently; start with 2 on a shared machine.",
    )
    parser.add_argument("--force", action="store_true",
                        help="Regenerate completed samples instead of resuming")
    args = parser.parse_args()
    plan = parse_fault_plan(args.fault_plan)
    if args.workers < 1:
        parser.error("--workers must be positive")
    config = args.config.resolve()
    if not config.is_file():
        raise FileNotFoundError(f"Voxelization config not found: {config}")
    if args.split_manifest is not None and not args.split_manifest.is_file():
        raise FileNotFoundError(f"Split manifest not found: {args.split_manifest}")

    frames = discover_hercules_frames(
        args.hercules_root,
        args.split,
        radar_variant="raw_lidar_fault_box_inspection",
        split_manifest=args.split_manifest,
    )
    indices = _selected_indices(len(frames), args.num_samples, args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    raw_root = args.output_root / "raw"
    ground_truth_root = args.output_root / "ground_truth"
    selector_root = args.output_root / "selector"
    view_root = args.output_root / "boxed_views"
    log_root = args.output_root / "logs"
    rows = []

    print(
        f"Selected {len(indices)} unique {args.split} frames from {len(frames)}; "
        f"fault plan={args.fault_plan}; max boxes={args.max_boxes}"
    )
    tasks = []
    for sample_index, frame_index in enumerate(indices):
        frame = frames[frame_index]
        fault, severity = plan[sample_index % len(plan)]
        injection_seed = int(args.seed + sample_index)
        tasks.append({
            "sample_index": sample_index,
            "frame_index": frame_index,
            "frame": frame,
            "fault": fault,
            "severity": severity,
            "injection_seed": injection_seed,
            "args": args,
            "config": config,
            "raw_root": raw_root,
            "ground_truth_root": ground_truth_root,
            "selector_root": selector_root,
            "view_root": view_root,
            "log_root": log_root,
        })

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_sample, **task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(
                f"[{completed:02d}/{len(tasks):02d}] {row['scene']} "
                f"{row['timestamp']} | {row['fault']} s{row['severity']} | {row['status']}",
                flush=True,
            )
            atomic_write_json(
                args.output_root / "inspection_manifest.json",
                sorted(rows, key=lambda item: item["sample_index"]),
            )

    rows.sort(key=lambda item: item["sample_index"])

    csv_path = args.output_root / "inspection_manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Completed {len(rows)} samples")
    print(f"Images:   {view_root}")
    print(f"Manifest: {args.output_root / 'inspection_manifest.json'}")


if __name__ == "__main__":
    main()
