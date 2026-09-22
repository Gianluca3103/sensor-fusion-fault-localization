"""Generate random View-of-Delft 3D fault-box inspection samples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from Fault_Localization_Model.fault_injector import parse_fault_plan
from Fault_Localization_Model.io_utils import atomic_write_json
from Fault_Localization_Model.vod_dataset.vod_io import (
    load_vod_split_ids,
    resolve_vod_public_root,
)
from scripts.generate_hercules_fault_box_inspection import (
    DEFAULT_FAULT_PLAN,
    _run,
    _selected_indices,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "val", "test", "train_val"), default="test"
    )
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
        "--force", action="store_true",
        help="Regenerate completed samples instead of resuming",
    )
    args = parser.parse_args()

    plan = parse_fault_plan(args.fault_plan)
    config = args.config.resolve()
    if not config.is_file():
        raise FileNotFoundError(f"Voxelization config not found: {config}")
    public_root = resolve_vod_public_root(args.vod_root)
    frame_ids = load_vod_split_ids(public_root, args.split)
    indices = _selected_indices(len(frame_ids), args.num_samples, args.seed)

    args.output_root.mkdir(parents=True, exist_ok=True)
    raw_root = args.output_root / "raw"
    ground_truth_root = args.output_root / "ground_truth"
    selector_root = args.output_root / "selector"
    view_root = args.output_root / "boxed_views"
    log_root = args.output_root / "logs"
    rows: list[dict[str, object]] = []

    print(
        f"Selected {len(indices)} unique {args.split} frames from {len(frame_ids)}; "
        f"fault plan={args.fault_plan}; max boxes={args.max_boxes}"
    )
    for sample_index, frame_index in enumerate(indices):
        frame_id = frame_ids[frame_index]
        fault, severity = plan[sample_index % len(plan)]
        injection_seed = args.seed + sample_index
        sample_name = f"{frame_id}_{fault}_s{severity}"
        raw_artifact = raw_root / sample_name / "raw_lidar_fault.npz"
        ground_truth = ground_truth_root / sample_name / "fault_ground_truth_3d.npz"
        selector = selector_root / sample_name / "oracle_selector_3d.npz"
        view_directory = view_root / sample_name
        final_image = view_directory / "lidar_removed_points_crop_boxes_3d.png"
        projection_image = (
            view_directory / "lidar_removed_points_crop_boxes_xyz_projections.png"
        )
        log_path = log_root / f"{sample_index:03d}_{sample_name}.log"
        status = (
            "cached"
            if final_image.is_file() and projection_image.is_file() and not args.force
            else "created"
        )

        print(
            f"[{sample_index + 1:02d}/{len(indices):02d}] {frame_id} | "
            f"{fault} s{severity} | {status}",
            flush=True,
        )
        if status == "created":
            _run(
                [
                    sys.executable, "-u", "-m", "scripts.visualize_vod_raw_lidar_fault",
                    "--vod-root", str(args.vod_root.resolve()),
                    "--split", args.split,
                    "--frame-id", frame_id,
                    "--fault", fault,
                    "--severity", str(severity),
                    "--seed", str(injection_seed),
                    "--config", str(config),
                    "--max-display-points", str(args.max_display_points),
                    "--output-root", str(raw_root.resolve()),
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

        selector_summary = json.loads(
            (selector_root / sample_name / "summary.json").read_text(encoding="utf-8")
        )
        view_summary = json.loads(
            (view_directory / "lidar_fault_boxes_summary.json").read_text(encoding="utf-8")
        )
        rows.append(
            {
                "sample_index": sample_index,
                "frame_index": frame_index,
                "frame_id": frame_id,
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
        )
        atomic_write_json(args.output_root / "inspection_manifest.json", rows)

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
