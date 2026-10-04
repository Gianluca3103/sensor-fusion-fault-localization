"""Summarize matched clean/faulty/reconstructed PV-RCNN KITTI 3D AP_R40."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import pickle

from pcdet.datasets.kitti.kitti_object_eval_python.eval import get_official_eval_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-infos", type=Path, required=True)
    parser.add_argument("--clean-results", type=Path, required=True)
    parser.add_argument("--faulty-results", type=Path, required=True)
    parser.add_argument("--reconstructed-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.clean_infos.open("rb") as handle:
        infos = pickle.load(handle)
    expected_ids = [str(info["point_cloud"]["lidar_idx"]) for info in infos]
    ground_truth = [info["annos"] for info in infos]
    rows = []
    for condition, result_path in (
        ("clean", args.clean_results), ("faulty", args.faulty_results),
        ("reconstructed", args.reconstructed_results),
    ):
        with result_path.open("rb") as handle:
            predictions = pickle.load(handle)
        actual_ids = [str(item["frame_id"]) for item in predictions]
        if actual_ids != expected_ids:
            raise ValueError(f"{condition} predictions do not match ordered validation IDs")
        _, metrics = get_official_eval_result(
            ground_truth, predictions, ["Car", "Pedestrian", "Cyclist"])
        row = {"condition": condition, "frames": len(expected_ids)}
        for name in ("Car", "Pedestrian", "Cyclist"):
            for difficulty in ("easy", "moderate", "hard"):
                key = f"{name}_3d/{difficulty}_R40"
                row[key] = float(metrics[key])
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['condition']:13} " + "  ".join(
            f"{name}={row[f'{name}_3d/moderate_R40']:.2f}"
            for name in ("Car", "Pedestrian", "Cyclist")))
    print(f"Saved KITTI 3D AP_R40 comparison: {args.output}")


if __name__ == "__main__":
    main()
