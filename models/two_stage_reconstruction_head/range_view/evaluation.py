"""Generated XYZ evaluation with fault-wise and threshold-wise breakdowns."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from .data import load_range_sample
from .geometry import RangeGeometry, project_lidar
from .merge import MergeConfig, merge_reconstruction
from .metrics import _point_set_scores, evaluate_xyz


def evaluate_range_model(
    model: torch.nn.Module,
    paths: list[Path],
    radar_root: Path,
    geometry: RangeGeometry,
    *,
    device: torch.device,
    merge_config: MergeConfig,
    fault_map_root: Path | None = None,
    distance_m: float = 0.2,
    compute_chamfer: bool = True,
    output_path: Path | None = None,
    visualization_root: Path | None = None,
    visualization_limit: int = 0,
    radar_floor_band_m: float = 0.0,
    filter_radar_by_lidar_min: bool = True,
) -> dict:
    model.eval()
    rows = []
    for index, path in enumerate(paths):
        sample = load_range_sample(path, radar_root, geometry, fault_map_root=fault_map_root,
                                   forward_only=merge_config.forward_only,
                                   radar_floor_band_m=radar_floor_band_m,
                                   filter_radar_by_lidar_min=filter_radar_by_lidar_min,
                                   use_ray_encoding=bool(getattr(model.config, "use_ray_encoding", False)))
        with torch.inference_mode():
            prediction = model(torch.from_numpy(sample.features)[None].to(device))
        add_probability = prediction["add_probability"][0].cpu().numpy()
        add_range = prediction["add_range_m"][0].cpu().numpy()
        delete_probability = prediction["delete_probability"][0].cpu().numpy()
        merged = merge_reconstruction(
            sample.faulty_points, sample.faulty_projection, geometry,
            add_probability, add_range, delete_probability,
            config=merge_config, radar_support=sample.radar_features[0],
            radar_points=sample.radar_points,
            add_intensity=(prediction["add_intensity"][0].cpu().numpy()
                           if "add_intensity" in prediction else None),
        )
        fault = str(sample.metadata.get("fault", "unknown"))
        record = {"sample": str(path), "fault": fault, "targets": sample.targets.counts()}
        record["radar_floor_removed_points"] = sample.metadata["radar_floor_removed_points"]
        record["radar_below_lidar_removed_points"] = sample.metadata["radar_below_lidar_removed_points"]
        record.update(evaluate_xyz(sample, merged, tolerance_m=distance_m,
                                   compute_chamfer=compute_chamfer))
        if bool(getattr(model.config, "radar_only_geometry", False)):
            # Score the radar translation itself, before the faulty LiDAR
            # fills in most of the scene and masks a weak predictor.
            empty = np.empty((0, sample.faulty_points.shape[1]), dtype=np.float32)
            radar_translation = merge_reconstruction(
                empty, project_lidar(empty, geometry), geometry,
                add_probability, add_range, delete_probability,
                config=merge_config, radar_support=sample.radar_features[0],
                radar_points=sample.radar_points,
                add_intensity=(prediction["add_intensity"][0].cpu().numpy()
                               if "add_intensity" in prediction else None),
            )
            record.update(_point_set_scores(
                radar_translation.points, sample.clean_points, distance_m,
                "radar_translation", compute_chamfer=compute_chamfer,
            ))
        rows.append(record)
        if visualization_root is not None and index < visualization_limit:
            from .visualization import save_range_comparison
            visualization_root.mkdir(parents=True, exist_ok=True)
            save_range_comparison(
                visualization_root / f"{path.stem}.png", sample, merged,
                add_probability, add_range, delete_probability, geometry,
            )
    by_fault: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_fault[row["fault"]].append(row)

    def summarize(records: list[dict]) -> dict[str, float]:
        if not records:
            return {}
        keys = [key for key, value in records[0].items() if isinstance(value, (int, float))]
        result = {}
        for key in keys:
            values = np.asarray([record[key] for record in records], dtype=np.float64)
            finite = np.isfinite(values)
            result[key] = float(values[finite].mean()) if finite.any() else float("nan")
        return result

    def target_counts(records: list[dict]) -> dict[str, int]:
        return {key: sum(int(row["targets"][key]) for row in records)
                for key in ("keep", "add", "delete", "replace", "delete_valid")}

    summary = {
        "count": len(rows),
        "chamfer_evaluated": compute_chamfer,
        "merge_config": vars(merge_config),
        "radar_floor_band_m": radar_floor_band_m,
        "filter_radar_by_lidar_min": filter_radar_by_lidar_min,
        "overall_macro": summarize(rows),
        "overall_target_counts": target_counts(rows),
        "by_fault_macro": {fault: {"count": len(group), "target_counts": target_counts(group), **summarize(group)}
                           for fault, group in sorted(by_fault.items())},
        "samples": rows,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    return summary
