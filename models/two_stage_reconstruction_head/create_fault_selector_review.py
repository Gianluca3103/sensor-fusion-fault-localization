"""Create a balanced 100-sample visual review set for the Fault Selector."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .coarse_reconstruction.coarse_config import build_selector_config, load_config
from .fault_selector import FaultSelector
from .visualize_fault_selection import (
    load_fault_selection_sample,
    render_fault_selection,
)


GROUPS = (
    ("fog_s3", "fog_sim", 3, "fog"),
    ("fog_s4", "fog_sim", 4, "fog"),
    ("fog_s5", "fog_sim", 5, "fog"),
    ("fov_s1", "fov_filter", 1, "fog"),
    ("old_laser_s0", "old_laser_degradation", 0, "old_laser"),
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fog-root", required=True)
    parser.add_argument("--old-laser-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--samples-per-group", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[2]
            / "configs"
            / "coarse_reconstruction.json"
        ),
    )
    return parser.parse_args()


def _metadata(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata_json in {path} must decode to an object")
    return metadata


def _candidates(root: Path, fault: str, severity: int) -> list[tuple[Path, dict]]:
    matches = []
    pattern = f"*_{fault}_s{severity}.npz"
    for path in sorted(root.rglob(pattern)):
        metadata = _metadata(path)
        if (
            str(metadata.get("fault")) == fault
            and int(metadata.get("severity", -1)) == severity
        ):
            matches.append((path, metadata))
    return matches


def _blob_text(blobs) -> str:
    return ";".join(
        f"{blob.cell_count}@{blob.nearest_distance_m:.2f}m"
        for blob in blobs
    )


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_contact_sheet(group_name, entries, output_path, selector):
    columns = 4
    rows = int(np.ceil(len(entries) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(16, 4 * rows))
    axes = np.asarray(axes, dtype=object).reshape(-1)
    for axis, entry in zip(axes, entries):
        _clean, faulty, _reliability, heatmap, evidence = (
            load_fault_selection_sample(entry["sample_path"])
        )
        selection = selector.select(heatmap, **evidence)
        axis.imshow(faulty, interpolation="nearest")
        overlay = np.ma.masked_where(
            ~selection.reconstruction_mask,
            selection.reconstruction_mask.astype(np.float32),
        )
        axis.imshow(
            overlay,
            cmap="autumn",
            vmin=0.0,
            vmax=1.0,
            alpha=0.2,
            interpolation="nearest",
        )
        healthy_overlay = np.ma.masked_where(
            ~selection.healthy_context_mask,
            selection.healthy_context_mask.astype(np.float32),
        )
        axis.imshow(
            healthy_overlay,
            cmap="Greens",
            vmin=0.0,
            vmax=1.0,
            alpha=0.9,
            interpolation="nearest",
        )
        axis.set_title(
            f"{entry['sample_index']:02d}: "
            f"repair={selection.selected_cell_count}, halo={selection.halo_cell_count}",
            fontsize=9,
        )
        axis.axis("off")
    for axis in axes[len(entries):]:
        axis.axis("off")
    figure.suptitle(f"Fault Selector review: {group_name}", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def main():
    args = _parse_args()
    if args.samples_per_group < 1:
        raise ValueError("samples-per-group must be positive")
    output_root = Path(args.output_root)
    preview_root = output_root / "previews"
    contact_root = output_root / "contact_sheets"
    preview_root.mkdir(parents=True, exist_ok=True)
    contact_root.mkdir(parents=True, exist_ok=True)
    source_roots = {
        "fog": Path(args.fog_root),
        "old_laser": Path(args.old_laser_root),
    }
    selector_config = build_selector_config(load_config(args.config))
    config_path = output_root / "selector_config.json"
    existing_config = None
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            existing_config = json.load(handle)
    refresh_existing_previews = existing_config != selector_config.__dict__
    selector = FaultSelector(selector_config)
    manifest_rows = []

    for group_index, (group_name, fault, severity, source_key) in enumerate(GROUPS):
        candidates = _candidates(source_roots[source_key], fault, severity)
        if len(candidates) < args.samples_per_group:
            raise FileNotFoundError(
                f"Requested {args.samples_per_group} {group_name} samples but found "
                f"only {len(candidates)} under {source_roots[source_key]}"
            )
        rng = random.Random(args.seed + group_index)
        chosen = sorted(rng.sample(candidates, args.samples_per_group), key=lambda item: item[0].name)
        group_entries = []
        group_output = preview_root / group_name
        group_output.mkdir(parents=True, exist_ok=True)
        for sample_index, (sample_path, metadata) in enumerate(chosen, start=1):
            preview_path = group_output / f"{sample_index:02d}_{sample_path.stem}.png"
            if preview_path.exists() and not refresh_existing_previews:
                _clean, _faulty, _reliability, heatmap, evidence = (
                    load_fault_selection_sample(sample_path)
                )
                selection = selector.select(heatmap, **evidence)
            else:
                selection = render_fault_selection(sample_path, preview_path, selector)
            row = {
                "group": group_name,
                "sample_index": sample_index,
                "fault": fault,
                "severity": severity,
                "generator_version": metadata.get("generator_version", ""),
                "ground_truth_method": metadata.get("ground_truth_method", ""),
                "timestamp": metadata.get("timestamp", ""),
                "sample_path": str(sample_path),
                "preview_path": str(preview_path),
                "original_fault_cells": selection.original_fault_cell_count,
                "excluded_added_only_cells": selection.excluded_added_only_cell_count,
                "thresholded_cells": selection.thresholded_cell_count,
                "reconstruction_area_cells": selection.selected_cell_count,
                "halo_area_cells": selection.halo_cell_count,
                "selected_fault_cells": selection.selected_fault_cell_count,
                "healthy_context_cells": selection.healthy_context_cell_count,
                "minimum_repair_fault_fraction": min(
                    (
                        blob.repair_fault_fraction
                        for blob in selection.selected_blobs
                    ),
                    default=0.0,
                ),
                "minimum_halo_healthy_fraction": min(
                    (
                        blob.halo_healthy_fraction
                        for blob in selection.selected_blobs
                    ),
                    default=0.0,
                ),
                "repair_targets_met": all(
                    blob.repair_target_met for blob in selection.selected_blobs
                ),
                "halo_targets_met": all(
                    blob.halo_target_met for blob in selection.selected_blobs
                ),
                "selected_blob_count": len(selection.selected_blobs),
                "qualifying_blob_count": len(selection.qualifying_blobs),
                "rejected_small_blob_count": len(selection.rejected_small_blobs),
                "selected_blobs_cells_at_distance": _blob_text(selection.selected_blobs),
            }
            manifest_rows.append(row)
            group_entries.append(row)
        _render_contact_sheet(
            group_name,
            group_entries,
            contact_root / f"{group_name}.png",
            selector,
        )
        print(f"Rendered {len(group_entries)} previews for {group_name}")

    fieldnames = list(manifest_rows[0])
    _write_csv(output_root / "manifest.csv", manifest_rows, fieldnames)
    summary_rows = []
    for group_name, fault, severity, _source_key in GROUPS:
        rows = [row for row in manifest_rows if row["group"] == group_name]
        summary_rows.append(
            {
                "group": group_name,
                "fault": fault,
                "severity": severity,
                "sample_count": len(rows),
                "mean_thresholded_cells": f"{np.mean([row['thresholded_cells'] for row in rows]):.3f}",
                "mean_excluded_added_only_cells": f"{np.mean([row['excluded_added_only_cells'] for row in rows]):.3f}",
                "mean_reconstruction_area_cells": f"{np.mean([row['reconstruction_area_cells'] for row in rows]):.3f}",
                "mean_halo_area_cells": f"{np.mean([row['halo_area_cells'] for row in rows]):.3f}",
                "mean_selected_fault_cells": f"{np.mean([row['selected_fault_cells'] for row in rows]):.3f}",
                "mean_healthy_context_cells": f"{np.mean([row['healthy_context_cells'] for row in rows]):.3f}",
                "mean_fault_coverage": f"{np.mean([row['selected_fault_cells'] / max(row['thresholded_cells'], 1) for row in rows]):.6f}",
                "mean_rejected_small_blobs": f"{np.mean([row['rejected_small_blob_count'] for row in rows]):.3f}",
            }
        )
    _write_csv(output_root / "summary.csv", summary_rows, list(summary_rows[0]))
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(selector_config.__dict__, handle, indent=2)
    print(f"Created {len(manifest_rows)}-sample review set at: {output_root}")


if __name__ == "__main__":
    main()
