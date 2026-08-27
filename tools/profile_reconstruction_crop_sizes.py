"""Profile exact repair+halo crop dimensions used by Fine Diffusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Fault_Localization_Model.io_utils import atomic_write_json
from models.Fault_Localization.training_utils import _split_paths
from models.two_stage_reconstruction_head import build_selector_config
from models.two_stage_reconstruction_head.diffusion_process.local_diffusion import (
    ReconstructionCropExtractor,
)
from models.two_stage_reconstruction_head.fault_selector_cache import (
    load_selector_cache,
    selector_cache_path,
)


STATISTICS = (
    "minimum",
    "p01",
    "p05",
    "median",
    "mean",
    "p95",
    "maximum",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--selector-config",
        required=True,
        type=Path,
        help="Configuration whose fault_selector section identifies the cache.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val", "test"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def _distribution(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {name: 0 for name in STATISTICS}
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": int(array.min()),
        "p01": float(np.percentile(array, 1)),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p95": float(np.percentile(array, 95)),
        "maximum": int(array.max()),
    }


def _crop_dimensions(
    reconstruction_mask: np.ndarray,
    halo_mask: np.ndarray,
) -> tuple[int, int] | None:
    crop_mask = np.maximum(reconstruction_mask, halo_mask)
    extent = ReconstructionCropExtractor._extent(
        torch.from_numpy(crop_mask) > 0.5
    )
    if extent is None:
        return None
    top, bottom, left, right = extent
    return bottom - top, right - left


def _summarize(
    heights: list[int],
    widths: list[int],
    *,
    total_samples: int,
) -> dict[str, object]:
    areas = [height * width for height, width in zip(heights, widths)]
    active_samples = len(heights)
    return {
        "total_samples": total_samples,
        "active_samples": active_samples,
        "empty_samples": total_samples - active_samples,
        "active_fraction": active_samples / max(total_samples, 1),
        "crop_height_cells": _distribution(heights),
        "crop_width_cells": _distribution(widths),
        "crop_area_cells": _distribution(areas),
    }


def _print_summary(label: str, summary: dict[str, object]) -> None:
    print()
    print(
        f"{label}: {summary['active_samples']}/{summary['total_samples']} "
        f"active; {summary['empty_samples']} empty"
    )
    print(
        f"{'measurement':<22} {'min':>9} {'p01':>9} {'p05':>9} "
        f"{'median':>9} {'mean':>9} {'p95':>9} {'max':>9}"
    )
    print("-" * 94)
    for key, name in (
        ("crop_height_cells", "crop height (cells)"),
        ("crop_width_cells", "crop width (cells)"),
        ("crop_area_cells", "crop area (cells)"),
    ):
        values = summary[key]
        print(
            f"{name:<22} {values['minimum']:9.2f} {values['p01']:9.2f} "
            f"{values['p05']:9.2f} {values['median']:9.2f} "
            f"{values['mean']:9.2f} {values['p95']:9.2f} "
            f"{values['maximum']:9.2f}"
        )


def main() -> None:
    args = _parse_args()
    payload = json.loads(args.selector_config.read_text(encoding="utf-8"))
    selector = build_selector_config(payload)
    combined_heights: list[int] = []
    combined_widths: list[int] = []
    per_split: dict[str, dict[str, object]] = {}
    combined_total = 0

    for split in args.splits:
        paths = _split_paths(args.data_root, split, None, 0)
        heights: list[int] = []
        widths: list[int] = []
        for index, sample_path in enumerate(paths, start=1):
            masks = load_selector_cache(
                selector_cache_path(sample_path, args.data_root), selector
            )
            dimensions = _crop_dimensions(
                masks["reconstruction_mask"], masks["halo_mask"]
            )
            if dimensions is not None:
                height, width = dimensions
                heights.append(height)
                widths.append(width)
            if index % 1000 == 0 or index == len(paths):
                print(f"{split}: processed {index}/{len(paths)}", flush=True)
        per_split[split] = _summarize(
            heights, widths, total_samples=len(paths)
        )
        combined_heights.extend(heights)
        combined_widths.extend(widths)
        combined_total += len(paths)

    combined = _summarize(
        combined_heights,
        combined_widths,
        total_samples=combined_total,
    )
    report = {
        "definition": (
            "Unpadded bounding box of reconstruction_mask OR halo_mask; "
            "empty samples are counted but excluded from distributions."
        ),
        "data_root": str(args.data_root),
        "selector_config": str(args.selector_config),
        "splits": list(args.splits),
        "combined": combined,
        "per_split": per_split,
    }
    _print_summary("COMBINED", combined)
    for split in args.splits:
        _print_summary(split.upper(), per_split[split])
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.output, report)
        print(f"\nSaved JSON report to {args.output}")


if __name__ == "__main__":
    main()
