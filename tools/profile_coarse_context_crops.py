"""Profile and visualize the coarse HRNet repair/halo context crops."""

from __future__ import annotations

import argparse
import json
import math
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
from models.two_stage_reconstruction_head.fault_selector_cache import (
    load_selector_cache,
    selector_cache_path,
)
from models.two_stage_reconstruction_head.reconstruction_crop import (
    ReconstructionCropExtractor,
)
from tools.profile_reconstruction_crop_sizes import _distribution


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--selector-config", required=True, type=Path)
    parser.add_argument("--minimum-size", type=int, default=80)
    parser.add_argument("--pad-multiple", type=int, default=8)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--visualization-dir", type=Path)
    parser.add_argument("--visualization-count", type=int, default=6)
    return parser.parse_args()


def _summary(records: list[dict[str, object]]) -> dict[str, object]:
    def values(name: str) -> list[int]:
        return [int(record[name]) for record in records]

    return {
        "samples": len(records),
        "source_height_cells": _distribution(values("source_height")),
        "source_width_cells": _distribution(values("source_width")),
        "source_area_cells": _distribution(values("source_area")),
        "context_height_cells": _distribution(values("context_height")),
        "context_width_cells": _distribution(values("context_width")),
        "context_area_cells": _distribution(values("context_area")),
        "padded_height_cells": _distribution(values("padded_height")),
        "padded_width_cells": _distribution(values("padded_width")),
        "deepest_height_cells": _distribution(values("deepest_height")),
        "deepest_width_cells": _distribution(values("deepest_width")),
        "below_10_deepest_height": sum(int(record["deepest_height"]) < 10 for record in records),
        "below_10_deepest_width": sum(int(record["deepest_width"]) < 10 for record in records),
        "below_10_deepest_min_dimension": sum(
            min(int(record["deepest_height"]), int(record["deepest_width"])) < 10
            for record in records
        ),
    }


def _print_distribution(label: str, values: dict[str, float | int]) -> None:
    print(
        f"{label:<25} min={values['minimum']:.2f} p01={values['p01']:.2f} "
        f"p05={values['p05']:.2f} median={values['median']:.2f} "
        f"mean={values['mean']:.2f} p95={values['p95']:.2f} max={values['maximum']:.2f}"
    )


def _visualize(records: list[dict[str, object]], directory: Path, count: int) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    directory.mkdir(parents=True, exist_ok=True)
    median_area = float(np.median([int(item["source_area"]) for item in records]))
    selectors = (
        min(records, key=lambda item: int(item["source_height"])),
        min(records, key=lambda item: int(item["source_width"])),
        min(
            records,
            key=lambda item: (
                not (
                    int(item["source_height"]) < 80
                    and int(item["source_width"]) < 80
                ),
                int(item["source_area"]),
            ),
        ),
        min(records, key=lambda item: abs(int(item["source_area"]) - median_area)),
        max(records, key=lambda item: int(item["source_area"])),
        min(
            records,
            key=lambda item: min(
                int(item["source_box"][0]),
                int(item["source_box"][2]),
                320 - int(item["source_box"][1]),
                320 - int(item["source_box"][3]),
            ),
        ),
    )
    candidates = []
    seen = set()
    for candidate in selectors:
        name = str(candidate["sample"])
        if name not in seen:
            candidates.append(candidate)
            seen.add(name)
    for candidate in sorted(records, key=lambda item: int(item["source_area"])):
        if len(candidates) >= count:
            break
        if str(candidate["sample"]) not in seen:
            candidates.append(candidate)
            seen.add(str(candidate["sample"]))
    for index, record in enumerate(candidates):
        repair = np.asarray(record.pop("repair"))
        halo = np.asarray(record.pop("halo"))
        display = np.zeros((*repair.shape, 3), dtype=np.float32)
        display[..., 0] = repair
        display[..., 1] = halo
        figure, axis = plt.subplots(figsize=(7, 7), dpi=140)
        axis.imshow(display, origin="upper")
        for box, color, label in (
            (record["source_box"], "cyan", "repair+halo bbox"),
            (record["context_box"], "yellow", "80-cell context bbox"),
        ):
            top, bottom, left, right = box
            axis.add_patch(Rectangle(
                (left, top), right - left, bottom - top,
                fill=False, edgecolor=color, linewidth=1.8, label=label,
            ))
        axis.set_xlim(0, repair.shape[1])
        axis.set_ylim(repair.shape[0], 0)
        axis.set_title(
            f"{record['split']}/{Path(str(record['sample'])).name}\n"
            f"source {record['source_height']}x{record['source_width']} -> "
            f"context {record['context_height']}x{record['context_width']} -> "
            f"deepest {record['deepest_height']}x{record['deepest_width']}"
        )
        axis.legend(loc="lower right")
        axis.set_xlabel("BEV column")
        axis.set_ylabel("BEV row")
        figure.tight_layout()
        figure.savefig(directory / f"context_crop_{index + 1:02d}.png")
        plt.close(figure)


def main() -> None:
    args = _args()
    payload = json.loads(args.selector_config.read_text(encoding="utf-8"))
    selector = build_selector_config(payload)
    extractor = ReconstructionCropExtractor(args.pad_multiple, args.minimum_size)
    records: list[dict[str, object]] = []
    empty_samples = 0
    for split in ("train", "val", "test"):
        paths = _split_paths(args.data_root, split, None, 0)
        for index, sample_path in enumerate(paths, start=1):
            masks = load_selector_cache(selector_cache_path(sample_path, args.data_root), selector)
            repair = torch.from_numpy(masks["reconstruction_mask"])
            halo = torch.from_numpy(masks["halo_mask"])
            context_box, source_box, active = extractor._boxes(repair, halo)
            if not active:
                empty_samples += 1
                continue
            top, bottom, left, right = source_box
            context_top, context_bottom, context_left, context_right = context_box
            source_height, source_width = bottom - top, right - left
            context_height = context_bottom - context_top
            context_width = context_right - context_left
            padded_height = math.ceil(context_height / args.pad_multiple) * args.pad_multiple
            padded_width = math.ceil(context_width / args.pad_multiple) * args.pad_multiple
            records.append({
                "split": split,
                "sample": str(sample_path),
                "source_box": source_box,
                "context_box": context_box,
                "source_height": source_height,
                "source_width": source_width,
                "source_area": source_height * source_width,
                "context_height": context_height,
                "context_width": context_width,
                "context_area": context_height * context_width,
                "padded_height": padded_height,
                "padded_width": padded_width,
                "deepest_height": padded_height // 8,
                "deepest_width": padded_width // 8,
                "repair": masks["reconstruction_mask"],
                "halo": masks["halo_mask"],
            })
            if index % 1000 == 0 or index == len(paths):
                print(f"{split}: {index}/{len(paths)}", flush=True)
    summary = _summary(records)
    summary.update({
        "total_samples": len(records) + empty_samples,
        "empty_samples": empty_samples,
        "minimum_context_crop_size": args.minimum_size,
        "technical_pad_multiple": args.pad_multiple,
        "hrnet_deepest_output_stride": 8,
    })
    for key in (
        "source_height_cells", "source_width_cells", "source_area_cells",
        "context_height_cells", "context_width_cells", "context_area_cells",
        "padded_height_cells", "padded_width_cells",
        "deepest_height_cells", "deepest_width_cells",
    ):
        _print_distribution(key, summary[key])
    print(f"Deepest height below 10: {summary['below_10_deepest_height']}")
    print(f"Deepest width below 10:  {summary['below_10_deepest_width']}")
    below = int(summary["below_10_deepest_min_dimension"])
    print(
        f"Samples with minimum deepest dimension below 10: {below} "
        f"({100.0 * below / max(len(records), 1):.6f}%)"
    )
    summary["below_10_deepest_min_dimension_percentage"] = (
        100.0 * below / max(len(records), 1)
    )
    serializable = {key: value for key, value in summary.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, serializable)
    if args.visualization_dir is not None:
        _visualize(records, args.visualization_dir, args.visualization_count)
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
