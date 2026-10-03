"""Show VoD 3D box envelopes and visible object returns on range images.

The rectangles are projections of annotated 3D box corners for inspection.
Only the colored clean first-return cells are object training targets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from models.two_stage_reconstruction_head.range_view.data import load_range_sample
from models.two_stage_reconstruction_head.range_view.geometry import RangeGeometry
from models.two_stage_reconstruction_head.range_view.object_targets import projected_box_rectangles


COLORS = {"Car": "#ff6b4a", "Pedestrian": "#39c5ff", "Cyclist": "#e8c64a", "bicycle": "#e8c64a"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--radar-root", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radar-region-row-radius", type=int, default=8)
    parser.add_argument("--radar-region-col-radius", type=int, default=32)
    args = parser.parse_args()
    geometry = RangeGeometry.from_json(args.geometry)
    sample = load_range_sample(
        args.sample, args.radar_root, geometry, include_object_targets=True,
        radar_region_row_radius=args.radar_region_row_radius,
        radar_region_col_radius=args.radar_region_col_radius,
    )
    clean = np.where(sample.clean_projection.valid, sample.clean_projection.range_m, np.nan)
    faulty = np.where(sample.faulty_projection.valid, sample.faulty_projection.range_m, np.nan)
    scale = float(np.nanpercentile(clean, 99)) if np.isfinite(clean).any() else 1.0
    fig, axes = plt.subplots(3, 1, figsize=(19, 10), sharex=True, sharey=True,
                             layout="constrained")
    for axis, image, title in zip(axes[:2], (clean, faulty),
                                  ("Clean range + visible object returns", "Faulty range + projected 3D boxes")):
        axis.imshow(image, origin="lower", aspect="auto", interpolation="nearest",
                    cmap="gray", vmin=0, vmax=max(scale, 1))
        axis.set_title(title)
        axis.set_ylabel("Elevation row")
    classes = sample.targets.object_class
    for class_id, name in ((1, "Car"), (2, "Pedestrian"), (3, "Cyclist")):
        row, col = np.where(classes == class_id)
        if len(row):
            axes[0].scatter(col, row, s=0.45, c=COLORS[name], rasterized=True,
                            label=f"{name} first returns ({len(row)})")
    for name, row0, col0, row1, col1 in projected_box_rectangles(sample.metadata, geometry):
        for axis in axes[:2]:
            axis.add_patch(Rectangle((col0, row0), max(col1 - col0, 1), max(row1 - row0, 1),
                                     fill=False, edgecolor=COLORS[name], linewidth=0.8))
    axes[0].legend(loc="upper right", markerscale=8)
    axes[2].imshow(sample.targets.radar_region, origin="lower", aspect="auto",
                   interpolation="nearest", cmap="Greens", vmin=0, vmax=1)
    row, col = np.where(sample.radar_features[0] > 0)
    if len(row):
        axes[2].scatter(col, row, s=0.7, c="#ffb84d", rasterized=True, label="Radar returns")
        axes[2].legend(loc="upper right", markerscale=6)
    axes[2].set(title="Radar-supported training region (green) and radar cells (orange)",
                xlabel="Azimuth bin", ylabel="Elevation row")
    axes[2].set_xlim(-0.5, geometry.azimuth_bins - 0.5)
    axes[2].set_ylim(-0.5, geometry.shape[0] - 0.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)
    print(f"Saved {args.output}; boxes={len(projected_box_rectangles(sample.metadata, geometry))}; "
          f"object returns={int((classes > 0).sum())}; radar region cells={int(sample.targets.radar_region.sum())}")


if __name__ == "__main__":
    main()
