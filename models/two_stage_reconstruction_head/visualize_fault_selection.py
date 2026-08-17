"""Create a diagnostic preview of perfect-heatmap blob selection."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np

from Fault_Localization_Model.sample_utils import validate_rgb_array
from .coarse_reconstruction.coarse_config import build_selector_config, load_config
from .fault_selector import FaultSelector
from .fault_selector_cache import load_selector_inputs


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[2]
            / "configs"
            / "coarse_reconstruction.json"
        ),
    )
    return parser.parse_args()


def load_fault_selection_sample(sample_path):
    sample_path = Path(sample_path)
    with np.load(sample_path, allow_pickle=False) as data:
        clean = validate_rgb_array(
            data["clean_rgb"], name="clean_rgb", path=sample_path
        )
        faulty = validate_rgb_array(
            data["faulty_rgb"], name="faulty_rgb", path=sample_path
        )
    evidence = load_selector_inputs(sample_path)
    heatmap = evidence.pop("fault_heatmap")
    reliability = evidence["reliability_map"]
    return clean, faulty, reliability, heatmap, evidence


def _draw_repair_boxes(axis, selection):
    for blob in selection.selected_blobs:
        top, left, bottom, right = blob.bbox
        axis.add_patch(
            Rectangle(
                (left - 0.5, top - 0.5),
                right - left,
                bottom - top,
                fill=False,
                edgecolor="cyan",
                linewidth=1.2,
            )
        )


def _draw_halo_outline(axis, selection):
    if not selection.halo_mask.any():
        return
    axis.contour(
        selection.halo_mask.astype(np.float32),
        levels=[0.5],
        colors=["#00ff66"],
        linewidths=1.2,
        linestyles="dashed",
    )


def render_fault_selection(sample_path, output_path, selector):
    clean, faulty, reliability, heatmap, evidence = load_fault_selection_sample(
        sample_path
    )
    selection = selector.select(heatmap, **evidence)

    figure, axes = plt.subplots(1, 4, figsize=(20, 5))
    reliability_image = axes[0].imshow(
        reliability,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axes[0].set_title("Perfect Reliability Map")
    figure.colorbar(reliability_image, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].imshow(
        reliability,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    healthy_overlay = np.ma.masked_where(
        ~selection.healthy_context_mask,
        selection.healthy_context_mask.astype(np.float32),
    )
    axes[1].imshow(
        healthy_overlay,
        cmap=ListedColormap(["#00ff66"]),
        vmin=0.0,
        vmax=1.0,
        alpha=0.9,
        interpolation="nearest",
    )
    selected_overlay = np.ma.masked_where(
        ~selection.reconstruction_mask,
        selection.reconstruction_mask.astype(np.float32),
    )
    axes[1].imshow(
        selected_overlay,
        cmap=ListedColormap(["#ff00ff"]),
        vmin=0.0,
        vmax=1.0,
        alpha=0.2,
        interpolation="nearest",
    )
    selector_title = (
        f"Primary >={100.0 * selector.config.min_lidar_loss_fraction:.0f}% loss"
    )
    if selector.config.max_secondary_repair_boxes:
        selector_title += (
            f"; up to {selector.config.max_secondary_repair_boxes} secondary "
            f">={100.0 * selector.config.min_secondary_lidar_loss_fraction:.0f}% "
            f"loss/{100.0 * selector.config.min_secondary_repair_fault_fraction:.0f}% "
            "box purity"
        )
    axes[1].set_title(
        f"{selector_title} (magenta/cyan) + halo (green)\n"
        f"Excluded added-only cells: {selection.excluded_added_only_cell_count}"
    )
    _draw_repair_boxes(axes[1], selection)
    _draw_halo_outline(axes[1], selection)

    axes[2].imshow(clean, interpolation="nearest")
    axes[2].set_title("Clean LiDAR BEV")

    axes[3].imshow(faulty, interpolation="nearest")
    _draw_repair_boxes(axes[3], selection)
    _draw_halo_outline(axes[3], selection)
    axes[3].set_title(
        "Faulty LiDAR with Repair Boxes (cyan) and Halo (green dashed)\n"
        f"Repair area: {selection.selected_cell_count}; halo area: {selection.halo_cell_count}; "
        f"fault evidence: {selection.selected_fault_cell_count}; "
        f"healthy context: {selection.healthy_context_cell_count}"
    )
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return selection


def main():
    args = _parse_args()
    selector_config = build_selector_config(load_config(args.config))
    selector = FaultSelector(selector_config)
    output_path = Path(args.output)
    selection = render_fault_selection(args.sample, output_path, selector)

    print(f"Saved selection preview to: {output_path}")
    for rank, blob in enumerate(selection.selected_blobs, start=1):
        print(
            f"#{rank}: component={blob.component_id}, cells={blob.cell_count}, "
            f"mass={blob.fault_mass:.3f}, nearest={blob.nearest_distance_m:.2f}m, "
            f"centroid={blob.centroid_distance_m:.2f}m, "
            f"repair_faulty={100.0 * blob.repair_fault_fraction:.2f}%, "
            f"repair_target_met={blob.repair_target_met}, "
            f"halo_healthy={100.0 * blob.halo_healthy_fraction:.2f}%, "
            f"halo_healthy_cells={blob.healthy_occupied_cell_count}/"
            f"{blob.required_healthy_context_cell_count}, "
            f"halo_width={blob.halo_dilation_cells}px, "
            f"halo_target_met={blob.halo_target_met}"
        )
    print(
        f"Selected repair box area: {selection.selected_cell_count} cells; "
        f"severe-loss evidence inside: {selection.selected_fault_cell_count} cells "
        f"at >= {100.0 * selector_config.min_lidar_loss_fraction:.1f}% loss."
    )
    print(
        f"Excluded {selection.excluded_added_only_cell_count} added-only fault cells."
    )


if __name__ == "__main__":
    main()
