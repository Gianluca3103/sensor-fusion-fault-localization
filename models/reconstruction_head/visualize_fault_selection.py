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


def _draw_selected_boxes(axis, selection):
    for rank, blob in enumerate(selection.selected_blobs, start=1):
        halo_top, halo_left, halo_bottom, halo_right = blob.halo_bbox
        axis.add_patch(
            Rectangle(
                (halo_left - 0.5, halo_top - 0.5),
                halo_right - halo_left,
                halo_bottom - halo_top,
                fill=False,
                edgecolor="#00ff66",
                linewidth=1.6,
                linestyle="--",
            )
        )
        top, left, bottom, right = blob.bbox
        axis.add_patch(
            Rectangle(
                (left - 0.5, top - 0.5),
                right - left,
                bottom - top,
                fill=False,
                edgecolor="cyan",
                linewidth=1.4,
            )
        )
        axis.text(
            left,
            max(0, top - 2),
            f"#{rank}: repair {100.0 * blob.repair_fault_fraction:.1f}% faulty; "
            f"halo {100.0 * blob.halo_healthy_fraction:.1f}% healthy, "
            f"{blob.healthy_occupied_cell_count}/"
            f"{blob.required_healthy_context_cell_count} cells, "
            f"+{blob.halo_dilation_cells}px",
            color="cyan",
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 1},
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
    _draw_selected_boxes(axes[1], selection)
    axes[1].set_title(
        "Fault Selector on Reliability Map\n"
        f"Excluded added-only cells: {selection.excluded_added_only_cell_count}"
    )

    axes[2].imshow(clean, interpolation="nearest")
    axes[2].set_title("Clean LiDAR BEV")

    axes[3].imshow(faulty, interpolation="nearest")
    _draw_selected_boxes(axes[3], selection)
    axes[3].set_title(
        "Faulty LiDAR with Reconstruction Boxes\n"
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
        f"Rejected {len(selection.rejected_small_blobs)} blobs smaller than "
        f"{selector_config.min_blob_cells} cells."
    )
    print(
        f"Excluded {selection.excluded_added_only_cell_count} added-only fault cells."
    )


if __name__ == "__main__":
    main()
