"""Visualize learned Radar-neighbor weights at representative ranges."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.reconstruction_head.coarse_reconstruction.coarse_config import (
    CoarseReconstructionConfig,
)
from models.reconstruction_head.coarse_reconstruction.range_aware_radar import (
    RangeAwareRadarAggregation,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--radar-npz", type=Path, required=True)
    parser.add_argument("--active-mask-npz", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--target-ranges", type=float, nargs="+", default=(10, 30, 50, 65)
    )
    return parser.parse_args()


def _load_radar(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        radar = np.asarray(payload["radar_bev"], dtype=np.float32)
    if radar.ndim != 3:
        raise ValueError(f"radar_bev must be 3D, got {radar.shape}")
    if radar.shape[0] != 4 and radar.shape[-1] == 4:
        radar = np.moveaxis(radar, -1, 0)
    if radar.shape[0] != 4:
        raise ValueError(f"radar_bev must have four channels, got {radar.shape}")
    return radar


def _load_active_mask(path: Path | None, shape: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.ones(shape, dtype=np.float32)
    with np.load(path, allow_pickle=False) as payload:
        if "active_mask" in payload:
            active = np.asarray(payload["active_mask"], dtype=np.float32)
        else:
            reconstruction = np.asarray(
                payload["reconstruction_mask"], dtype=np.float32
            )
            halo = np.asarray(payload["halo_mask"], dtype=np.float32)
            active = np.maximum(reconstruction, halo)
    active = np.squeeze(active)
    if active.shape != shape:
        raise ValueError(f"active mask shape {active.shape} != {shape}")
    return (active > 0).astype(np.float32)


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    if "model_config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain a coarse model")
    model_config = CoarseReconstructionConfig.from_dict(
        dict(checkpoint["model_config"])
    )
    if not model_config.range_aware_radar.enabled:
        raise ValueError("Checkpoint does not use range-aware Radar aggregation")
    module = RangeAwareRadarAggregation(
        model_config.radar_channels, model_config.range_aware_radar
    ).to(device)
    prefix = "range_aware_radar."
    state = {
        key[len(prefix) :]: value
        for key, value in checkpoint["model_state_dict"].items()
        if key.startswith(prefix)
    }
    module.load_state_dict(state, strict=True)
    module.eval()

    radar_array = _load_radar(args.radar_npz)
    active_array = _load_active_mask(
        args.active_mask_npz, radar_array.shape[-2:]
    )
    radar = torch.from_numpy(radar_array)[None].to(device)
    active = torch.from_numpy(active_array)[None, None].to(device)
    with torch.inference_mode():
        _, debug = module(radar, active, return_attention=True)
    weights = debug["attention_weights"][0].cpu().numpy()
    radius = debug["radius_m"][0, 0].cpu().numpy()
    _, _, ranges, _, _ = module._geometry(
        radar.shape[-2], radar.shape[-1], torch.device("cpu"), torch.float32
    )
    ranges = ranges.numpy()

    valid_centers = active_array > 0
    figure, axes = plt.subplots(
        4,
        len(args.target_ranges),
        figsize=(4 * len(args.target_ranges), 15),
        facecolor="black",
    )
    axes = np.asarray(axes).reshape(4, -1)
    occupancy = radar_array[0]
    power = radar_array[1]
    kernel_size = weights.shape[-1]
    padding = kernel_size // 2
    x_step = (
        model_config.range_aware_radar.x_max_m
        - model_config.range_aware_radar.x_min_m
    ) / radar.shape[-2]
    for column, target in enumerate(args.target_ranges):
        difference = np.where(valid_centers, np.abs(ranges - target), np.inf)
        row, col = np.unravel_index(np.argmin(difference), difference.shape)
        axes[0, column].imshow(occupancy, cmap="gray", vmin=0, vmax=1)
        axes[0, column].scatter(col, row, c="cyan", s=35, marker="x")
        axes[0, column].add_patch(
            Circle(
                (col, row),
                radius[row, col] / x_step,
                fill=False,
                color="cyan",
                linewidth=1.5,
            )
        )
        axes[0, column].set_title(
            f"target {target:g} m | actual {ranges[row, col]:.1f} m",
            color="white",
        )
        axes[0, column].axis("off")
        padded_occupancy = np.pad(occupancy, padding)
        padded_power = np.pad(power, padding)
        occupancy_patch = padded_occupancy[
            row : row + kernel_size, col : col + kernel_size
        ]
        power_patch = padded_power[
            row : row + kernel_size, col : col + kernel_size
        ]
        offset_cells = np.arange(-padding, padding + 1)
        offset_rows, offset_cols = np.meshgrid(
            offset_cells, offset_cells, indexing="ij"
        )
        candidate_distance = np.sqrt(
            (offset_rows * x_step) ** 2 + (offset_cols * x_step) ** 2
        )
        considered = (
            (candidate_distance <= radius[row, col] + 1.0e-6)
            & (
                occupancy_patch
                >= model_config.range_aware_radar.occupancy_threshold
            )
        )
        candidate_display = np.where(considered, power_patch, np.nan)
        candidate_image = axes[1, column].imshow(
            candidate_display, cmap="viridis", vmin=0, vmax=1
        )
        axes[1, column].set_title(
            f"considered occupied cells\npower | radius {radius[row, col]:.2f} m",
            color="white",
        )
        axes[1, column].set_xlabel("neighbor column offset", color="white")
        axes[1, column].set_ylabel("neighbor row offset", color="white")
        axes[1, column].tick_params(colors="white")
        figure.colorbar(candidate_image, ax=axes[1, column], fraction=0.046)

        weight_image = axes[2, column].imshow(
            weights[row, col], cmap="magma", vmin=0
        )
        axes[2, column].set_title("learned attention weights", color="white")
        axes[2, column].tick_params(colors="white")
        figure.colorbar(weight_image, ax=axes[2, column], fraction=0.046)

        support_image = axes[3, column].imshow(
            weights[row, col] * power_patch,
            cmap="plasma",
            vmin=0,
        )
        axes[3, column].set_title(
            "power-weighted support (debug only)", color="white"
        )
        axes[3, column].tick_params(colors="white")
        figure.colorbar(support_image, ax=axes[3, column], fraction=0.046)
    figure.suptitle(
        "Learned range-aware Radar attention (11x11 candidates)",
        color="white",
        fontsize=16,
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="black")
    plt.close(figure)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
