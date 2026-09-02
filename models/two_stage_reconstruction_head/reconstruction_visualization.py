"""Shared occupancy-only visualizations for reconstruction evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def occupancy_image(bev: torch.Tensor) -> np.ndarray:
    """Return the canonical occupancy channel as a display-ready array."""

    if bev.ndim != 3 or bev.shape[0] < 1:
        raise ValueError(f"Expected BEV shaped [C,H,W], got {tuple(bev.shape)}")
    return bev.detach().float().cpu()[0].clamp(0.0, 1.0).numpy()


def radar_lidar_occupancy_overlay(
    lidar_bev: torch.Tensor,
    radar_bev: torch.Tensor,
    *,
    occupancy_threshold: float = 0.5,
) -> np.ndarray:
    """Overlay LiDAR occupancy and radar support using fixed semantic colors.

    LiDAR-only cells are cyan, radar-only cells are magenta, cells supported by
    both modalities are white, and empty cells are black.
    """

    if radar_bev.ndim != 3:
        raise ValueError(
            f"Expected radar BEV shaped [C,H,W], got {tuple(radar_bev.shape)}"
        )
    lidar_support = occupancy_image(lidar_bev) >= occupancy_threshold
    radar_support = (
        radar_bev.detach().float().abs().amax(dim=0).cpu().numpy() > 0.0
    )
    if lidar_support.shape != radar_support.shape:
        raise ValueError(
            "LiDAR and radar BEV spatial shapes differ: "
            f"{lidar_support.shape} versus {radar_support.shape}"
        )
    return np.stack(
        (radar_support, lidar_support, lidar_support | radar_support),
        axis=-1,
    ).astype(np.float32)


def save_three_panel_reconstruction(
    destination: Path,
    *,
    clean_bev: torch.Tensor,
    faulty_bev: torch.Tensor,
    reconstructed_bev: torch.Tensor,
    radar_bev: torch.Tensor,
    reconstruction_mask: torch.Tensor,
    reconstruction_title: str,
    figure_title: str,
    occupancy_threshold: float = 0.5,
) -> None:
    """Save clean, faulty+radar, and reconstructed+radar occupancy panels."""

    clean = occupancy_image(clean_bev)
    faulty_overlay = radar_lidar_occupancy_overlay(
        faulty_bev,
        radar_bev,
        occupancy_threshold=occupancy_threshold,
    )
    reconstructed_overlay = radar_lidar_occupancy_overlay(
        reconstructed_bev,
        radar_bev,
        occupancy_threshold=occupancy_threshold,
    )
    mask = reconstruction_mask.detach().bool().squeeze().cpu().numpy()

    figure, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="black")
    panels = (
        (clean, "Clean LiDAR occupancy", "gray"),
        (
            faulty_overlay,
            "Faulty LiDAR + radar",
            None,
        ),
        (
            reconstructed_overlay,
            f"{reconstruction_title} + radar",
            None,
        ),
    )
    for axis, (image, title, cmap) in zip(axes, panels):
        axis.imshow(
            image,
            cmap=cmap,
            vmin=0.0 if cmap else None,
            vmax=1.0 if cmap else None,
            interpolation="nearest",
        )
        if mask.any() and not mask.all():
            axis.contour(
                mask.astype(np.uint8),
                levels=(0.5,),
                colors="yellow",
                linewidths=0.8,
            )
        axis.set_title(title, color="white")
        axis.axis("off")
    figure.suptitle(
        figure_title
        + f" | occupancy threshold {occupancy_threshold:.2f}"
        + "\nCyan: LiDAR | Magenta: radar | White: overlap | Yellow: repair mask",
        color="white",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination,
        dpi=150,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)
