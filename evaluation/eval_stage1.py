from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Fault_Localization_Model.io_utils import atomic_write_json
from PFS_Radar.train_pfs_radar import sample_paths_from_root
from PFS.training_utils import resolve_device, seed_everything
from models.reconstruction_head.coarse_reconstructor import CoarseLiDARRadarReconstructor
from models.reconstruction_head.datasets import Stage1RadarReconstructionDataset, collate_stage1_batch
from models.reconstruction_head.diffusion_scheduler import DiffusionSchedule
from models.reconstruction_head.losses import healthy_region_change, masked_feature_loss
from models.reconstruction_head.residual_diffusion_unet import ResidualDiffusionUNet
from models.reconstruction_head.stage1_pipeline import Stage1ReconstructionPipeline
from training.train_stage1_diffusion import load_frozen_coarse


def _bev_image(tensor: torch.Tensor) -> torch.Tensor:
    """Collapse a [C,H,W] BEV tensor into one visualization plane."""
    tensor = tensor.detach().float().cpu()
    if tensor.shape[0] == 1:
        image = tensor[0]
    else:
        image = tensor.abs().mean(dim=0)
    finite = torch.isfinite(image)
    if not bool(finite.all()):
        image = torch.where(finite, image, torch.zeros_like(image))
    min_value = image.min()
    max_value = image.max()
    if float(max_value - min_value) > 1e-8:
        image = (image - min_value) / (max_value - min_value)
    return image


def save_visualization(path: Path, panels: list[tuple[str, torch.Tensor]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = min(5, len(panels))
    rows = (len(panels) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = np.asarray(axes).reshape(-1).tolist()
    for axis, (title, tensor) in zip(axes, panels):
        axis.imshow(_bev_image(tensor), cmap="magma", origin="lower")
        axis.set_title(title)
        axis.axis("off")
    for axis in axes[len(panels):]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def load_diffusion(path: Path, device, residual_channels, coarse_channels, conditioning_channels):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    args = checkpoint.get("args", {})
    model = ResidualDiffusionUNet(
        residual_channels=residual_channels,
        coarse_channels=coarse_channels,
        conditioning_channels=conditioning_channels,
        base_channels=int(args.get("base_channels", 16)),
        levels=int(args.get("levels", 4)),
        normalization=args.get("normalization", "batch"),
        dropout=float(args.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    schedule = DiffusionSchedule(num_train_timesteps=int(args.get("num_train_timesteps", 1000))).to(device)
    return model, schedule


def main():
    parser = argparse.ArgumentParser(description="Evaluate complete Stage I reconstruction pipeline.")
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--radar-root", required=True)
    parser.add_argument("--coarse-checkpoint", required=True)
    parser.add_argument("--diffusion-checkpoint", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resize-height", type=int, default=320)
    parser.add_argument("--resize-width", type=int, default=320)
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--max-visualizations", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = resolve_device(args.device)
    paths = sample_paths_from_root(Path(args.test_root))
    if not paths:
        raise FileNotFoundError("--test-root must contain .npz files")
    dataset = Stage1RadarReconstructionDataset(
        paths,
        Path(args.radar_root),
        resize_hw=(args.resize_height, args.resize_width),
        mask_threshold=args.mask_threshold,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_stage1_batch)
    coarse = load_frozen_coarse(Path(args.coarse_checkpoint), device)
    first = dataset[0]
    with torch.no_grad():
        probe = coarse(
            first["lidar_corrupt"][None].to(device),
            first["radar"][None].to(device),
            first["mask"][None].to(device),
            first["occupancy"][None].to(device),
        )
    diffusion = None
    schedule = None
    if args.diffusion_checkpoint:
        diffusion, schedule = load_diffusion(
            Path(args.diffusion_checkpoint),
            device,
            first["lidar_corrupt"].shape[0],
            first["lidar_corrupt"].shape[0],
            probe["conditioning_features"].shape[1],
        )
    pipeline = Stage1ReconstructionPipeline(coarse, diffusion, schedule)
    totals = {
        "corrupt_mae": 0.0,
        "coarse_mae": 0.0,
        "final_mae": 0.0,
        "coarse_healthy_change": 0.0,
        "final_healthy_change": 0.0,
        "seconds_per_sample": 0.0,
    }
    samples = 0
    visualized = 0
    for batch in tqdm(loader, desc="eval"):
        lidar_corrupt = batch["lidar_corrupt"].to(device)
        lidar_clean = batch["lidar_clean"].to(device)
        radar = batch["radar"].to(device)
        mask = batch["mask"].to(device)
        occupancy = batch["occupancy"].to(device)
        start = time.perf_counter()
        with torch.no_grad():
            coarse_out = coarse(lidar_corrupt, radar, mask, occupancy)
            if diffusion is None:
                final = coarse_out["coarse_features"]
                residual = torch.zeros_like(final)
            else:
                residual = pipeline.sample_residual(
                    coarse_out["coarse_features"],
                    coarse_out["conditioning_features"],
                    mask,
                    num_inference_steps=args.num_inference_steps,
                    seed=args.seed,
                )
                final = lidar_corrupt * (1.0 - mask) + (coarse_out["coarse_features"] + mask * residual) * mask
        if args.max_visualizations > 0 and visualized < args.max_visualizations:
            for item_index in range(min(lidar_corrupt.shape[0], args.max_visualizations - visualized)):
                metadata = batch["metadata"][item_index]
                fault = metadata.get("fault", "unknown")
                severity = metadata.get("severity", "x")
                stem = f"{visualized:04d}_{fault}_s{severity}"
                coarse_error = mask[item_index] * (coarse_out["coarse_features"][item_index] - lidar_clean[item_index]).abs()
                target_residual = mask[item_index] * (lidar_clean[item_index] - coarse_out["coarse_features"][item_index])
                final_error = mask[item_index] * (final[item_index] - lidar_clean[item_index]).abs()
                save_visualization(
                    Path(args.output_root) / "visualizations" / f"{stem}_stage1_eval.png",
                    [
                        ("corrupted LiDAR", lidar_corrupt[item_index]),
                        ("clean target", lidar_clean[item_index]),
                        ("GT fault mask", mask[item_index]),
                        ("radar evidence", radar[item_index]),
                        ("coarse recon", coarse_out["coarse_features"][item_index]),
                        ("coarse error", coarse_error),
                        ("target residual", target_residual),
                        ("pred residual", residual[item_index]),
                        ("final recon", final[item_index]),
                        ("final error", final_error),
                    ],
                )
                visualized += 1
        elapsed = time.perf_counter() - start
        batch_size = lidar_corrupt.shape[0]
        totals["corrupt_mae"] += float(masked_feature_loss(lidar_corrupt, lidar_clean, mask, mode="l1")) * batch_size
        totals["coarse_mae"] += float(masked_feature_loss(coarse_out["coarse_features"], lidar_clean, mask, mode="l1")) * batch_size
        totals["final_mae"] += float(masked_feature_loss(final, lidar_clean, mask, mode="l1")) * batch_size
        totals["coarse_healthy_change"] += float(healthy_region_change(coarse_out["coarse_features"], lidar_corrupt, mask)) * batch_size
        totals["final_healthy_change"] += float(healthy_region_change(final, lidar_corrupt, mask)) * batch_size
        totals["seconds_per_sample"] += elapsed
        samples += batch_size
    metrics = {key: value / max(samples, 1) for key, value in totals.items()}
    metrics["coarse_improvement_over_corrupt"] = metrics["corrupt_mae"] - metrics["coarse_mae"]
    metrics["final_improvement_over_coarse"] = metrics["coarse_mae"] - metrics["final_mae"]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "stage1_eval_metrics.json", metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
