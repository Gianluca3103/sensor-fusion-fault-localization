from pathlib import Path
import argparse
import csv
import json
import os
import random
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model_v5_like import FaultHeatmapV5Like


DEFAULT_DATASET_ROOT = SCRIPT_DIR / "grid_reliability_change_marks"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "runs" / "reliability_map_v5_like"


class ReliabilityMapDataset(Dataset):
    def __init__(self, paths, resize_hw):
        self.paths = list(paths)
        self.resize_hw = resize_hw
        if not self.paths:
            raise FileNotFoundError("No .npz reliability-map samples found.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        with np.load(path, allow_pickle=False) as data:
            rgb = data["faulty_rgb"].astype(np.float32) / 255.0
            target = data["fault_heatmap"].astype(np.float32)
            metadata = json.loads(str(data["metadata_json"]))

        x = torch.from_numpy(np.transpose(rgb, (2, 0, 1)))
        y = torch.from_numpy(target).unsqueeze(0)
        if self.resize_hw:
            x = F.interpolate(x.unsqueeze(0), size=self.resize_hw, mode="bilinear", align_corners=False).squeeze(0)
            y = F.interpolate(y.unsqueeze(0), size=self.resize_hw, mode="nearest").squeeze(0)
        return {"x": x, "y": y, "rgb": (rgb * 255).astype(np.uint8), "path": str(path), "metadata": metadata}


def collate(batch):
    return {
        "x": torch.stack([item["x"] for item in batch]),
        "y": torch.stack([item["y"] for item in batch]),
        "rgb": [item["rgb"] for item in batch],
        "path": [item["path"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }


def split_paths(paths, val_ratio, seed):
    rng = random.Random(seed)
    paths = list(paths)
    rng.shuffle(paths)
    val_count = max(1, int(round(len(paths) * val_ratio))) if len(paths) > 1 else 0
    return paths[val_count:], paths[:val_count]


def dice_loss(logits, target, eps=1e-6):
    pred = torch.sigmoid(logits)
    intersection = torch.sum(pred * target, dim=(1, 2, 3))
    union = torch.sum(pred, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3))
    return 1.0 - torch.mean((2.0 * intersection + eps) / (union + eps))


def reliability_loss(logits, target, grid_size=100):
    pred = torch.sigmoid(logits)
    weight = 1.0 + 5.0 * target
    bce = F.binary_cross_entropy_with_logits(logits, target, weight=weight)
    l1 = F.l1_loss(pred, target)
    mse = F.mse_loss(pred, target)
    pred_grid = F.adaptive_avg_pool2d(pred, output_size=(grid_size, grid_size))
    target_grid = F.adaptive_avg_pool2d(target, output_size=(grid_size, grid_size))
    grid_l1 = F.l1_loss(pred_grid, target_grid)
    grid_mse = F.mse_loss(pred_grid, target_grid)
    return bce + 0.75 * l1 + 0.25 * mse + 0.20 * dice_loss(logits, target) + 1.25 * grid_l1 + 0.50 * grid_mse


def run_epoch(model, loader, optimizer, device, train, grid_size):
    model.train(train)
    total = 0.0
    count = 0
    for batch in tqdm(loader, leave=False):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
            if logits.shape[-2:] != y.shape[-2:]:
                logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
            loss = reliability_loss(logits, y, grid_size=grid_size)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total += float(loss.item()) * x.shape[0]
        count += x.shape[0]
    return total / max(count, 1)


def blue_red_reliability(unreliability):
    reliability = 1.0 - np.clip(unreliability, 0.0, 1.0)
    rgb = np.zeros((*reliability.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip((1.0 - reliability) * 255, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(reliability * 255, 0, 255).astype(np.uint8)
    return rgb


def make_grid_like(values, grid_size=100):
    tensor = torch.from_numpy(values.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(tensor, output_size=(grid_size, grid_size))
    blocky = F.interpolate(pooled, size=values.shape, mode="nearest")
    return blocky.squeeze(0).squeeze(0).numpy()


def draw_cell_boundaries(rgb, grid_size=100):
    output = rgb.copy()
    height, width = output.shape[:2]
    row_step = max(1, height // grid_size)
    col_step = max(1, width // grid_size)
    line_color = np.array([18, 18, 18], dtype=np.uint8)
    for row in range(row_step, height, row_step):
        output[row : row + 1, :] = line_color
    for col in range(col_step, width, col_step):
        output[:, col : col + 1] = line_color
    return output


def add_label_above(rgb, label):
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    pad = 28
    canvas = np.zeros((rgb.shape[0] + pad, rgb.shape[1], 3), dtype=np.uint8)
    canvas[:pad] = np.array([18, 18, 18], dtype=np.uint8)
    canvas[pad:] = rgb
    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.text((8, 7), label, fill=(255, 255, 255), font=font)
    return np.array(image)


def add_reliability_colorbar(rgb):
    bar_width = 34
    label_width = 104
    pad = 8
    height = rgb.shape[0]
    canvas = np.zeros((height, rgb.shape[1] + bar_width + label_width + pad, 3), dtype=np.uint8)
    canvas[:, : rgb.shape[1]] = rgb
    x0 = rgb.shape[1] + pad

    values = np.linspace(1.0, 0.0, height, dtype=np.float32)
    bar = np.zeros((height, bar_width, 3), dtype=np.uint8)
    bar[..., 0] = np.clip((1.0 - values[:, None]) * 255, 0, 255).astype(np.uint8)
    bar[..., 2] = np.clip(values[:, None] * 255, 0, 255).astype(np.uint8)
    canvas[:, x0 : x0 + bar_width] = bar

    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    text_x = x0 + bar_width + 8
    draw.text((text_x, 8), "Reliable", fill=(80, 160, 255), font=font)
    draw.text((text_x, 26), "1.0", fill=(80, 160, 255), font=font)
    draw.text((text_x, height // 2 - 9), "0.5", fill=(210, 120, 255), font=font)
    draw.text((text_x, height - 42), "0.0", fill=(255, 90, 90), font=font)
    draw.text((text_x, height - 24), "Unreliable", fill=(255, 90, 90), font=font)
    return np.array(image)


def side_by_side(images):
    max_height = max(image.shape[0] for image in images)
    padded = []
    for image in images:
        if image.shape[0] == max_height:
            padded.append(image)
            continue
        canvas = np.zeros((max_height, image.shape[1], 3), dtype=np.uint8)
        canvas[: image.shape[0]] = image
        padded.append(canvas)
    return np.concatenate(padded, axis=1)


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path)


def save_predictions(model, loader, output_root, device, max_images):
    pred_dir = output_root / "val_predictions"
    rows = []
    saved = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].cpu().numpy()
            pred = torch.sigmoid(model(x)).cpu().numpy()
            for i in range(x.shape[0]):
                meta = batch["metadata"][i]
                stem = f"{saved:04d}_{meta['fault']}_s{meta['severity']}_{meta['timestamp']}"
                target = y[i, 0]
                target = make_grid_like(target, grid_size=100)
                pred_map = make_grid_like(pred[i, 0], grid_size=100)
                target_rgb = draw_cell_boundaries(blue_red_reliability(target), grid_size=100)
                pred_rgb = draw_cell_boundaries(blue_red_reliability(pred_map), grid_size=100)
                input_rgb = batch["rgb"][i]
                if input_rgb.shape[:2] != target_rgb.shape[:2]:
                    input_rgb = np.array(
                        Image.fromarray(input_rgb, mode="RGB").resize(
                            (target_rgb.shape[1], target_rgb.shape[0]),
                            Image.Resampling.BILINEAR,
                        )
                    )
                panel = side_by_side(
                    [
                        add_label_above(input_rgb, f"faulty BEV input: {meta['fault']} S{meta['severity']}"),
                        add_reliability_colorbar(add_label_above(target_rgb, f"ideal reliability: {meta['fault']} S{meta['severity']}")),
                        add_reliability_colorbar(add_label_above(pred_rgb, f"learned reliability: {meta['fault']} S{meta['severity']}")),
                    ]
                )
                save_image(pred_dir / f"{stem}_comparison.png", panel)
                save_image(pred_dir / f"{stem}_target_reliability.png", target_rgb)
                save_image(pred_dir / f"{stem}_pred_reliability.png", pred_rgb)
                rows.append(
                    {
                        "fault": meta["fault"],
                        "severity": meta["severity"],
                        "timestamp": meta["timestamp"],
                        "mae": float(np.mean(np.abs(pred_map - target))),
                        "mean_pred_unreliability": float(np.mean(pred_map)),
                        "mean_target_unreliability": float(np.mean(target)),
                    }
                )
                saved += 1
                if saved >= max_images:
                    return rows
    return rows


def save_curve(history, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], label="train")
    plt.plot(history["epoch"], history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Reliability map training")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train a v5-like model to predict ideal Hercules reliability maps.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--resize-height", type=int, default=320)
    parser.add_argument("--resize-width", type=int, default=320)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-val-images", type=int, default=24)
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    paths = sorted(dataset_root.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {dataset_root}")

    train_paths, val_paths = split_paths(paths, args.val_ratio, args.seed)
    resize_hw = (args.resize_height, args.resize_width)
    device = torch.device(args.device)

    train_loader = DataLoader(
        ReliabilityMapDataset(train_paths, resize_hw),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        ReliabilityMapDataset(val_paths, resize_hw),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    model = FaultHeatmapV5Like(in_channels=3, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history = {"epoch": [], "train_loss": [], "val_loss": []}
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True, grid_size=args.grid_size)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False, grid_size=args.grid_size)
        scheduler.step()
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch {epoch:03d}: train={train_loss:.6f} val={val_loss:.6f}", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_val_loss": best_val,
                    "target": "fault_heatmap/unreliability; reliability=1-target",
                    "input": "faulty_rgb_bev",
                },
                checkpoint_dir / "best_model.pt",
            )

    save_curve(history, output_root / "plots" / "training_curve.png")
    with (output_root / "training_history.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        for i, epoch in enumerate(history["epoch"]):
            writer.writerow({"epoch": epoch, "train_loss": history["train_loss"][i], "val_loss": history["val_loss"][i]})

    checkpoint = torch.load(checkpoint_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    rows = save_predictions(model, val_loader, output_root, device, args.max_val_images)
    if rows:
        with (output_root / "val_predictions" / "prediction_metrics.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved run: {output_root}")


if __name__ == "__main__":
    main()
