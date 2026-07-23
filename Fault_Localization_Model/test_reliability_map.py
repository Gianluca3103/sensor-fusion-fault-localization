from pathlib import Path
import argparse
import csv
import json
import os
import random
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

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


DEFAULT_DATASET_ROOT = SCRIPT_DIR / "mixed_target_old_laser_dataset"
DEFAULT_CHECKPOINT = SCRIPT_DIR / "runs" / "mixed_target_old_laser_v5_like" / "checkpoints" / "best_model.pt"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "runs" / "mixed_target_old_laser_v5_like" / "test_gt_vs_pred"


class ReliabilityDataset(Dataset):
    def __init__(self, paths, resize_hw):
        self.paths = list(paths)
        self.resize_hw = resize_hw
        if not self.paths:
            raise FileNotFoundError("No .npz samples found for testing.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with np.load(path, allow_pickle=False) as data:
            rgb = data["faulty_rgb"].astype(np.float32) / 255.0
            target = data["fault_heatmap"].astype(np.float32)
            metadata = json.loads(str(data["metadata_json"]))

        x = torch.from_numpy(np.transpose(rgb, (2, 0, 1)))
        y = torch.from_numpy(target).unsqueeze(0)
        if self.resize_hw:
            x = F.interpolate(x.unsqueeze(0), size=self.resize_hw, mode="bilinear", align_corners=False).squeeze(0)
            y = F.interpolate(y.unsqueeze(0), size=self.resize_hw, mode="nearest").squeeze(0)
        return {"x": x, "y": y, "metadata": metadata, "path": str(path)}


def split_paths(paths, val_ratio, seed):
    rng = random.Random(seed)
    paths = list(paths)
    rng.shuffle(paths)
    val_count = max(1, int(round(len(paths) * val_ratio))) if len(paths) > 1 else len(paths)
    return paths[val_count:], paths[:val_count]


def collate(batch):
    return {
        "x": torch.stack([item["x"] for item in batch]),
        "y": torch.stack([item["y"] for item in batch]),
        "metadata": [item["metadata"] for item in batch],
        "path": [item["path"] for item in batch],
    }


def colorize_unreliability(values):
    reliability = 1.0 - np.clip(values, 0.0, 1.0)
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
    # Do not erase full-resolution maps by drawing a boundary on every pixel.
    if row_step <= 1 or col_step <= 1:
        return output
    line_color = np.array([18, 18, 18], dtype=np.uint8)
    for row in range(row_step, height, row_step):
        output[row : row + 1, :] = line_color
    for col in range(col_step, width, col_step):
        output[:, col : col + 1] = line_color
    return output


def add_header(rgb, title, subtitle="blue=reliable, red=unreliable"):
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    lines = [title, subtitle]
    pad = 46
    canvas = np.zeros((rgb.shape[0] + pad, rgb.shape[1], 3), dtype=np.uint8)
    canvas[:pad] = np.array([18, 18, 18], dtype=np.uint8)
    canvas[pad:] = rgb
    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    y = 6
    for line in lines:
        draw.text((8, y), line, fill=(255, 255, 255), font=font)
        y += 18
    return np.array(image)


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path)


def main():
    parser = argparse.ArgumentParser(description="Test reliability map model and save GT-vs-pred heatmaps.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resize-height", type=int, default=320)
    parser.add_argument("--resize-width", type=int, default=320)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    checkpoint_path = Path(args.checkpoint)
    output_root = Path(args.output_root)
    paths = sorted(dataset_root.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {dataset_root}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {})
    base_channels = args.base_channels or int(checkpoint_args.get("base_channels", 16))
    resize_hw = (args.resize_height, args.resize_width)

    _, test_paths = split_paths(paths, args.val_ratio, args.seed)
    loader = DataLoader(
        ReliabilityDataset(test_paths, resize_hw),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    device = torch.device(args.device)
    model = FaultHeatmapV5Like(in_channels=3, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows = []
    saved = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="testing"):
            x = batch["x"].to(device)
            target = batch["y"].cpu().numpy()
            pred = torch.sigmoid(model(x)).cpu().numpy()
            for i in range(x.shape[0]):
                meta = batch["metadata"][i]
                gt = make_grid_like(target[i, 0], grid_size=args.grid_size)
                model_map = make_grid_like(pred[i, 0], grid_size=args.grid_size)
                fault_label = f"{meta['fault']} severity {meta['severity']}"
                gt_rgb = draw_cell_boundaries(colorize_unreliability(gt), grid_size=args.grid_size)
                pred_rgb = draw_cell_boundaries(colorize_unreliability(model_map), grid_size=args.grid_size)
                gt_rgb = add_header(gt_rgb, "GROUND TRUTH HEAT MAP 100x100 GRID", fault_label)
                pred_rgb = add_header(pred_rgb, "MODEL TESTED HEAT MAP 100x100 GRID", fault_label)
                side_by_side = np.concatenate([gt_rgb, pred_rgb], axis=1)
                stem = f"{saved:04d}_{meta['fault']}_s{meta['severity']}_{meta['timestamp']}"
                save_image(output_root / f"{stem}_gt_vs_model.png", side_by_side)
                rows.append(
                    {
                        "index": saved,
                        "fault": meta["fault"],
                        "severity": meta["severity"],
                        "timestamp": meta["timestamp"],
                        "mae": float(np.mean(np.abs(model_map - gt))),
                        "mse": float(np.mean((model_map - gt) ** 2)),
                        "mean_ground_truth_unreliability": float(np.mean(gt)),
                        "mean_model_unreliability": float(np.mean(model_map)),
                        "image": str(output_root / f"{stem}_gt_vs_model.png"),
                        "source_npz": batch["path"][i],
                    }
                )
                saved += 1
                if saved >= args.max_images:
                    break
            if saved >= args.max_images:
                break

    if rows:
        with (output_root / "test_metrics.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved {saved} GT-vs-model comparisons: {output_root}")


if __name__ == "__main__":
    main()
