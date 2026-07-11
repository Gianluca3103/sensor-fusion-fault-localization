from pathlib import Path
import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FAULT_MODEL_DIR = REPO_ROOT / "Fault_Localization_Model"
if str(FAULT_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(FAULT_MODEL_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pfs_model import PFSReliabilityModel
from train_pfs_reliability_map import PFSReliabilityDataset, collate
from train_reliability_map import save_predictions, split_paths


DEFAULT_DATASET_ROOT = FAULT_MODEL_DIR / "grid_reliability_7500_fog_s3_x64_y32"
DEFAULT_CHECKPOINT = SCRIPT_DIR / "runs" / "pfs_7500_fog_s3_x64_y32" / "checkpoints" / "last_checkpoint.pt"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "runs" / "pfs_7500_fog_s3_x64_y32" / "test_10_predictions"


def main():
    parser = argparse.ArgumentParser(description="Save PFS model reliability-map prediction samples.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resize-height", type=int, default=320)
    parser.add_argument("--resize-width", type=int, default=320)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=10)
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

    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    base_channels = args.base_channels or int(checkpoint_args.get("base_channels", 16))
    dropout = args.dropout if args.dropout is not None else float(checkpoint_args.get("dropout", 0.0))

    _, test_paths = split_paths(paths, args.val_ratio, args.seed)
    resize_hw = (args.resize_height, args.resize_width)
    loader = DataLoader(
        PFSReliabilityDataset(test_paths, resize_hw),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    model = PFSReliabilityModel(in_channels=3, base_channels=base_channels, dropout=dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    rows = save_predictions(model, loader, output_root, device, args.max_images)
    print(f"Saved {len(rows)} PFS prediction comparisons: {output_root}")


if __name__ == "__main__":
    main()
