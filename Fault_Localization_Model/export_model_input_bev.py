from pathlib import Path
import argparse
import json

import numpy as np
from PIL import Image


def load_faulty_rgb(npz_path: Path) -> tuple[np.ndarray, dict]:
    with np.load(npz_path, allow_pickle=False) as data:
        if "faulty_rgb" not in data:
            raise KeyError(f"{npz_path} does not contain 'faulty_rgb'")
        rgb = data["faulty_rgb"]
        metadata = json.loads(str(data["metadata_json"])) if "metadata_json" in data else {}
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected faulty_rgb shape [H, W, 3], got {rgb.shape}")
    return rgb, metadata


def main():
    parser = argparse.ArgumentParser(description="Export the exact faulty_rgb BEV image seen by the model.")
    parser.add_argument("--npz", required=True, help="Path to one generated reliability-map .npz sample.")
    parser.add_argument("--output", default=None, help="Output PNG path. Defaults beside the .npz.")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    rgb, metadata = load_faulty_rgb(npz_path)
    output = Path(args.output) if args.output else npz_path.with_name(npz_path.stem + "_MODEL_INPUT_faulty_rgb.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(output)

    normalized = rgb.astype(np.float32) / 255.0
    print(f"Saved exact model-input BEV: {output}")
    print(f"faulty_rgb uint8 shape: {rgb.shape}, dtype: {rgb.dtype}, min: {rgb.min()}, max: {rgb.max()}")
    print(f"model tensor values after /255: min={normalized.min():.6f}, max={normalized.max():.6f}")
    if metadata:
        print(f"sample: fault={metadata.get('fault')} severity={metadata.get('severity')} timestamp={metadata.get('timestamp')}")


if __name__ == "__main__":
    main()
