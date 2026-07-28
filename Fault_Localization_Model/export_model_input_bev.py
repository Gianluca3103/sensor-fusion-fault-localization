from pathlib import Path
import argparse
import json
import sys

import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from Fault_Localization_Model.sample_utils import InvalidSampleError, validate_rgb_array


def load_faulty_rgb(npz_path: Path) -> tuple[np.ndarray, dict]:
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            if "faulty_rgb" not in data:
                raise KeyError("faulty_rgb is missing")
            rgb = validate_rgb_array(
                data["faulty_rgb"],
                name="faulty_rgb",
                path=npz_path,
            )
            metadata = (
                json.loads(str(data["metadata_json"]))
                if "metadata_json" in data
                else {}
            )
    except InvalidSampleError:
        raise
    except Exception as exc:
        raise InvalidSampleError(f"Cannot load model input {npz_path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise InvalidSampleError(f"metadata_json in {npz_path} must decode to an object")
    return rgb.astype(np.uint8), metadata


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
