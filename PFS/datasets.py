import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from Fault_Localization_Model.sample_utils import (
    InvalidSampleError,
    validate_heatmap_array,
    validate_rgb_array,
)
from Fault_Localization_Model.model_blocks import resize_reliability_map


class PFSReliabilityDataset(Dataset):
    def __init__(self, paths, resize_hw):
        self.paths = list(paths)
        self.resize_hw = resize_hw
        if not self.paths:
            raise FileNotFoundError("No .npz reliability-map samples found.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        try:
            with np.load(path, allow_pickle=False) as data:
                required = {"faulty_rgb", "clean_rgb", "fault_heatmap", "metadata_json"}
                missing = required - set(data.files)
                if missing:
                    raise KeyError(f"missing arrays: {', '.join(sorted(missing))}")
                faulty_rgb = validate_rgb_array(
                    data["faulty_rgb"], name="faulty_rgb", path=path
                )
                clean_rgb = validate_rgb_array(
                    data["clean_rgb"], name="clean_rgb", path=path
                )
                target = validate_heatmap_array(data["fault_heatmap"], path=path)
                metadata = json.loads(str(data["metadata_json"]))
        except InvalidSampleError:
            raise
        except Exception as exc:
            raise InvalidSampleError(f"Cannot load training sample {path}: {exc}") from exc

        if faulty_rgb.shape != clean_rgb.shape:
            raise InvalidSampleError(
                f"faulty_rgb and clean_rgb in {path} have different shapes: "
                f"{faulty_rgb.shape} and {clean_rgb.shape}"
            )
        if not isinstance(metadata, dict):
            raise InvalidSampleError(f"metadata_json in {path} must decode to an object")

        faulty_rgb = faulty_rgb.astype(np.float32) / 255.0
        clean_rgb = clean_rgb.astype(np.float32) / 255.0
        target = target.astype(np.float32)

        faulty = torch.from_numpy(faulty_rgb.transpose(2, 0, 1))
        clean = torch.from_numpy(clean_rgb.transpose(2, 0, 1))
        target = torch.from_numpy(target).unsqueeze(0)
        if self.resize_hw:
            faulty = F.interpolate(
                faulty[None],
                size=self.resize_hw,
                mode="bilinear",
                align_corners=False,
            )[0]
            clean = F.interpolate(
                clean[None],
                size=self.resize_hw,
                mode="bilinear",
                align_corners=False,
            )[0]
            target = resize_reliability_map(target[None], self.resize_hw)[0]
        return {
            "x": faulty,
            "clean": clean,
            "y": target,
            "rgb": (faulty_rgb * 255).astype(np.uint8),
            "path": str(path),
            "metadata": metadata,
        }


def collate_reliability_batch(batch):
    return {
        "x": torch.stack([item["x"] for item in batch]),
        "clean": torch.stack([item["clean"] for item in batch]),
        "y": torch.stack([item["y"] for item in batch]),
        "rgb": [item["rgb"] for item in batch],
        "path": [item["path"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }
