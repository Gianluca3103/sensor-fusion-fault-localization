"""Dataset adapter that adds oracle repair and trusted halo masks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .bev_triplets import load_bev_triplet
from .fault_selector import FaultSelector, FaultSelectorConfig


class CoarseReconstructionDataset(Dataset):
    """Load aligned BEVs and derive Stage-I oracle masks with the Fault Selector."""

    def __init__(
        self,
        sample_paths,
        radar_root: str | Path,
        *,
        resize_hw: tuple[int, int] | None = (320, 320),
        selector_config: FaultSelectorConfig | None = None,
    ):
        self.sample_paths = tuple(Path(path) for path in sample_paths)
        if not self.sample_paths:
            raise FileNotFoundError("No coarse reconstruction samples were provided")
        self.radar_root = Path(radar_root)
        self.resize_hw = tuple(resize_hw) if resize_hw is not None else None
        self.selector = FaultSelector(selector_config)

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = load_bev_triplet(
            self.sample_paths[index],
            self.radar_root,
            resize_hw=self.resize_hw,
        )
        selection = self.selector.select(
            item["fault_heatmap"][0].numpy(),
            reliability_map=item["reliability_map"][0].numpy(),
            faulty_counts=item["faulty_counts"][0].numpy(),
            added_faulty_counts=item["added_faulty_counts"][0].numpy(),
            missing_faulty_counts=item["missing_faulty_counts"][0].numpy(),
            moved_faulty_counts=item["moved_faulty_counts"][0].numpy(),
        )
        reconstruction_mask = torch.from_numpy(
            selection.reconstruction_mask.astype(np.float32, copy=False)
        )[None]
        # Only occupied and reliable halo cells are trusted local LiDAR context.
        healthy_context_mask = torch.from_numpy(
            selection.healthy_context_mask.astype(np.float32, copy=False)
        )[None]
        halo_mask = torch.from_numpy(
            selection.halo_mask.astype(np.float32, copy=False)
        )[None]
        item.update(
            {
                "reconstruction_mask": reconstruction_mask,
                "healthy_context_mask": healthy_context_mask,
                "halo_mask": halo_mask,
                "selected_blob_count": len(selection.selected_blobs),
                "reconstruction_cell_count": selection.selected_cell_count,
                "healthy_context_cell_count": selection.healthy_context_cell_count,
            }
        )
        return item
