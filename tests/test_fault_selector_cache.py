import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from models.two_stage_reconstruction_head.cache_fault_selector_masks import (
    _discover_sample_paths,
)
from models.two_stage_reconstruction_head import (
    CoarseReconstructionDataset,
    FaultSelectorConfig,
    GeometricAugmentationConfig,
    InvalidSelectorCacheError,
    build_selector_cache_entry,
    load_selector_cache,
    load_selector_inputs,
    selector_cache_path,
)


class FaultSelectorCacheTests(unittest.TestCase):
    def _write_sample(self, root: Path):
        data_root = root / "data"
        sample_path = data_root / "train" / "sample.npz"
        sample_path.parent.mkdir(parents=True)
        radar_root = root / "radar"
        radar_path = radar_root / "train" / "00123.npz"
        radar_path.parent.mkdir(parents=True)

        heatmap = np.zeros((8, 6), dtype=np.float32)
        heatmap[2:6, 2:5] = 1.0
        reliability = np.ones_like(heatmap)
        reliability[heatmap > 0] = 0.0
        missing = (heatmap > 0).astype(np.float32)
        faulty_counts = np.zeros_like(heatmap)
        faulty_counts[1, :] = 1.0
        faulty_counts[6, :] = 1.0
        metadata = {
            "dataset": "View-of-Delft",
            "split": "train",
            "frame_id": "00123",
            "x_range": [0.0, 8.0],
            "y_range": [-3.0, 3.0],
        }
        faulty_density = np.zeros((320, 320), dtype=np.float32)
        faulty_density[::2, ::2] = 1.0
        np.savez_compressed(
            sample_path,
            clean_rgb=np.full((320, 320, 3), 255, dtype=np.uint8),
            faulty_rgb=np.zeros((320, 320, 3), dtype=np.uint8),
            faulty_density=faulty_density,
            fault_heatmap=heatmap,
            reliability_map=reliability,
            faulty_counts=faulty_counts,
            added_faulty_counts=np.zeros_like(heatmap),
            missing_faulty_counts=missing,
            moved_faulty_counts=np.zeros_like(heatmap),
            valid_support_mask=np.ones_like(heatmap, dtype=np.uint8),
            faulty_lidar_points=np.asarray(
                [[1.0, 0.0, 0.0, 12.0]], dtype=np.float32
            ),
            metadata_json=np.asarray(json.dumps(metadata)),
        )
        np.savez_compressed(
            radar_path,
            radar_bev=np.zeros((4, 320, 320), dtype=np.float32),
            radar_points=np.asarray(
                [[1.0, 0.0, 0.0, 0.2, -0.1]], dtype=np.float32
            ),
        )
        return data_root, sample_path, radar_root

    def test_cache_discovery_accepts_flat_and_split_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            flat_sample = root / "flat" / "sample.npz"
            split_sample = root / "split" / "train" / "sample.npz"
            flat_sample.parent.mkdir(parents=True)
            split_sample.parent.mkdir(parents=True)
            np.savez_compressed(flat_sample, value=np.asarray(1))
            np.savez_compressed(split_sample, value=np.asarray(1))

            self.assertEqual(_discover_sample_paths(root / "flat"), [flat_sample])
            self.assertEqual(
                _discover_sample_paths(root / "split"),
                [split_sample],
            )

    def test_cache_round_trip_and_dataset_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root, sample_path, radar_root = self._write_sample(root)
            config = FaultSelectorConfig()

            status = build_selector_cache_entry(
                sample_path,
                data_root,
                config,
            )
            self.assertEqual(status, "created")
            selector_inputs = load_selector_inputs(sample_path)
            self.assertTrue(
                all(array.shape == (8, 6) for array in selector_inputs.values())
            )
            cache_path = selector_cache_path(sample_path, data_root)
            cached = load_selector_cache(
                cache_path,
                config,
            )

            dataset = CoarseReconstructionDataset(
                [sample_path],
                radar_root,
                data_root=data_root,
                selector_config=config,
            )
            item = dataset[0]

            self.assertEqual(item["reconstruction_mask"].shape, (1, 320, 320))
            self.assertEqual(item["halo_mask"].shape, (1, 320, 320))
            self.assertEqual(item["healthy_context_mask"].shape, (1, 320, 320))
            self.assertEqual(item["reconstruction_mask"].dtype, torch.uint8)
            self.assertEqual(item["halo_mask"].dtype, torch.uint8)
            self.assertEqual(item["healthy_context_mask"].dtype, torch.uint8)
            with np.load(sample_path, allow_pickle=False) as sample:
                occupied = sample["faulty_density"] > 0
            context = item["healthy_context_mask"][0].numpy().astype(bool)
            self.assertGreater(int(context.sum()), 0)
            self.assertFalse(np.any(context & ~occupied))
            self.assertEqual(
                int(item["reconstruction_mask"].sum()),
                int(cached["reconstruction_mask"].sum()),
            )
            self.assertNotIn("fault_heatmap", item)
            self.assertNotIn("faulty_counts", item)
            self.assertLess(cache_path.stat().st_size, 50_000)

            point_dataset = CoarseReconstructionDataset(
                [sample_path],
                radar_root,
                data_root=data_root,
                selector_config=config,
                use_pointpillars=True,
            )
            point_item = point_dataset[0]
            self.assertEqual(point_item["faulty_lidar_points"].shape, (1, 4))
            self.assertEqual(point_item["radar_points"].shape, (1, 5))

            augmented_dataset = CoarseReconstructionDataset(
                [sample_path],
                radar_root,
                data_root=data_root,
                selector_config=config,
                augmentation_config=GeometricAugmentationConfig.from_dict(
                    {"enabled": True}
                ),
            )
            self.assertIsNotNone(augmented_dataset.augmentation)
            self.assertIsNone(dataset.augmentation)

    def test_changed_selector_configuration_rejects_stale_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root, sample_path, _radar_root = self._write_sample(root)
            config = FaultSelectorConfig()
            build_selector_cache_entry(
                sample_path,
                data_root,
                config,
            )
            changed = replace(
                config,
                min_lidar_loss_fraction=config.min_lidar_loss_fraction - 0.05,
            )

            with self.assertRaisesRegex(
                InvalidSelectorCacheError,
                "stale",
            ):
                load_selector_cache(
                    selector_cache_path(sample_path, data_root),
                    changed,
                )

    def test_missing_cache_has_precompute_instruction(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.npz"
            with self.assertRaisesRegex(
                InvalidSelectorCacheError,
                "cache_fault_selector_masks",
            ):
                load_selector_cache(
                    missing,
                    FaultSelectorConfig(),
                )


if __name__ == "__main__":
    unittest.main()
