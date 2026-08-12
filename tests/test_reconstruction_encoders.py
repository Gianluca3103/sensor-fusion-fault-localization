import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from models.reconstruction_head import (
    FaultSelector,
    FaultSelectorConfig,
    load_bev_triplet,
)
from models.reconstruction_head.encoders import BEVEncoder
from models.reconstruction_head.fault_selector import _best_repair_box


class ReconstructionEncoderTests(unittest.TestCase):
    def test_vectorized_repair_box_matches_reference_search(self):
        def reference(severe, healthy, min_cells, min_fraction):
            height, _ = severe.shape
            best_score = None
            best_box = None
            for top in range(height):
                for bottom in range(top + 1, height + 1):
                    fault_count = int(severe[top:bottom].sum())
                    if fault_count < min_cells:
                        continue
                    healthy_count = int(healthy[top:bottom].sum())
                    fraction = fault_count / (fault_count + healthy_count)
                    if fraction < min_fraction:
                        continue
                    score = (fault_count, -healthy_count, -(bottom - top), -top)
                    if best_score is None or score > best_score:
                        columns = np.flatnonzero(severe[top:bottom].any(axis=0))
                        best_score = score
                        best_box = (
                            top,
                            int(columns[0]),
                            bottom,
                            int(columns[-1]) + 1,
                        )
            return best_box

        rng = np.random.default_rng(17)
        for _ in range(100):
            severe = rng.random((12, 9)) < 0.12
            healthy = (rng.random((12, 9)) < 0.18) & ~severe
            min_cells = int(rng.integers(1, 5))
            min_fraction = float(rng.choice((0.5, 0.75, 0.95, 1.0)))
            self.assertEqual(
                _best_repair_box(
                    severe,
                    healthy,
                    min_cells,
                    min_fraction,
                ),
                reference(severe, healthy, min_cells, min_fraction),
            )

    def test_bev_encoder_produces_expected_bottleneck(self):
        encoder = BEVEncoder(
            in_channels=3,
            base_channels=4,
            channel_multipliers=(1, 2, 4),
        ).eval()
        with torch.no_grad():
            bottleneck = encoder(torch.zeros(2, 3, 65, 63))

        self.assertEqual(tuple(bottleneck.shape), (2, 16, 17, 16))

    def test_dataset_loads_and_normalizes_existing_artifact_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path = root / "sample.npz"
            radar_root = root / "radar"
            radar_path = radar_root / "Scene01" / "01_Day" / "123.npz"
            radar_path.parent.mkdir(parents=True)
            metadata = {
                "scene": "Scene01",
                "session": "01_Day",
                "timestamp": "123",
            }
            np.savez_compressed(
                sample_path,
                clean_rgb=np.full((8, 6, 3), 255, dtype=np.uint8),
                faulty_rgb=np.zeros((8, 6, 3), dtype=np.uint8),
                fault_heatmap=np.full((8, 6), 0.25, dtype=np.float32),
                reliability_map=np.full((8, 6), 0.75, dtype=np.float32),
                faulty_counts=np.ones((8, 6), dtype=np.float32),
                added_faulty_counts=np.zeros((8, 6), dtype=np.int32),
                missing_faulty_counts=np.ones((8, 6), dtype=np.int32),
                moved_faulty_counts=np.zeros((8, 6), dtype=np.int32),
                observability_confidence=np.full(
                    (8, 6), 0.625, dtype=np.float16
                ),
                metadata_json=np.asarray(json.dumps(metadata)),
            )
            np.savez_compressed(
                radar_path,
                radar_bev=np.full((4, 8, 6), 0.5, dtype=np.float32),
            )

            item = load_bev_triplet(sample_path, radar_root)

        self.assertEqual(item["clean_bev"].shape, (3, 8, 6))
        self.assertEqual(item["radar_bev"].shape, (4, 8, 6))
        self.assertEqual(item["faulty_bev"].shape, (3, 8, 6))
        self.assertEqual(item["observability_confidence"].shape, (1, 8, 6))
        self.assertNotIn("fault_heatmap", item)
        self.assertNotIn("reliability_map", item)
        self.assertNotIn("faulty_counts", item)
        self.assertNotIn("missing_faulty_counts", item)
        self.assertNotIn("metadata_json", item)
        self.assertNotIn("radar_path", item)
        self.assertTrue(torch.all(item["clean_bev"] == 1.0))
        self.assertTrue(torch.all(item["radar_bev"] == 0.5))
        self.assertTrue(torch.all(item["faulty_bev"] == 0.0))
        self.assertTrue(torch.all(item["observability_confidence"] == 0.625))

    def _select_loss(self, faulty, missing, **config):
        zeros = np.zeros_like(faulty, dtype=np.float32)
        original = faulty + missing
        heatmap = np.zeros_like(faulty, dtype=np.float32)
        occupied = original > 0
        heatmap[occupied] = missing[occupied] / original[occupied]
        return FaultSelector(FaultSelectorConfig(**config)).select(
            heatmap,
            reliability_map=1.0 - heatmap,
            faulty_counts=faulty,
            added_faulty_counts=zeros,
            missing_faulty_counts=missing,
            moved_faulty_counts=zeros,
        )

    def test_fault_selector_uses_ninety_five_percent_loss_threshold(self):
        faulty = np.zeros((12, 12), dtype=np.float32)
        missing = np.zeros_like(faulty)
        faulty[3, 3] = 1
        missing[3, 3] = 18  # Below 95% loss: retain the real LiDAR.
        faulty[4, 4] = 1
        missing[4, 4] = 19  # Exactly 95% lost: reconstruct.
        faulty[5, 5] = 1
        missing[5, 5] = 20  # More than 95% lost: reconstruct.

        selection = self._select_loss(
            faulty,
            missing,
            min_repair_box_cells=1,
            min_halo_healthy_fraction=0.0,
            min_halo_healthy_cells=0,
            min_halo_context_ratio=0.0,
            min_halo_width_cells=1,
            max_halo_dilation_cells=2,
        )

        self.assertFalse(selection.reconstruction_mask[3, 3])
        self.assertTrue(selection.reconstruction_mask[4, 4])
        self.assertTrue(selection.reconstruction_mask[5, 5])
        self.assertEqual(selection.selected_blobs[0].bbox, (4, 4, 6, 6))
        self.assertEqual(selection.selected_cell_count, 4)

    def test_fault_selector_uses_exact_cells_without_filling_rectangle(self):
        faulty = np.zeros((12, 12), dtype=np.float32)
        missing = np.zeros_like(faulty)
        faulty[3, 3] = 0
        missing[3, 3] = 1
        faulty[8, 9] = 0
        missing[8, 9] = 1

        selection = self._select_loss(
            faulty,
            missing,
            min_repair_box_cells=1,
            min_halo_healthy_fraction=0.0,
            min_halo_healthy_cells=0,
            min_halo_context_ratio=0.0,
            min_halo_width_cells=1,
            max_halo_dilation_cells=1,
        )

        expected = np.zeros_like(faulty, dtype=bool)
        expected[3:9, 3:10] = True
        self.assertTrue(np.array_equal(selection.reconstruction_mask, expected))

    def test_fault_selector_leaves_severe_cells_out_instead_of_adding_healthy_cells(self):
        faulty = np.zeros((20, 10), dtype=np.float32)
        missing = np.zeros_like(faulty)
        severe = np.zeros_like(faulty, dtype=bool)
        severe[0:5, 0:4] = True
        severe[12, 0:5] = True
        missing[severe] = 1
        faulty[0, 9] = 1
        faulty[12, 5:10] = 1

        selection = self._select_loss(
            faulty,
            missing,
            min_repair_box_cells=2,
            min_halo_healthy_fraction=0.0,
            min_halo_healthy_cells=0,
            min_halo_context_ratio=0.0,
            min_halo_width_cells=1,
            max_halo_dilation_cells=1,
        )

        self.assertEqual(selection.selected_blobs[0].bbox, (0, 0, 5, 4))
        self.assertFalse(selection.reconstruction_mask[0, 9])
        self.assertFalse(selection.reconstruction_mask[12, 0])
        self.assertAlmostEqual(
            selection.selected_blobs[0].repair_fault_fraction,
            1.0,
        )

    def test_fault_selector_ignores_added_points_as_surviving_lidar(self):
        shape = (8, 8)
        faulty = np.zeros(shape, dtype=np.float32)
        missing = np.zeros(shape, dtype=np.float32)
        added = np.zeros(shape, dtype=np.float32)
        faulty[4, 4] = 10
        added[4, 4] = 10
        missing[4, 4] = 2
        heatmap = np.zeros(shape, dtype=np.float32)

        selection = FaultSelector(
            FaultSelectorConfig(
                min_repair_box_cells=1,
                min_halo_healthy_fraction=0.0,
                min_halo_healthy_cells=0,
                min_halo_context_ratio=0.0,
                min_halo_width_cells=1,
                max_halo_dilation_cells=1,
            )
        ).select(
            heatmap,
            faulty_counts=faulty,
            added_faulty_counts=added,
            missing_faulty_counts=missing,
        )

        self.assertTrue(selection.reconstruction_mask[4, 4])

    def test_fault_selector_halo_contains_only_occupied_trusted_cells(self):
        faulty = np.zeros((20, 20), dtype=np.float32)
        missing = np.zeros_like(faulty)
        faulty[9, 8:13] = 1
        faulty[11, 8:13] = 1
        missing[10, 10] = 1

        selection = self._select_loss(
            faulty,
            missing,
            min_repair_box_cells=1,
            min_halo_healthy_fraction=1.0,
            min_halo_healthy_cells=8,
            min_halo_context_ratio=0.0,
            min_halo_width_cells=1,
            max_halo_dilation_cells=4,
        )

        self.assertTrue(selection.selected_blobs[0].halo_target_met)
        self.assertFalse(
            np.any(selection.healthy_context_mask & ~(faulty > 0))
        )
        self.assertFalse(
            np.any(selection.reconstruction_mask & selection.halo_mask)
        )

    def test_fault_selector_clips_masks_to_valid_radar_support(self):
        faulty = np.zeros((10, 10), dtype=np.float32)
        missing = np.ones_like(faulty)
        support = np.tri(10, 10, dtype=bool)
        zeros = np.zeros_like(faulty)

        selection = FaultSelector(
            FaultSelectorConfig(
                min_repair_box_cells=1,
                min_halo_healthy_fraction=0.0,
                min_halo_healthy_cells=0,
                min_halo_context_ratio=0.0,
                min_halo_width_cells=1,
                max_halo_dilation_cells=1,
            )
        ).select(
            np.ones_like(faulty),
            faulty_counts=faulty,
            added_faulty_counts=zeros,
            missing_faulty_counts=missing,
            valid_support_mask=support,
        )

        self.assertTrue(np.array_equal(selection.reconstruction_mask, support))
        self.assertFalse(np.any(selection.halo_mask & ~support))


if __name__ == "__main__":
    unittest.main()
