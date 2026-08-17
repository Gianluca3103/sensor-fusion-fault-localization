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
from models.reconstruction_head.fault_selector import (
    _adaptive_halo,
    _best_repair_box,
)


class ReconstructionEncoderTests(unittest.TestCase):
    def test_incremental_halo_matches_large_kernel_reference(self):
        from scipy.ndimage import binary_dilation

        shape = (32, 30)
        reconstruction = np.zeros(shape, dtype=bool)
        reconstruction[5:16, 8:19] = True
        trusted = np.zeros(shape, dtype=bool)
        trusted[2:22:2, 4:26:3] = True
        occupied = trusted | reconstruction
        support = np.ones(shape, dtype=bool)
        support[:, :2] = False
        config = FaultSelectorConfig(
            min_halo_healthy_fraction=0.75,
            min_halo_healthy_cells=30,
            min_halo_context_ratio=0.25,
            min_halo_width_cells=2,
            max_halo_dilation_cells=8,
        )

        def reference():
            required = max(
                config.min_halo_healthy_cells,
                int(np.ceil(80 * config.min_halo_context_ratio)),
            )
            best = None
            previous = None
            for amount in range(
                config.min_halo_width_cells,
                config.max_halo_dilation_cells + 1,
            ):
                expanded = binary_dilation(
                    reconstruction,
                    structure=np.ones(
                        (2 * amount + 1, 2 * amount + 1), dtype=bool
                    ),
                )
                halo = expanded & support & ~reconstruction
                if previous is not None and np.array_equal(halo, previous):
                    break
                previous = halo
                healthy_count = int((halo & trusted).sum())
                occupied_count = int((halo & occupied).sum())
                fraction = healthy_count / occupied_count if occupied_count else 0.0
                met = healthy_count >= required and fraction >= 0.75
                candidate = (halo, amount, healthy_count, fraction, met)
                if best is None or (
                    met,
                    min(healthy_count, required),
                    fraction,
                    -amount,
                ) > (
                    best[4],
                    min(best[2], required),
                    best[3],
                    -best[1],
                ):
                    best = candidate
                if met:
                    break
            return best

        actual = _adaptive_halo(
            reconstruction,
            80,
            trusted,
            occupied,
            support,
            config,
        )
        expected = reference()
        self.assertTrue(np.array_equal(actual[0], expected[0]))
        self.assertEqual(actual[1:], expected[1:])

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

    def test_dataset_loads_and_normalizes_existing_artifact_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path = root / "sample.npz"
            radar_root = root / "radar"
            radar_path = radar_root / "train" / "00123.npz"
            radar_path.parent.mkdir(parents=True)
            metadata = {
                "dataset": "view-of-delft",
                "split": "train",
                "frame_id": "123",
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

    def test_fault_selector_adds_two_coherent_secondary_repair_boxes(self):
        faulty = np.zeros((32, 12), dtype=np.float32)
        missing = np.zeros_like(faulty)
        missing[8:13, 4:8] = 10
        missing[2:5, 4:7] = 7
        faulty[2:5, 4:7] = 3
        missing[17:20, 5:8] = 7
        faulty[17:20, 5:8] = 3
        missing[31, 11] = 7
        faulty[31, 11] = 3
        faulty[5:8, 0] = 1
        faulty[13:17, 0] = 1

        selection = self._select_loss(
            faulty,
            missing,
            min_lidar_loss_fraction=0.8,
            min_repair_box_cells=5,
            max_secondary_repair_boxes=2,
            min_secondary_lidar_loss_fraction=0.7,
            min_secondary_repair_cells=5,
            min_halo_healthy_fraction=0.0,
            min_halo_healthy_cells=0,
            min_halo_context_ratio=0.0,
            min_halo_width_cells=1,
            max_halo_dilation_cells=1,
        )

        self.assertTrue(selection.reconstruction_mask[8:13, 4:8].all())
        self.assertTrue(selection.reconstruction_mask[2:5, 4:7].all())
        self.assertTrue(selection.reconstruction_mask[17:20, 5:8].all())
        self.assertFalse(selection.reconstruction_mask[31, 11])
        self.assertEqual(len(selection.selected_blobs), 3)
        self.assertEqual(selection.selected_blobs[0].bbox, (2, 4, 5, 7))
        self.assertEqual(selection.selected_blobs[1].bbox, (17, 5, 20, 8))
        self.assertEqual(selection.selected_blobs[2].bbox, (8, 4, 13, 8))

    def test_secondary_box_can_merge_related_faults_at_lower_purity(self):
        faulty = np.zeros((30, 20), dtype=np.float32)
        missing = np.zeros_like(faulty)
        missing[0:5, 0:5] = 10
        missing[10:13, 2:5] = 7
        faulty[10:13, 2:5] = 3
        missing[10:13, 10:13] = 7
        faulty[10:13, 10:13] = 3
        faulty[6:8, 6:9] = 1

        selection = self._select_loss(
            faulty,
            missing,
            min_lidar_loss_fraction=0.8,
            min_repair_box_cells=5,
            min_repair_fault_fraction=0.95,
            max_secondary_repair_boxes=2,
            min_secondary_lidar_loss_fraction=0.7,
            min_secondary_repair_fault_fraction=0.7,
            min_secondary_repair_cells=5,
            secondary_merge_gap_cells=8,
            min_halo_healthy_fraction=0.0,
            min_halo_healthy_cells=0,
            min_halo_context_ratio=0.0,
            min_halo_width_cells=1,
            max_halo_dilation_cells=1,
        )

        self.assertEqual(len(selection.selected_blobs), 2)
        self.assertEqual(selection.selected_blobs[0].bbox, (10, 2, 13, 13))
        self.assertTrue(selection.reconstruction_mask[10:13, 2:5].all())
        self.assertTrue(selection.reconstruction_mask[10:13, 10:13].all())
        self.assertGreaterEqual(
            selection.selected_blobs[0].repair_fault_fraction,
            0.7,
        )

    def test_secondary_box_expands_to_configured_minimum_side_length(self):
        faulty = np.zeros((40, 40), dtype=np.float32)
        missing = np.zeros_like(faulty)
        missing[0:5, 0:5] = 10
        missing[20:22, 25:28] = 7
        faulty[20:22, 25:28] = 3

        selection = self._select_loss(
            faulty,
            missing,
            min_lidar_loss_fraction=0.8,
            min_repair_box_cells=5,
            max_secondary_repair_boxes=4,
            min_secondary_lidar_loss_fraction=0.7,
            min_secondary_repair_fault_fraction=0.7,
            min_secondary_repair_cells=5,
            min_secondary_box_side_cells=10,
            min_halo_healthy_fraction=0.0,
            min_halo_healthy_cells=0,
            min_halo_context_ratio=0.0,
            min_halo_width_cells=1,
            max_halo_dilation_cells=1,
        )

        secondary = selection.selected_blobs[0].bbox
        self.assertEqual(secondary, (16, 22, 26, 32))
        self.assertEqual(secondary[2] - secondary[0], 10)
        self.assertEqual(secondary[3] - secondary[1], 10)
        self.assertTrue(selection.reconstruction_mask[20:22, 25:28].all())

    def test_expanded_secondary_boxes_are_merged_when_they_overlap(self):
        faulty = np.zeros((40, 40), dtype=np.float32)
        missing = np.zeros_like(faulty)
        missing[0:5, 0:5] = 10
        missing[12:14, 20:22] = 7
        faulty[12:14, 20:22] = 3
        missing[20:22, 20:22] = 7
        faulty[20:22, 20:22] = 3

        selection = self._select_loss(
            faulty,
            missing,
            min_lidar_loss_fraction=0.8,
            min_repair_box_cells=5,
            max_secondary_repair_boxes=4,
            min_secondary_lidar_loss_fraction=0.7,
            min_secondary_repair_fault_fraction=0.7,
            min_secondary_repair_cells=4,
            secondary_merge_gap_cells=0,
            min_secondary_box_side_cells=10,
            min_halo_healthy_fraction=0.0,
            min_halo_healthy_cells=0,
            min_halo_context_ratio=0.0,
            min_halo_width_cells=1,
            max_halo_dilation_cells=1,
        )

        self.assertEqual(len(selection.selected_blobs), 2)
        self.assertEqual(selection.selected_blobs[0].bbox, (8, 16, 26, 26))
        self.assertTrue(selection.reconstruction_mask[12:14, 20:22].all())
        self.assertTrue(selection.reconstruction_mask[20:22, 20:22].all())

    def test_primary_box_may_span_sparse_corner_faults(self):
        faulty = np.zeros((100, 100), dtype=np.float32)
        missing = np.zeros_like(faulty)
        missing[5:8, 5:8] = 10
        missing[90:93, 90:93] = 10

        selection = self._select_loss(
            faulty,
            missing,
            min_lidar_loss_fraction=0.8,
            min_repair_box_cells=5,
            min_repair_fault_fraction=0.95,
            primary_expansion_gap_cells=12,
            max_secondary_repair_boxes=2,
            min_secondary_lidar_loss_fraction=0.7,
            min_secondary_repair_fault_fraction=0.7,
            min_secondary_repair_cells=5,
            secondary_merge_gap_cells=8,
            min_halo_healthy_fraction=0.0,
            min_halo_healthy_cells=0,
            min_halo_context_ratio=0.0,
            min_halo_width_cells=1,
            max_halo_dilation_cells=1,
        )

        self.assertEqual(
            [blob.bbox for blob in selection.selected_blobs],
            [(5, 5, 93, 93)],
        )
        self.assertTrue(selection.reconstruction_mask[50, 50])

    def test_primary_box_absorbs_nearby_lower_threshold_faults(self):
        faulty = np.zeros((40, 40), dtype=np.float32)
        missing = np.zeros_like(faulty)
        missing[10:15, 10:15] = 10
        missing[18:21, 12:16] = 7
        faulty[18:21, 12:16] = 3

        selection = self._select_loss(
            faulty,
            missing,
            min_lidar_loss_fraction=0.8,
            min_repair_box_cells=5,
            min_repair_fault_fraction=0.95,
            primary_expansion_gap_cells=12,
            max_secondary_repair_boxes=2,
            min_secondary_lidar_loss_fraction=0.7,
            min_secondary_repair_fault_fraction=0.7,
            min_secondary_repair_cells=5,
            min_halo_healthy_fraction=0.0,
            min_halo_healthy_cells=0,
            min_halo_context_ratio=0.0,
            min_halo_width_cells=1,
            max_halo_dilation_cells=1,
        )

        self.assertEqual(len(selection.selected_blobs), 1)
        self.assertEqual(selection.selected_blobs[0].bbox, (10, 10, 21, 16))
        self.assertTrue(selection.reconstruction_mask[18:21, 12:16].all())

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
