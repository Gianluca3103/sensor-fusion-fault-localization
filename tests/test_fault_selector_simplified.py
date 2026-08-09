import unittest

import numpy as np

from models.reconstruction_head import (
    FaultSelectorSimplified,
    SimplifiedFaultSelectorConfig,
)


class SimplifiedFaultSelectorTests(unittest.TestCase):
    def test_preserves_component_shape_instead_of_filling_bbox(self):
        heatmap = np.zeros((12, 12), dtype=np.float32)
        heatmap[3:8, 3] = 1.0
        heatmap[7, 3:8] = 1.0
        occupancy = np.ones_like(heatmap)

        selection = FaultSelectorSimplified(
            SimplifiedFaultSelectorConfig(
                min_blob_cells=1,
                merge_radius_cells=0,
                repair_dilation_cells=0,
                halo_width_cells=1,
                max_blobs=None,
            )
        ).select(
            heatmap,
            reliability_map=1.0 - heatmap,
            faulty_counts=occupancy,
        )

        self.assertTrue(selection.reconstruction_mask[3, 3])
        self.assertTrue(selection.reconstruction_mask[7, 7])
        self.assertFalse(selection.reconstruction_mask[4, 4])
        self.assertEqual(selection.selected_cell_count, 9)

    def test_merges_nearby_faults_without_selecting_dilation_bridge(self):
        heatmap = np.zeros((20, 20), dtype=np.float32)
        heatmap[5:7, 5:7] = 1.0
        heatmap[5:7, 10:12] = 1.0
        heatmap[18, 18] = 1.0

        selection = FaultSelectorSimplified(
            SimplifiedFaultSelectorConfig(
                min_blob_cells=5,
                merge_radius_cells=2,
                repair_dilation_cells=0,
                halo_width_cells=1,
                max_blobs=None,
            )
        ).select(
            heatmap,
            reliability_map=1.0 - heatmap,
            faulty_counts=np.ones_like(heatmap),
        )

        self.assertEqual(len(selection.selected_components), 1)
        self.assertEqual(selection.selected_fault_cell_count, 8)
        self.assertFalse(selection.reconstruction_mask[5, 8])
        self.assertFalse(selection.reconstruction_mask[18, 18])
        self.assertEqual(len(selection.rejected_small_components), 1)

    def test_dilates_repair_and_uses_only_reliable_occupied_halo_cells(self):
        heatmap = np.zeros((15, 15), dtype=np.float32)
        heatmap[7, 7] = 1.0
        reliability = np.ones_like(heatmap)
        reliability[4, 4] = 0.5
        occupancy = np.zeros_like(heatmap)
        occupancy[4:11, 4:11] = 1.0
        occupancy[5, 5] = 0.0

        selection = FaultSelectorSimplified(
            SimplifiedFaultSelectorConfig(
                min_blob_cells=1,
                merge_radius_cells=0,
                repair_dilation_cells=1,
                halo_width_cells=2,
                max_blobs=1,
            )
        ).select(
            heatmap,
            reliability_map=reliability,
            faulty_counts=occupancy,
        )

        self.assertEqual(selection.selected_cell_count, 9)
        self.assertFalse(np.any(selection.reconstruction_mask & selection.halo_mask))
        self.assertFalse(selection.healthy_context_mask[5, 5])
        self.assertFalse(selection.healthy_context_mask[4, 4])
        self.assertTrue(selection.healthy_context_mask[5, 6])

    def test_clips_repair_and_halo_to_valid_radar_support(self):
        support = np.tri(10, 10, dtype=bool)
        heatmap = np.zeros((10, 10), dtype=np.float32)
        heatmap[4, 4] = 1.0

        selection = FaultSelectorSimplified(
            SimplifiedFaultSelectorConfig(
                min_blob_cells=1,
                merge_radius_cells=0,
                repair_dilation_cells=3,
                halo_width_cells=3,
            )
        ).select(
            heatmap,
            reliability_map=np.ones_like(heatmap),
            faulty_counts=np.ones_like(heatmap),
            valid_support_mask=support,
        )

        self.assertFalse(np.any(selection.reconstruction_mask & ~support))
        self.assertFalse(np.any(selection.halo_mask & ~support))
        self.assertFalse(np.any(selection.healthy_context_mask & ~support))


if __name__ == "__main__":
    unittest.main()
