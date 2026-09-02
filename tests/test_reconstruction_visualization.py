import unittest

import numpy as np
import torch

from models.two_stage_reconstruction_head.reconstruction_visualization import (
    occupancy_image,
    radar_lidar_occupancy_overlay,
)
from models.two_stage_reconstruction_head.diffusion_process.diffusion_metrics import (
    reconstruction_mask_boundary_bands,
)


class ReconstructionVisualizationTests(unittest.TestCase):
    def test_occupancy_visualization_uses_only_first_lidar_channel(self):
        bev = torch.zeros(3, 2, 2)
        bev[0, 0, 0] = 1.0
        bev[1:, :, :] = 1.0
        image = occupancy_image(bev)
        np.testing.assert_array_equal(
            image, np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        )

    def test_radar_overlay_has_stable_semantic_colors(self):
        lidar = torch.zeros(3, 2, 2)
        radar = torch.zeros(4, 2, 2)
        lidar[0, 0, 0] = 1.0
        radar[0, 0, 1] = 1.0
        lidar[0, 1, 1] = 1.0
        radar[2, 1, 1] = 0.5

        overlay = radar_lidar_occupancy_overlay(lidar, radar)

        np.testing.assert_array_equal(overlay[0, 0], [0.0, 1.0, 1.0])
        np.testing.assert_array_equal(overlay[0, 1], [1.0, 0.0, 1.0])
        np.testing.assert_array_equal(overlay[1, 1], [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(overlay[1, 0], [0.0, 0.0, 0.0])

    def test_overlay_threshold_reveals_lower_confidence_reconstruction(self):
        lidar = torch.zeros(3, 1, 1)
        radar = torch.zeros(4, 1, 1)
        lidar[0, 0, 0] = 0.4
        high_threshold = radar_lidar_occupancy_overlay(
            lidar, radar, occupancy_threshold=0.5
        )
        low_threshold = radar_lidar_occupancy_overlay(
            lidar, radar, occupancy_threshold=0.3
        )
        np.testing.assert_array_equal(high_threshold[0, 0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(low_threshold[0, 0], [0.0, 1.0, 1.0])

    def test_boundary_bands_cover_mask_once_at_expected_depths(self):
        mask = torch.ones(1, 1, 10, 10)
        bands = reconstruction_mask_boundary_bands(mask)
        self.assertEqual(int(bands["0-1 cells"].sum()), 64)
        self.assertEqual(int(bands["2-3 cells"].sum()), 32)
        self.assertEqual(int(bands["4-7 cells"].sum()), 4)
        self.assertEqual(int(bands["8+ cells"].sum()), 0)
        coverage = sum(band.int() for band in bands.values())
        self.assertTrue(torch.equal(coverage, mask.int()))


if __name__ == "__main__":
    unittest.main()
