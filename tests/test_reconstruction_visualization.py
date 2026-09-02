import unittest

import numpy as np
import torch

from models.two_stage_reconstruction_head.reconstruction_visualization import (
    occupancy_image,
    radar_lidar_occupancy_overlay,
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


if __name__ == "__main__":
    unittest.main()
