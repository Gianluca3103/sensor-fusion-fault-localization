import unittest

import numpy as np

from Fault_Localization_Model.bev_utils import make_rgb_preview, project_lidar_bev


class BevUtilsTests(unittest.TestCase):
    def test_projection_clamps_float32_upper_edge_rounding(self):
        lateral = np.nextafter(
            np.float32(32.0),
            np.float32(-np.inf),
            dtype=np.float32,
        )
        points = np.asarray(
            [[1.0, lateral, 0.0, 1.0]],
            dtype=np.float32,
        )

        layers = project_lidar_bev(
            points,
            x_range=(0.0, 64.0),
            y_range=(-32.0, 32.0),
            resolution=0.2,
        )

        self.assertEqual(float(layers["raw_density"].sum()), 1.0)
        self.assertEqual(float(layers["raw_density"][:, -1].sum()), 1.0)

    def test_projection_outputs_occupancy_density_and_robust_upper_height(self):
        points = np.asarray(
            [
                [1.0, 0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0, 0.5],
                [1.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 100.0, 1.0],
            ],
            dtype=np.float32,
        )

        layers = project_lidar_bev(
            points,
            x_range=(0.0, 2.0),
            y_range=(-1.0, 1.0),
            resolution=1.0,
        )

        occupied = layers["occupancy"] > 0
        self.assertEqual(int(occupied.sum()), 1)
        self.assertEqual(float(layers["occupancy"][occupied][0]), 1.0)
        self.assertEqual(float(layers["density"][occupied][0]), 1.0)
        self.assertAlmostEqual(
            float(layers["robust_upper_height"][occupied][0]),
            0.5,
        )
        self.assertNotIn("intensity", layers)
        self.assertNotIn("height", layers)

        preview = make_rgb_preview(layers)
        self.assertEqual(preview.shape, (2, 2, 3))
        self.assertEqual(int(preview[..., 0].max()), 255)
        self.assertEqual(int(preview[..., 1].max()), 255)
        self.assertEqual(int(preview[..., 2].max()), 127)


if __name__ == "__main__":
    unittest.main()
