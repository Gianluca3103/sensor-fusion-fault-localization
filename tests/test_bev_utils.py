import unittest

import numpy as np

from Fault_Localization_Model.bev_utils import project_lidar_bev


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


if __name__ == "__main__":
    unittest.main()
