import unittest

import numpy as np

from Fault_Localization_Model.vod_dataset import (
    BEVGeometry,
    lidar_analysis_channels,
    radar_analysis_channels,
)


class ViewOfDelftChannelAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.geometry = BEVGeometry(
            x_range=(0.0, 2.0),
            y_range=(-1.0, 1.0),
            resolution=1.0,
        )

    def test_radar_channels_preserve_physical_statistics(self):
        raw = np.asarray(
            [
                [1.2, 0.2, 0.0, 10.0, -2.0, -1.0, -4.0],
                [1.3, 0.2, 2.0, 20.0, 4.0, 3.0, 0.0],
            ],
            dtype=np.float32,
        )
        channels = radar_analysis_channels(raw, raw.copy(), self.geometry)
        cell = (0, 1)

        self.assertEqual(channels["occupancy"][cell], 1.0)
        self.assertEqual(channels["point_count"][cell], 2.0)
        self.assertAlmostEqual(channels["rcs_mean_db"][cell], 15.0)
        self.assertAlmostEqual(channels["rcs_max_db"][cell], 20.0)
        self.assertAlmostEqual(
            channels["raw_radial_velocity_mean_mps"][cell], 1.0
        )
        self.assertAlmostEqual(
            channels["compensated_radial_velocity_mean_mps"][cell], 1.0
        )
        self.assertAlmostEqual(channels["scan_time_index_mean"][cell], -2.0)
        self.assertAlmostEqual(channels["scan_time_index_span"][cell], 4.0)
        self.assertAlmostEqual(
            channels["height_spread_p90_p10_m"][cell], 2.0
        )
        self.assertAlmostEqual(
            channels["doppler_spread_p90_p10_mps"][cell], 6.0
        )

    def test_lidar_channels_report_density_height_and_reflectivity(self):
        points = np.asarray(
            [
                [1.2, 0.2, -1.0, 10.0],
                [1.3, 0.2, 1.0, 20.0],
            ],
            dtype=np.float32,
        )
        channels = lidar_analysis_channels(points, self.geometry)
        cell = (0, 1)

        self.assertEqual(channels["occupancy"][cell], 1.0)
        self.assertEqual(channels["point_count"][cell], 2.0)
        self.assertAlmostEqual(channels["mean_height_m"][cell], 0.0)
        self.assertAlmostEqual(channels["height_std_m"][cell], 1.0)
        self.assertAlmostEqual(
            channels["height_spread_p90_p10_m"][cell], 2.0
        )
        self.assertAlmostEqual(channels["reflectivity_mean"][cell], 15.0)
        self.assertAlmostEqual(channels["reflectivity_max"][cell], 20.0)


if __name__ == "__main__":
    unittest.main()
