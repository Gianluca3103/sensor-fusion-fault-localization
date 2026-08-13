import unittest

import numpy as np

from models.reconstruction_head.analyze_radar_bev_information import (
    AnalysisDomain,
    CorrespondenceCounts,
    _distance_to_support,
    _update_domain,
)


class RadarBEVInformationTests(unittest.TestCase):
    def test_exact_and_tolerant_correspondence_are_directional(self):
        radar = np.zeros((5, 5), dtype=bool)
        lidar = np.zeros_like(radar)
        radar[2, 2] = True
        lidar[2, 3] = True
        lidar[0, 0] = True
        domain = np.ones_like(radar)
        resolution = (0.2, 0.2)
        accumulator = CorrespondenceCounts()
        accumulator.update(
            radar,
            lidar,
            _distance_to_support(lidar, resolution),
            _distance_to_support(radar, resolution),
            domain,
        )
        result = accumulator.summary()
        self.assertEqual(
            result["p_clean_lidar_occupied_given_radar_occupied"], 0.0
        )
        self.assertEqual(
            result["radar_to_lidar_correspondence_at_0_2m"], 1.0
        )
        self.assertEqual(result["lidar_to_radar_coverage_at_0_2m"], 0.5)

    def test_domain_restricts_both_source_and_target_support(self):
        clean = np.zeros((3, 5, 5), dtype=np.float32)
        radar = np.zeros((4, 5, 5), dtype=np.float32)
        radar[0, 2, 2] = 1.0
        clean[0, 2, 3] = 1.0
        domain = np.zeros((5, 5), dtype=bool)
        domain[2, 2] = True
        ranges = {"all": np.ones((5, 5), dtype=bool)}
        for name in ("0_15m", "15_30m", "30_45m", "45_60m", "over_60m"):
            ranges[name] = np.zeros((5, 5), dtype=bool)
        accumulator = AnalysisDomain(scatter_limit=10, seed=1)
        _update_domain(
            accumulator,
            clean,
            radar,
            domain,
            ranges,
            (0.2, 0.2),
            0.5,
            0.0,
            None,
        )
        result = accumulator.summary()["static_occupancy"]["overall"]
        self.assertEqual(result["radar_occupied_cells"], 1)
        self.assertEqual(
            result["radar_to_lidar_correspondence_at_1_0m"], 0.0
        )

    def test_height_and_power_use_supported_cells(self):
        clean = np.zeros((3, 3, 3), dtype=np.float32)
        radar = np.zeros((4, 3, 3), dtype=np.float32)
        clean[:, 1, 1] = (1.0, 0.4, 0.7)
        radar[:, 1, 1] = (1.0, 0.5, 0.2, 0.6)
        domain = np.ones((3, 3), dtype=bool)
        ranges = {"all": domain}
        for name in ("0_15m", "15_30m", "30_45m", "45_60m", "over_60m"):
            ranges[name] = np.zeros_like(domain)
        accumulator = AnalysisDomain(scatter_limit=10, seed=1)
        _update_domain(
            accumulator,
            clean,
            radar,
            domain,
            ranges,
            (0.2, 0.2),
            0.5,
            0.0,
            None,
        )
        result = accumulator.summary()
        self.assertAlmostEqual(
            result["clean_lidar_at_static_radar_cells"]["mean_clean_normalized_density"],
            0.4,
        )
        self.assertAlmostEqual(
            result["radar_height_vs_clean_lidar_height"]["mae"], 0.1
        )


if __name__ == "__main__":
    unittest.main()
