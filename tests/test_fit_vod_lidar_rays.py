"""The fitted VoD ray grid must preserve scan structure before training."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from models.two_stage_reconstruction_head.range_view.geometry import RangeGeometry
from scripts.fit_vod_lidar_rays import (
    audit_geometry, fit_beam_elevations, fit_virtual_elevations,
)


class VoDRayFitTests(unittest.TestCase):
    def test_virtual_rows_cover_sparse_elevation_tails(self) -> None:
        rng = np.random.default_rng(7)
        angles = np.r_[rng.normal(-20, 0.2, 100),
                       rng.normal(-5, 1.0, 10000),
                       rng.normal(4, 0.2, 100)]
        histogram, edges = np.histogram(angles, bins=np.linspace(-25, 10, 3501))
        fitted = fit_virtual_elevations(histogram, edges, rows=32)
        self.assertLess(fitted[0], -19)
        self.assertGreater(fitted[-1], 3)
        self.assertTrue(np.all(np.diff(fitted) > 0))

    def test_fit_uneven_beam_angles(self) -> None:
        expected = np.array([-19.3, -11.7, -7.2, -2.4, 1.1, 3.4])
        rng = np.random.default_rng(8)
        angles = np.concatenate([
            rng.normal(beam, 0.025, size=2500) for beam in expected
        ])
        histogram, edges = np.histogram(angles, bins=np.linspace(-25, 10, 3501))
        fitted = fit_beam_elevations(
            histogram, edges, beams=len(expected),
            min_separation_deg=0.15, smoothing_deg=0.025)
        np.testing.assert_allclose(fitted, expected, atol=0.02)

    def test_audit_detects_capacity_and_low_projection_error(self) -> None:
        geometry = RangeGeometry(
            beam_elevations_rad=tuple(np.deg2rad([-12, -3, 2])),
            azimuth_bins=128, min_range_m=0.5, max_range_m=80,
            azimuth_span_rad=np.pi, azimuth_offset_rad=-np.pi / 2,
        )
        directions = geometry.ray_directions()
        rows, cols = np.meshgrid(np.arange(3), np.arange(64), indexing="ij")
        xyz = directions[rows, cols] * 20.0
        scan = np.column_stack((xyz.reshape(-1, 3), np.ones(192))).astype(np.float32)
        duplicated = np.concatenate((scan, scan), axis=0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scan.bin"
            duplicated.tofile(path)
            audit = audit_geometry([path], geometry)
        self.assertEqual(audit["scans"], 1)
        self.assertEqual(audit["forward_points"], 192)
        self.assertEqual(audit["raw_forward_points"], 384)
        self.assertEqual(audit["duplicate_fraction"], 0.5)
        self.assertEqual(audit["assigned_fraction"], 1)
        self.assertEqual(audit["collision_fraction"], 0)
        self.assertLess(audit["roundtrip_p95_m"], 1e-4)


if __name__ == "__main__":
    unittest.main()
