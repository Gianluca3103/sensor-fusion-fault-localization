import numpy as np
from dataclasses import dataclass
import unittest

from pcdet_integration.reconstructed_points import (
    repair_point_cloud,
    repair_point_cloud_with_clean_points,
)


@dataclass(frozen=True)
class BEVGridGeometry:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    height: int
    width: int

    @property
    def pillar_size_x(self):
        return (self.x_max - self.x_min) / self.height

    @property
    def pillar_size_y(self):
        return (self.y_max - self.y_min) / self.width

    def validate(self):
        assert self.x_max > self.x_min and self.y_max > self.y_min


class ReconstructedPointTests(unittest.TestCase):
    def test_repair_preserves_outside_and_replaces_inside_deterministically(self):
        geometry = BEVGridGeometry(0.0, 2.0, 0.0, 2.0, height=2, width=2)
        # First point is in row 0/col 0. Second is outside the repair mask.
        faulty = np.asarray(
            [[1.75, 0.25, 0.2, 0.4], [0.25, 1.75, -0.1, 0.8]],
            dtype=np.float32,
        )
        repair = np.zeros((2, 2), dtype=np.float32)
        repair[0, 0] = 1.0
        bev = np.zeros((3, 2, 2), dtype=np.float32)
        bev[0, 0, 0] = 0.9
        bev[2, 0, 0] = 0.5

        repaired = repair_point_cloud(faulty, bev, repair, geometry)

        np.testing.assert_array_equal(repaired[0], faulty[1])
        np.testing.assert_allclose(repaired[1, :3], [1.5, 0.5, 1.0])
        self.assertEqual(repaired[1, 3], faulty[1, 3])
        self.assertEqual(repaired.dtype, np.float32)

    def test_repair_with_no_predicted_occupancy_only_removes_masked_points(self):
        geometry = BEVGridGeometry(0.0, 2.0, 0.0, 2.0, height=2, width=2)
        faulty = np.asarray(
            [[1.75, 0.25, 0.2, 0.4], [0.25, 1.75, -0.1, 0.8]],
            dtype=np.float32,
        )
        repair = np.zeros((2, 2), dtype=np.float32)
        repair[0, 0] = 1.0
        bev = np.zeros((3, 2, 2), dtype=np.float32)

        repaired = repair_point_cloud(faulty, bev, repair, geometry)

        np.testing.assert_array_equal(repaired, faulty[1:])

    def test_oracle_raw_repair_uses_clean_points_only_inside_mask(self):
        geometry = BEVGridGeometry(0.0, 2.0, 0.0, 2.0, height=2, width=2)
        faulty = np.asarray(
            [[1.75, 0.25, 0.2, 0.4], [0.25, 1.75, -0.1, 0.8]],
            dtype=np.float32,
        )
        clean = np.asarray(
            [
                [1.65, 0.35, 0.6, 0.3],
                [1.55, 0.45, 0.8, 0.5],
                [0.35, 1.65, -0.2, 0.9],
            ],
            dtype=np.float32,
        )
        repair = np.zeros((2, 2), dtype=np.float32)
        repair[0, 0] = 1.0

        repaired = repair_point_cloud_with_clean_points(
            faulty, clean, repair, geometry
        )

        np.testing.assert_array_equal(repaired[0], faulty[1])
        np.testing.assert_array_equal(repaired[1:], clean[:2])
        self.assertEqual(repaired.dtype, np.float32)
