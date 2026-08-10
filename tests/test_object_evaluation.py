import unittest

import numpy as np

from Fault_Localization_Model.kradar_dataset import KRadarObjectAnnotation
from models.reconstruction_head.object_evaluation import (
    BEVGeometry,
    object_mask_overlap,
    oriented_box_corners_xy,
    summarize_object_overlaps,
)


def annotation(**overrides):
    values = {
        "class_name": "Sedan",
        "x": 2.0,
        "y": 0.0,
        "z": 0.0,
        "yaw": 0.0,
        "length": 2.0,
        "width": 2.0,
        "height": 1.5,
    }
    values.update(overrides)
    return KRadarObjectAnnotation(**values)


class ObjectEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.geometry = BEVGeometry((0.0, 4.0), (-2.0, 2.0), 4, 4)

    def test_axis_aligned_object_has_exact_partial_coverage(self):
        mask = np.zeros((4, 4), dtype=bool)
        mask[1, 1] = True
        overlap = object_mask_overlap(annotation(), mask, self.geometry)
        self.assertTrue(overlap.any_overlap)
        self.assertAlmostEqual(overlap.overlap_area_m2, 1.0)
        self.assertAlmostEqual(overlap.affected_fraction, 0.25)

    def test_row_zero_corresponds_to_maximum_x(self):
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 1] = True
        overlap = object_mask_overlap(
            annotation(x=3.5, y=-0.5, length=1.0, width=1.0),
            mask,
            self.geometry,
        )
        self.assertAlmostEqual(overlap.affected_fraction, 1.0)

    def test_rotated_corners_preserve_footprint_area(self):
        box = annotation(yaw=np.pi / 4.0, length=4.0, width=2.0)
        corners = oriented_box_corners_xy(box)
        x = corners[:, 0]
        y = corners[:, 1]
        area = 0.5 * abs(
            np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
        )
        self.assertAlmostEqual(area, 8.0)

    def test_summary_separates_unaffected_objects(self):
        records = [
            {"class": "Sedan", "any_overlap": False, "affected_fraction": 0.0},
            {"class": "Sedan", "any_overlap": True, "affected_fraction": 0.5},
        ]
        summary = summarize_object_overlaps(records)
        self.assertEqual(summary["overall"]["affected_objects"], 1)
        self.assertEqual(summary["by_class"]["Sedan"]["objects"], 2)
        self.assertEqual(
            summary["by_affected_fraction"]["unaffected"]["objects"], 1
        )


if __name__ == "__main__":
    unittest.main()
