import unittest

import numpy as np

from scripts.visualize_lidar_fault_boxes_3d import (
    _merge_boxes_to_limit,
    _removed_clean_points,
    _voxel_boxes_to_metric,
)
from voxelization import VoxelGridConfig


class LidarFaultBoxVisualizationTests(unittest.TestCase):
    def test_merge_boxes_never_discards_source_extents(self):
        minimum = np.asarray([[0, 0, 0], [0, 0, 2], [4, 4, 4]], dtype=np.int32)
        maximum = minimum + 1
        merged_min, merged_max = _merge_boxes_to_limit(
            minimum, maximum, 2, (1.0, 1.0, 1.0)
        )
        self.assertEqual(len(merged_min), 2)
        for lower, upper in zip(minimum, maximum):
            enclosed = np.all(merged_min <= lower, axis=1) & np.all(merged_max >= upper, axis=1)
            self.assertTrue(enclosed.any())

    def test_one_box_is_global_enclosure(self):
        minimum = np.asarray([[1, 2, 3], [4, 0, 8]], dtype=np.int32)
        maximum = np.asarray([[2, 5, 7], [6, 2, 9]], dtype=np.int32)
        merged_min, merged_max = _merge_boxes_to_limit(
            minimum, maximum, 1, (0.2, 0.2, 0.25)
        )
        self.assertTrue(np.array_equal(merged_min, [[1, 0, 3]]))
        self.assertTrue(np.array_equal(merged_max, [[6, 5, 9]]))

    def test_final_boxes_have_no_positive_volume_overlap(self):
        minimum = np.asarray(
            [[0, 0, 0], [0, 0, 4], [0, 3, 2], [0, 3, 6]], dtype=np.int32
        )
        maximum = np.asarray(
            [[2, 4, 3], [2, 4, 7], [2, 7, 5], [2, 7, 9]], dtype=np.int32
        )
        merged_min, merged_max = _merge_boxes_to_limit(
            minimum, maximum, 2, (1.0, 1.0, 1.0)
        )
        for first in range(len(merged_min) - 1):
            for second in range(first + 1, len(merged_min)):
                intersection_lower = np.maximum(merged_min[first], merged_min[second])
                intersection_upper = np.minimum(merged_max[first], merged_max[second])
                self.assertFalse(np.all(intersection_lower < intersection_upper))

    def test_removed_points_come_from_missing_source_ids(self):
        clean = np.asarray(
            [[1, 0, 0, 1], [2, 0, 0, 2], [3, 0, 0, 3]], dtype=np.float32
        )
        removed = _removed_clean_points(clean, np.asarray([0, 2], dtype=np.int64))
        self.assertTrue(np.array_equal(removed, clean[[1]]))

    def test_zyx_box_converts_to_metric_xyz_boundaries(self):
        grid = VoxelGridConfig(
            x_range=(0, 4), y_range=(-2, 2), z_range=(-1, 1),
            voxel_size=(1, 1, 1),
        )
        boxes = _voxel_boxes_to_metric(
            np.asarray([[0, 1, 2]]), np.asarray([[2, 3, 4]]), grid
        )
        self.assertTrue(np.array_equal(boxes[0][0], [2, -1, -1]))
        self.assertTrue(np.array_equal(boxes[0][1], [4, 1, 1]))


if __name__ == "__main__":
    unittest.main()
