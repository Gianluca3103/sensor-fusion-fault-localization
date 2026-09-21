import unittest

import numpy as np

from voxelization import VoxelGridConfig, build_voxel_fault_targets


class VoxelFaultTargetTests(unittest.TestCase):
    def setUp(self):
        self.grid = VoxelGridConfig(
            x_range=(0.0, 2.0),
            y_range=(0.0, 2.0),
            z_range=(0.0, 2.0),
            voxel_size=(1.0, 1.0, 1.0),
        )

    def test_preserved_point_is_not_a_fault(self):
        clean = np.asarray([[0.2, 0.2, 0.2, 0.5]], dtype=np.float32)
        result = build_voxel_fault_targets(clean, clean.copy(), np.asarray([0]), self.grid)
        self.assertEqual(int(result.preserve_mask.sum()), 1)
        self.assertEqual(int(result.change_mask.sum()), 0)

    def test_removed_point_marks_its_clean_voxel_for_repair(self):
        clean = np.asarray([[0.2, 0.2, 0.2, 0.5]], dtype=np.float32)
        faulty = np.empty((0, 4), dtype=np.float32)
        result = build_voxel_fault_targets(clean, faulty, np.empty(0, dtype=np.int64), self.grid)
        self.assertTrue(result.repair_mask[0, 0, 0])
        self.assertFalse(result.remove_mask[0, 0, 0])
        self.assertEqual(result.repair_fraction[0, 0, 0], 1.0)

    def test_synthetic_point_marks_its_faulty_voxel_for_removal(self):
        clean = np.empty((0, 4), dtype=np.float32)
        faulty = np.asarray([[1.2, 1.2, 1.2, 0.5]], dtype=np.float32)
        result = build_voxel_fault_targets(clean, faulty, np.asarray([-1]), self.grid)
        self.assertTrue(result.remove_mask[1, 1, 1])
        self.assertFalse(result.repair_mask[1, 1, 1])

    def test_moved_point_marks_origin_for_repair_and_destination_for_removal(self):
        clean = np.asarray([[0.2, 0.2, 0.2, 0.5]], dtype=np.float32)
        faulty = np.asarray([[1.2, 1.2, 1.2, 0.5]], dtype=np.float32)
        result = build_voxel_fault_targets(clean, faulty, np.asarray([0]), self.grid)
        self.assertTrue(result.repair_mask[0, 0, 0])
        self.assertTrue(result.remove_mask[1, 1, 1])

    def test_feature_corruption_marks_same_voxel_for_both_operations(self):
        clean = np.asarray([[0.2, 0.2, 0.2, 0.5]], dtype=np.float32)
        faulty = np.asarray([[0.2, 0.2, 0.2, 0.9]], dtype=np.float32)
        result = build_voxel_fault_targets(clean, faulty, np.asarray([0]), self.grid)
        self.assertTrue(result.repair_mask[0, 0, 0])
        self.assertTrue(result.remove_mask[0, 0, 0])


if __name__ == "__main__":
    unittest.main()
