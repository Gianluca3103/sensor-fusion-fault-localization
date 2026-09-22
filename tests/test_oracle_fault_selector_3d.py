import unittest

import numpy as np

from voxelization import (
    OracleFaultSelector3DConfig,
    VoxelGridConfig,
    select_oracle_fault_regions_3d,
)


class OracleFaultSelector3DTests(unittest.TestCase):
    def setUp(self):
        self.grid = VoxelGridConfig(
            x_range=(0.0, 5.0),
            y_range=(0.0, 5.0),
            z_range=(0.0, 5.0),
            voxel_size=(1.0, 1.0, 1.0),
        )

    def test_operation_cores_remain_exact_and_halo_is_disjoint(self):
        repair = np.zeros((5, 5, 5), dtype=bool)
        remove = np.zeros_like(repair)
        repair[2, 2, 2] = True
        remove[2, 2, 3] = True
        result = select_oracle_fault_regions_3d(
            repair,
            remove,
            self.grid,
            OracleFaultSelector3DConfig(halo_m=1.0, grouping_radius_m=0.0),
        )
        self.assertTrue(np.array_equal(result.repair_core, repair))
        self.assertTrue(np.array_equal(result.remove_core, remove))
        self.assertFalse(np.any(result.context_halo & result.operation_mask))
        self.assertTrue(np.array_equal(result.context_mask, result.context_halo | result.operation_mask))

    def test_grouping_merges_close_fault_voxels_without_expanding_core(self):
        repair = np.zeros((5, 5, 5), dtype=bool)
        repair[2, 2, 1] = True
        repair[2, 2, 3] = True
        result = select_oracle_fault_regions_3d(
            repair,
            np.zeros_like(repair),
            self.grid,
            OracleFaultSelector3DConfig(halo_m=0.0, grouping_radius_m=1.0),
        )
        self.assertEqual(len(result.components), 1)
        self.assertEqual(int(result.operation_mask.sum()), 2)

    def test_minimum_crop_size_and_grid_clipping(self):
        repair = np.zeros((5, 5, 5), dtype=bool)
        repair[0, 0, 0] = True
        result = select_oracle_fault_regions_3d(
            repair,
            np.zeros_like(repair),
            self.grid,
            OracleFaultSelector3DConfig(
                halo_m=0.0,
                grouping_radius_m=0.0,
                min_crop_shape_zyx=(4, 4, 4),
            ),
        )
        component = result.components[0]
        self.assertEqual(component.crop_min_zyx, (0, 0, 0))
        self.assertEqual(component.crop_max_exclusive_zyx, (4, 4, 4))

    def test_small_components_can_be_filtered(self):
        repair = np.zeros((5, 5, 5), dtype=bool)
        repair[1, 1, 1] = True
        result = select_oracle_fault_regions_3d(
            repair,
            np.zeros_like(repair),
            self.grid,
            OracleFaultSelector3DConfig(
                halo_m=0.0,
                grouping_radius_m=0.0,
                min_component_voxels=2,
            ),
        )
        self.assertFalse(result.operation_mask.any())
        self.assertEqual(result.components, ())


if __name__ == "__main__":
    unittest.main()
