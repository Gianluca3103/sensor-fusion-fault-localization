import unittest

import numpy as np

from scripts.analyze_oracle_3d_selector_composition import mask_composition


class OracleSelectorCompositionTests(unittest.TestCase):
    def test_operation_and_context_composition_are_reported_separately(self):
        repair = np.zeros((1, 1, 3), dtype=bool)
        repair[0, 0, 0] = True
        preserve = np.zeros_like(repair)
        preserve[0, 0, 1] = True
        masks = {
            "operation_mask": np.array([[[True, False, False]]]),
            "context_mask": np.array([[[True, True, False]]]),
            "context_halo": np.array([[[False, True, False]]]),
        }
        targets = {
            "repair_mask": repair,
            "remove_mask": np.zeros_like(repair),
            "preserve_mask": preserve,
            "clean_occupancy": repair | preserve,
            "faulty_occupancy": preserve.copy(),
        }
        result = mask_composition(masks, targets)
        self.assertEqual(result["operation_missing_repair_voxels"], 1)
        self.assertEqual(result["operation_healthy_clean_voxels"], 0)
        self.assertEqual(result["context_missing_repair_voxels"], 1)
        self.assertEqual(result["context_healthy_clean_voxels"], 1)
        self.assertEqual(result["context_missing_share_of_selected_clean"], 0.5)


if __name__ == "__main__":
    unittest.main()
