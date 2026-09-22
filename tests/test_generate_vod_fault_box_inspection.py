import unittest

from scripts.generate_vod_fault_box_inspection import _selected_indices


class GenerateVodFaultBoxInspectionTest(unittest.TestCase):
    def test_unique_deterministic_indices(self):
        first = _selected_indices(100, 50, 42)
        second = _selected_indices(100, 50, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 50)
        self.assertEqual(len(set(first)), 50)


if __name__ == "__main__":
    unittest.main()
