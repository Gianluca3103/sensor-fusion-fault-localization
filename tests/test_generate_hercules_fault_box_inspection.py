import unittest

from scripts.generate_hercules_fault_box_inspection import _selected_indices


class HerculesFaultBoxInspectionTests(unittest.TestCase):
    def test_selection_is_unique_and_repeatable(self):
        first = _selected_indices(100, 50, 42)
        second = _selected_indices(100, 50, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 50)
        self.assertEqual(len(set(first)), 50)

    def test_selection_rejects_more_samples_than_frames(self):
        with self.assertRaises(ValueError):
            _selected_indices(4, 5, 42)


if __name__ == "__main__":
    unittest.main()
