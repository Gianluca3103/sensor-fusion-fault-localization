import tempfile
import unittest
from pathlib import Path
import numpy as np
from Fault_Localization_Model.io_utils import atomic_savez, atomic_savez_compressed


class FastNpzTests(unittest.TestCase):
    def test_all_compression_modes_preserve_arrays_and_dtypes(self):
        arrays = {'points': np.random.default_rng(7).normal(size=(200, 4)).astype(np.float32),
                  'height': np.array([0, 1, np.nan], dtype=np.float16),
                  'counts': np.array([0, 2**32-1], dtype=np.uint32),
                  'metadata_json': np.asarray('{"scene":"example"}'),
                  'empty': np.empty((0, 3), dtype=np.float32)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_savez_compressed(root / 'reference.npz', **arrays)
            for level in (0, 1, 6, 9):
                target = root / f'level{level}.npz'
                atomic_savez(target, compression_level=level, **arrays)
                with np.load(target, allow_pickle=False) as actual:
                    self.assertEqual(set(actual.files), set(arrays))
                    for key, expected in arrays.items():
                        self.assertEqual(actual[key].dtype, expected.dtype)
                        np.testing.assert_array_equal(actual[key], expected)
            self.assertFalse(list(root.glob('*.tmp*')))

    def test_bad_level_rejected(self):
        with self.assertRaises(ValueError):
            atomic_savez('unused.npz', compression_level=10)
