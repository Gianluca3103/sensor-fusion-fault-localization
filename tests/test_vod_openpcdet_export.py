from pathlib import Path
import unittest

import numpy as np

from Fault_Localization_Model.vod_dataset import load_vod_split_ids


class VODOpenPCDetExportTests(unittest.TestCase):
    def test_vod_split_loader_preserves_official_order(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_sets = root / "lidar" / "ImageSets"
            image_sets.mkdir(parents=True)
            (image_sets / "train.txt").write_text("00002\n00001\n", encoding="utf-8")
            self.assertEqual(load_vod_split_ids(root, "train"), ["00002", "00001"])

    def test_openpcdet_point_files_are_four_float_features(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            points = np.asarray([[1.0, 2.0, 3.0, 0.25]], dtype=np.float32)
            destination = root / "points" / "00001.npy"
            destination.parent.mkdir()
            np.save(destination, points, allow_pickle=False)
            loaded = np.load(destination, allow_pickle=False)
            self.assertEqual(loaded.shape, (1, 4))
            self.assertEqual(loaded.dtype, np.float32)
