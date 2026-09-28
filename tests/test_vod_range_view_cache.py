import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from Fault_Localization_Model.vod_dataset.vod_io import VODFrame
from scripts.create_vod_range_view_dataset import _ensure_radar_cache


class VoDRangeViewCacheTests(unittest.TestCase):
    def test_lean_radar_cache_preserves_five_scan_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            radar_path = root / "radar.bin"
            lidar_calibration = root / "lidar.txt"
            radar_calibration = root / "radar.txt"
            calibration = "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n"
            lidar_calibration.write_text(calibration, encoding="utf-8")
            radar_calibration.write_text(calibration, encoding="utf-8")
            raw = np.asarray([[5, 0, 0, 2, 0, 1, index]
                              for index in (-4, -3, -2, -1, 0)], dtype=np.float32)
            raw.tofile(radar_path)
            frame = VODFrame(
                frame_id="00001", split="train", lidar_path=root / "lidar.bin",
                radar_path=radar_path, lidar_calibration_path=lidar_calibration,
                radar_calibration_path=radar_calibration,
                radar_variant="radar_5frames_rangeview",
            )
            cache_root = root / "aligned"

            self.assertEqual(_ensure_radar_cache(frame, cache_root, 5), ("created", 5))
            self.assertEqual(_ensure_radar_cache(frame, cache_root, 5), ("cached", 5))
            with np.load(cache_root / "train" / "00001.npz", allow_pickle=False) as archive:
                self.assertEqual(archive.files, ["radar_points", "metadata_json"])
                self.assertEqual(archive["radar_points"].shape, (5, 5))
                metadata = json.loads(str(archive["metadata_json"].item()))
            self.assertEqual(metadata["radar_stack_frames"], 5)
            self.assertEqual(metadata["radar_variant"], "radar_5frames_rangeview")

    def test_warmup_stack_requires_explicit_lower_minimum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            radar_path = root / "radar.bin"
            np.asarray([[5, 0, 0, 2, 0, 1, 0]], dtype=np.float32).tofile(radar_path)
            calibration = "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n"
            for name in ("lidar.txt", "radar.txt"):
                (root / name).write_text(calibration, encoding="utf-8")
            frame = VODFrame(
                frame_id="00001", split="test", lidar_path=root / "lidar.bin",
                radar_path=radar_path,
                lidar_calibration_path=root / "lidar.txt",
                radar_calibration_path=root / "radar.txt",
                radar_variant="radar_5frames_rangeview",
            )

            self.assertIsNone(_ensure_radar_cache(frame, root / "aligned", 5))
            self.assertEqual(_ensure_radar_cache(frame, root / "aligned", 1), ("created", 1))


if __name__ == "__main__":
    unittest.main()
