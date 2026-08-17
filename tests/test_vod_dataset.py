import tempfile
from pathlib import Path
import unittest

import numpy as np

from Fault_Localization_Model.vod_dataset import (
    align_radar_to_lidar,
    discover_vod_frames,
    load_vod_lidar,
    load_vod_radar,
    load_vod_radar_to_lidar,
)
from models.two_stage_reconstruction_head.coarse_dataset import radar_cache_path


def _write_calibration(path: Path, translation=(0.0, 0.0, 0.0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x, y, z = translation
    path.write_text(
        "Tr_velo_to_cam: 1 0 0 "
        f"{x} 0 1 0 {y} 0 0 1 {z}\n",
        encoding="utf-8",
    )


class ViewOfDelftDatasetTests(unittest.TestCase):
    def test_binary_loaders_follow_official_column_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lidar = np.arange(8, dtype=np.float32).reshape(2, 4)
            radar = np.arange(14, dtype=np.float32).reshape(2, 7)
            lidar.tofile(root / "lidar.bin")
            radar.tofile(root / "radar.bin")

            np.testing.assert_array_equal(load_vod_lidar(root / "lidar.bin"), lidar)
            np.testing.assert_array_equal(load_vod_radar(root / "radar.bin"), radar)

    def test_radar_alignment_uses_the_shared_camera_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lidar_calibration = root / "lidar.txt"
            radar_calibration = root / "radar.txt"
            _write_calibration(lidar_calibration, (1.0, 0.0, 0.0))
            _write_calibration(radar_calibration, (3.0, 4.0, 0.0))
            transform = load_vod_radar_to_lidar(
                lidar_calibration,
                radar_calibration,
            )
            radar = np.asarray([[1, 2, 3, 4, 5, 6, -1]], dtype=np.float32)
            aligned = align_radar_to_lidar(radar, transform)

        np.testing.assert_allclose(aligned[0, :3], [3, 6, 3])
        np.testing.assert_array_equal(aligned[0, 3:], radar[0, 3:])

    def test_accumulated_radar_discovery_falls_back_to_radar_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            public = Path(directory) / "view_of_delft_PUBLIC"
            (public / "lidar" / "ImageSets").mkdir(parents=True)
            (public / "lidar" / "ImageSets" / "train.txt").write_text(
                "00001\n",
                encoding="utf-8",
            )
            lidar = public / "lidar" / "training" / "velodyne" / "00001.bin"
            radar = (
                public
                / "radar_3frames"
                / "training"
                / "velodyne"
                / "00001.bin"
            )
            lidar.parent.mkdir(parents=True)
            radar.parent.mkdir(parents=True)
            lidar.touch()
            radar.touch()
            _write_calibration(
                public / "lidar" / "training" / "calib" / "00001.txt"
            )
            radar_calibration = (
                public / "radar" / "training" / "calib" / "00001.txt"
            )
            _write_calibration(radar_calibration)

            frame = discover_vod_frames(directory, "train")[0]

        self.assertEqual(frame.frame_id, "00001")
        self.assertEqual(frame.radar_variant, "radar_3frames")
        self.assertEqual(frame.radar_calibration_path, radar_calibration)

    def test_vod_radar_cache_is_indexed_by_split_and_frame(self):
        path = radar_cache_path(
            Path("cache"),
            {
                "dataset": "View-of-Delft",
                "split": "val",
                "frame_id": "00342",
            },
        )
        self.assertEqual(path, Path("cache") / "val" / "00342.npz")


if __name__ == "__main__":
    unittest.main()
