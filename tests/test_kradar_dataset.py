import tempfile
from pathlib import Path
import unittest

import numpy as np

from Fault_Localization_Model.kradar_dataset import (
    kradar_source_metadata,
    load_radar_from_lidar_transform,
    radar_bev_support_mask,
    radar_overlap_mask,
    read_kradar_lidar_pcd,
    select_temporal_split_frames,
)


def write_ascii_pcd(path, points):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS x y z intensity t reflectivity ring ambient range",
        "SIZE 4 4 4 4 4 2 1 2 4",
        "TYPE F F F F U U U U U",
        "COUNT 1 1 1 1 1 1 1 1 1",
        f"WIDTH {len(points)}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {len(points)}",
        "DATA ascii",
    ]
    rows.extend(
        f"{x} {y} {z} {intensity} 0 0 0 0 0"
        for x, y, z, intensity in points
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


class KRadarDatasetTests(unittest.TestCase):
    def test_pcd_reader_preserves_xyzi(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.pcd"
            expected = np.asarray(
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                dtype=np.float32,
            )
            write_ascii_pcd(path, expected)
            actual = read_kradar_lidar_pcd(path)
        self.assertEqual(actual.dtype, np.float32)
        self.assertTrue(np.array_equal(actual, expected))

    def test_radar_overlap_uses_radar_origin_and_azimuth(self):
        transform = np.eye(4)
        points = np.asarray(
            [
                [10.0, 0.0, 0.0, 1.0],
                [1.0, 10.0, 0.0, 1.0],
                [-10.0, 0.0, 0.0, 1.0],
                [120.0, 0.0, 0.0, 1.0],
                [10.0, 0.0, 10.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.assertEqual(
            radar_overlap_mask(points, transform).tolist(),
            [True, False, False, False, False],
        )

    def test_radar_bev_support_uses_calibrated_cell_centers(self):
        support = radar_bev_support_mask(
            (2, 4),
            (0.0, 2.0),
            (-2.0, 2.0),
            np.eye(4),
            azimuth_range_rad=(-np.pi / 6.0, np.pi / 6.0),
            radar_range_m=(0.0, 10.0),
        )

        self.assertTrue(
            np.array_equal(
                support,
                np.asarray(
                    [
                        [False, True, True, False],
                        [False, False, False, False],
                    ]
                ),
            )
        )

    def test_pairing_metadata_and_temporal_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = root / "lidar" / "1"
            lidar_dir = sequence / "os2-64"
            label_dir = sequence / "info_label"
            calibration_dir = sequence / "info_calib"
            radar_dir = root / "radar" / "pc10p" / "1"
            for path in (lidar_dir, label_dir, calibration_dir, radar_dir):
                path.mkdir(parents=True, exist_ok=True)
            (calibration_dir / "calib_radar_lidar.txt").write_text(
                "frame difference,x,y\n32,-2.54,0.3\n", encoding="utf-8"
            )
            frames = []
            for index in range(1, 11):
                lidar_index = f"{index:05d}"
                radar_index = f"{index + 32:05d}"
                lidar_path = lidar_dir / f"os2-64_{lidar_index}.pcd"
                lidar_path.touch()
                frames.append(lidar_path)
                (radar_dir / f"rpc_{radar_index}.npy").touch()
                (label_dir / f"{radar_index}_{lidar_index}.txt").write_text(
                    "* idx(tesseract_os2-64_cam-front_os1-128_cam-lrr)="
                    f"{radar_index}_{lidar_index}_00001_00001_00001, "
                    f"timestamp=1643292946.{index:09d}\n",
                    encoding="utf-8",
                )

            metadata = kradar_source_metadata(frames[0], root)
            train, _ = select_temporal_split_frames(frames, root, "train")
            validation, _ = select_temporal_split_frames(frames, root, "val")
            test, _ = select_temporal_split_frames(frames, root, "test")
            radar_from_lidar = load_radar_from_lidar_transform(
                calibration_dir / "calib_radar_lidar.txt"
            )

        self.assertEqual(metadata["sequence"], "1")
        self.assertEqual(metadata["lidar_index"], "00001")
        self.assertEqual(metadata["radar_index"], "00033")
        self.assertEqual((len(train), len(validation), len(test)), (7, 1, 2))
        self.assertTrue(
            np.allclose(radar_from_lidar[:3, 3], [-2.54, 0.3, 0.7])
        )


if __name__ == "__main__":
    unittest.main()
