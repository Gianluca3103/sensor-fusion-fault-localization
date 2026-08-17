import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from Fault_Localization_Model.vod_dataset.radar_accumulation import (
    accumulate_vod_radar_scans,
    radar_current_from_source,
)


def _write_pose(path: Path, transform: np.ndarray) -> None:
    path.write_text(json.dumps({"odomToCamera": transform.reshape(-1).tolist()}))


def _write_calibration(path: Path, transform: np.ndarray) -> None:
    values = " ".join(str(value) for value in transform[:3].reshape(-1))
    path.write_text(f"Tr_velo_to_cam: {values}\n")


class VoDRadarAccumulationTests(unittest.TestCase):
    def test_source_scan_is_motion_compensated_into_current_radar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_pose = root / "source_pose.json"
            current_pose = root / "current_pose.json"
            source_calibration = root / "source_calib.txt"
            current_calibration = root / "current_calib.txt"
            source = np.eye(4)
            source[0, 3] = 2.0
            current = np.eye(4)
            calibration = np.eye(4)
            _write_pose(source_pose, source)
            _write_pose(current_pose, current)
            _write_calibration(source_calibration, calibration)
            _write_calibration(current_calibration, calibration)

            transform = radar_current_from_source(
                source_pose,
                current_pose,
                source_calibration,
                current_calibration,
            )
            np.testing.assert_allclose(transform[:3, 3], [2.0, 0.0, 0.0])

    def test_accumulation_preserves_features_and_sets_time_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            radar_paths = []
            pose_paths = []
            calibration_paths = []
            for index in range(2):
                radar_path = root / f"radar_{index}.bin"
                pose_path = root / f"pose_{index}.json"
                calibration_path = root / f"calib_{index}.txt"
                np.asarray(
                    [[1.0, 2.0, 3.0, 4.0 + index, 5.0, 6.0, 0.0]],
                    dtype=np.float32,
                ).tofile(radar_path)
                _write_pose(pose_path, np.eye(4))
                _write_calibration(calibration_path, np.eye(4))
                radar_paths.append(radar_path)
                pose_paths.append(pose_path)
                calibration_paths.append(calibration_path)

            output = accumulate_vod_radar_scans(
                radar_paths, pose_paths, calibration_paths
            )
            np.testing.assert_allclose(output[:, 3:6], [[4, 5, 6], [5, 5, 6]])
            np.testing.assert_allclose(output[:, 6], [-1, 0])


if __name__ == "__main__":
    unittest.main()
