from pathlib import Path

import numpy as np
from PIL import Image

from scripts.export_vod_pvrcnn import (
    _image_placeholder, detector_points, normalized_labels,
)


def test_labels_map_bicycle_and_place_dontcare_last(tmp_path: Path) -> None:
    label = tmp_path / "frame.txt"
    label.write_text(
        "truck 0 0 0 1 2 3 4 1 2 3 4 5 6 7 99\n"
        "bicycle 0 0 0 1 2 3 4 1 2 3 4 5 6 7 100\n",
        encoding="utf-8",
    )
    rows = [line.split() for line in normalized_labels(label).splitlines()]
    assert [row[0] for row in rows] == ["Cyclist", "DontCare"]
    assert all(len(row) == 15 for row in rows)


def test_detector_points_early_fusion_and_forward_crop() -> None:
    lidar = np.array([[1, 2, 3, 0.5], [-1, 2, 3, 0.8]], dtype=np.float32)
    radar = np.array([[2, 3, 4, 10, -2], [-2, 3, 4, 11, -1]], dtype=np.float32)
    result = detector_points(lidar, radar, forward_only=True)
    np.testing.assert_array_equal(result, [[1, 2, 3, 0.5], [2, 3, 4, 10]])


def test_empty_fault_gets_detector_only_sentinel() -> None:
    result = detector_points(np.empty((0, 4), dtype=np.float32), None,
                             forward_only=True)
    np.testing.assert_allclose(result, [[0.01, 0.0, -2.9, 0.0]])


def test_placeholder_keeps_camera_dimensions(tmp_path: Path) -> None:
    source = tmp_path / "vod" / "image_2"
    source.mkdir(parents=True)
    Image.new("RGB", (24, 12), color="white").save(source / "00001.jpg")
    target = tmp_path / "export" / "image_2" / "00001.png"
    _image_placeholder(source.parent, target, "00001")
    with Image.open(target) as image:
        assert image.size == (24, 12)
