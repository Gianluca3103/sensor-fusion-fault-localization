import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from models.two_stage_reconstruction_head.range_view.merge import MergeConfig
from scripts.export_vod_pvrcnn import export_sve_reconstructed_validation


def _fixture(tmp_path: Path):
    public = tmp_path / "vod"
    image_sets = public / "lidar" / "ImageSets"
    image_sets.mkdir(parents=True)
    (image_sets / "val.txt").write_text("00001\n00002\n", encoding="utf-8")
    samples = tmp_path / "samples"
    (samples / "val").mkdir(parents=True)
    for frame in ("00001", "00002"):
        metadata = {"dataset": "View-of-Delft", "split": "val", "frame_id": frame,
                    "range_view_full_scan": True}
        np.savez(samples / "val" / f"{frame}_fog_sim_s1.npz",
                 metadata_json=json.dumps(metadata))
    args = argparse.Namespace(samples_root=samples, radar_root=tmp_path / "radar",
                              output_root=tmp_path / "out", checkpoint=tmp_path / "epoch80.pt",
                              device="cpu")
    loaded = (None, None, MergeConfig(forward_only=True), 80, 0.0, True)
    return args, public, loaded


def test_sve_export_writes_only_matched_validation_clouds(tmp_path: Path):
    args, public, loaded = _fixture(tmp_path)
    cloud = np.asarray([[4, 0, 0, 0.5], [-4, 0, 0, 0.5]], dtype=np.float32)
    with patch("scripts.export_vod_pvrcnn._reconstruct", return_value=cloud) as reconstruct:
        export_sve_reconstructed_validation(args, public, loaded)
        export_sve_reconstructed_validation(args, public, loaded)
    root = args.output_root / "lidar" / "reconstructed"
    manifest = json.loads((root / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["condition"] == "reconstructed"
    assert manifest["splits"] == {"train": 0, "val": 2}
    assert not (root / "training" / "calib").exists()
    assert not (root / "training" / "velodyne" / "00003.bin").exists()
    for frame in ("00001", "00002"):
        points = np.fromfile(root / "training" / "velodyne" / f"{frame}.bin",
                             dtype="<f4").reshape(-1, 4)
        np.testing.assert_array_equal(points, cloud[:1])
    assert reconstruct.call_count == 4


def test_sve_export_refuses_incomplete_validation_cache(tmp_path: Path):
    args, public, loaded = _fixture(tmp_path)
    (args.samples_root / "val" / "00002_fog_sim_s1.npz").unlink()
    with pytest.raises(ValueError, match="every official VoD validation frame"):
        export_sve_reconstructed_validation(args, public, loaded)
