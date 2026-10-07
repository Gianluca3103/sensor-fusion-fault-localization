"""CPU invariants for observed-ray conditioning and residual supervision."""

import numpy as np
import torch
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from models.radar_lidar_stage1.config import VoxelGrid
from models.radar_lidar_stage2.candidate_domain import CandidateDomain, voxel_centers_xyz
from models.radar_lidar_stage2_residual.config import ResidualStage2Config
from models.radar_lidar_stage2_residual.coverage import faulty_coverage
from models.radar_lidar_stage2_residual.data import (
    PairedFaultDataset, collate_paired, fault_region_from_metadata, within_fault_region,
)
from models.radar_lidar_stage2_residual.losses import residual_loss
from models.radar_lidar_stage2_residual.model import ResidualOutput
from models.radar_lidar_stage2_residual.targets import make_residual_targets
from models.radar_lidar_stage2_residual.train import ResidualMetrics


GRID = VoxelGrid((0., -4., -4.), (8., 4., 4.), (.5, .5, .5))
REGION = {"x_range": [0., 8.], "y_range": [-4., 4.], "min_range_m": 0., "max_range_m": 20.}


def _domain(*xyz_indices):
    # (x,y,z) indices; CandidateDomain requires lexicographically sorted BZYX.
    rows = sorted([[0, z, y, x] for x, y, z in xyz_indices])
    coords = torch.tensor(rows, dtype=torch.long)
    return CandidateDomain(coords, torch.ones(len(rows)), GRID, {})


def test_measured_voxel_is_negative_and_missing_clean_voxel_is_positive():
    domain = _domain((2, 8, 8), (4, 9, 8))
    centers = voxel_centers_xyz(domain.coordinates, GRID)
    observed = centers[0]
    missing = centers[1]
    faulty = torch.cat((observed, torch.tensor([.6])))[None, None]
    clean = torch.stack((torch.cat((observed, torch.tensor([.6]))),
                         torch.cat((missing, torch.tensor([.4])))))[None]
    coverage = faulty_coverage(domain, faulty, torch.ones((1, 1), dtype=torch.bool), [REGION])
    target = make_residual_targets(domain, clean, torch.ones((1, 2), dtype=torch.bool),
                                   coverage, free_ray_tolerance_m=.1)
    assert target.observed.tolist() == [True, False]
    assert target.addition.tolist() == [False, True]
    assert bool((target.addition & target.observed).any()) is False


def test_measured_ray_blocks_behind_and_outside_region_is_not_supervised():
    domain = _domain((2, 8, 8), (7, 9, 9), (14, 8, 8))
    center = domain.centers_xyz
    faulty = torch.cat((center[0], torch.tensor([.5])))[None, None]
    region = dict(REGION, x_range=[0., 4.])
    coverage = faulty_coverage(domain, faulty, torch.ones((1, 1), dtype=torch.bool), [region],
                               ray_tolerance_m=.2)
    assert coverage.measured_voxel[0]
    # The farther center is on a slightly different ray, so this also checks
    # that the mask is geometric rather than a broad axis-aligned strip.
    assert coverage.in_fault_region.tolist() == [True, False, True]
    assert not coverage.may_add[0]
    assert coverage.measured_ray[2]
    assert not coverage.may_add[1]
    assert not coverage.may_add[2]


def test_empty_faulty_scan_allows_clean_additions_without_clean_forward_input():
    domain = _domain((2, 8, 8))
    clean = torch.cat((domain.centers_xyz[0], torch.tensor([.2])))[None, None]
    faulty = torch.empty((1, 0, 4))
    coverage = faulty_coverage(domain, faulty, torch.zeros((1, 0), dtype=torch.bool), [REGION])
    target = make_residual_targets(domain, clean, torch.ones((1, 1), dtype=torch.bool),
                                   coverage, free_ray_tolerance_m=.1)
    assert target.addition.tolist() == [True]
    assert not bool(coverage.blocked.any())


def test_loss_has_observed_negative_gradient_and_ignores_unknown():
    domain = _domain((2, 8, 8), (4, 9, 8), (5, 12, 8))
    centers = domain.centers_xyz
    faulty = torch.cat((centers[0], torch.tensor([.5])))[None, None]
    clean = torch.stack((torch.cat((centers[0], torch.tensor([.5]))),
                         torch.cat((centers[1], torch.tensor([.5])))))[None]
    coverage = faulty_coverage(domain, faulty, torch.ones((1, 1), dtype=torch.bool), [REGION])
    target = make_residual_targets(domain, clean, torch.ones((1, 2), dtype=torch.bool),
                                   coverage, free_ray_tolerance_m=.1)
    logits = torch.zeros(3, requires_grad=True)
    offsets = torch.zeros((3, 3), requires_grad=True)
    output = ResidualOutput(domain, coverage, logits, torch.sigmoid(logits), offsets,
                            torch.empty((0, 3)), torch.empty(0), {})
    residual_loss(output, target, ResidualStage2Config())["total"].backward()
    assert logits.grad[0] > 0  # suppress a duplicate
    assert logits.grad[1] < 0  # reward a missing clean surface
    assert logits.grad[2] == 0  # unmeasured, unknown space
    assert torch.equal(offsets.grad[0], torch.zeros(3))


def test_fault_region_matches_cache_range_and_bev_filter():
    xyz = np.array([[1., 0., 0.], [9., 0., 0.], [1., 5., 0.]], np.float32)
    assert within_fault_region(xyz, REGION).tolist() == [True, False, False]


def test_full_scan_marker_replaces_the_missing_crop_metadata():
    region = fault_region_from_metadata({"range_view_full_scan": True}, "sample.npz")
    xyz = np.array([[1., 0., 0.], [9., 0., 0.], [-7., 4., 2.]], np.float32)
    assert region == {"full_scan": True}
    assert within_fault_region(xyz, region).tolist() == [True, True, True]
    try:
        fault_region_from_metadata({}, "ambiguous.npz")
    except ValueError as error:
        assert "Cannot determine" in str(error)
    else:
        raise AssertionError("Missing crop and full-scan marker must fail closed")


def test_global_missing_metric_counts_clean_points_beyond_candidate_domain():
    domain = _domain((2, 8, 8))
    clean = torch.tensor([[[1.25, .25, .25, .1], [2.25, 1.25, .25, .1]]])
    faulty = torch.tensor([[[1.25, .25, .25, .1]]])
    coverage = faulty_coverage(domain, faulty, torch.ones((1, 1), dtype=torch.bool), [REGION])
    target = make_residual_targets(domain, clean, torch.ones((1, 2), dtype=torch.bool),
                                   coverage, free_ray_tolerance_m=.1)
    logits = torch.tensor([-10.])
    output = ResidualOutput(domain, coverage, logits, torch.sigmoid(logits), torch.zeros((1, 3)),
                            torch.empty((0, 3)), torch.empty(0), {})
    metrics = ResidualMetrics()
    metrics.add(output, target, .5, clean, torch.ones((1, 2), dtype=torch.bool),
                faulty, torch.ones((1, 1), dtype=torch.bool))
    summary = metrics.summary()
    assert summary["missing_clean_points"] == 1
    assert summary["point_target"] == 0  # the one missing point had no radar candidate
    assert summary["global_missing_recall_0.2m"] == 0


def test_fault_cache_pairing_and_clean_crop():
    class FakeBase:
        def __init__(self, *args, **kwargs):
            self.frames = [SimpleNamespace(frame_id="00001")]

        def __getitem__(self, index):
            return {"frame_id": "00001", "split": "train",
                    "radar": torch.zeros((1, 7)),
                    "clean_lidar": torch.tensor([[1., 0., 0., .1], [9., 0., 0., .2]])}

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "train").mkdir()
        np.savez(root / "train" / "00001_fov_filter_s1.npz",
                 faulty_lidar_points=np.array([[1., 0., 0., .1]], np.float32),
                 metadata_json=np.asarray(json.dumps({"frame_id": "00001", "split": "train",
                                                      "point_filter": REGION})))
        with patch("models.radar_lidar_stage2_residual.data.VoDStage1Dataset", FakeBase):
            dataset = PairedFaultDataset(root, root, "train", radar_variant="test")
            sample = dataset[0]
            batch = collate_paired([sample])
        assert batch["clean_lidar_valid"].sum() == 1
        assert batch["faulty_lidar_valid"].sum() == 1
        assert batch["fault_region"] == [REGION]


def test_full_scan_cache_keeps_clean_targets_uncropped():
    class FakeBase:
        def __init__(self, *args, **kwargs):
            self.frames = [SimpleNamespace(frame_id="00001")]

        def __getitem__(self, index):
            return {"frame_id": "00001", "split": "train",
                    "radar": torch.zeros((1, 7)),
                    "clean_lidar": torch.tensor([[1., 0., 0., .1], [9., 0., 0., .2]])}

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "train").mkdir()
        np.savez(root / "train" / "00001_fov_filter_s1.npz",
                 faulty_lidar_points=np.array([[1., 0., 0., .1]], np.float32),
                 metadata_json=np.asarray(json.dumps({"frame_id": "00001", "split": "train",
                                                      "range_view_full_scan": True})))
        with patch("models.radar_lidar_stage2_residual.data.VoDStage1Dataset", FakeBase):
            sample = PairedFaultDataset(root, root, "train", radar_variant="test")[0]
        assert len(sample["clean_lidar"]) == 2
        assert sample["fault_region"] == {"full_scan": True}
