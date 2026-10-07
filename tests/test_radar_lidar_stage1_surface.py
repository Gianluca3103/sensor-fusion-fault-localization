"""Focused tests for radar-conditioned surface proposals and uncapped views."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np
import torch

from models.radar_lidar_stage1.config import Stage1Config, VoxelGrid
from models.radar_lidar_stage1.model import RadarLidarStage1, Stage1Output, SurfaceProposals
from models.radar_lidar_stage1.sparse import SparseSites
from models.radar_lidar_stage1.train import config_from_dict, export_radar_only, load_radar_only
from models.radar_lidar_stage2.candidate_domain import make_candidates
from scripts.rerender_stage1_clouds import read_ply
from scripts.visualize_stage1_confidence_cloud import save_viewer, write_ply


GRID = VoxelGrid((0.,0.,0.), (8.,8.,8.), (.5,.5,.5))


def configuration(**changes):
    base = Stage1Config(grid=GRID, channels=(8,16,32,64),
                        attention_radii_m=(.75,1.5,3.,6.),
                        positive_radii_m=(.4,.8,1.6,3.2),
                        attention_dim=8, max_neighbors=6, growth_scales=(4,),
                        surface_proposals_per_site=4,
                        surface_geometry_weight=.5, surface_confidence_weight=.2)
    return replace(base, **changes)


class SurfaceStage1Tests(unittest.TestCase):
    def test_surface_output_is_radar_only_and_receives_geometry_gradient(self):
        torch.manual_seed(7)
        model = RadarLidarStage1(configuration())
        radar = torch.tensor([[[1.,1.,1.,5.,0.,0.,0.], [1.5,1.5,1.,4.,0.,0.,1.]]])
        clean = torch.tensor([[[1.2,1.,1.,.5], [1.4,1.3,1.,.5]]])
        valid = torch.ones((1,2), dtype=torch.bool)
        before = model.forward_radar(radar, valid).surface
        loss, _ = model.forward_train(radar, valid, clean, valid)
        after = model.forward_radar(radar, valid).surface
        self.assertEqual(before.xyz.shape, (2,4,3))
        self.assertTrue(torch.equal(before.xyz, after.xyz))
        self.assertTrue(torch.equal(before.score, after.score))
        self.assertTrue(torch.isfinite(loss["loss/surface_geom"]))
        loss["loss/total"].backward()
        gradient = model.radar_only.surface_head.head[-1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_stage2_seeds_predicted_lidar_positions_not_radar_voxels(self):
        radar_cell = torch.tensor([[0,2,2,2]], dtype=torch.long)
        fine = SparseSites(radar_cell, torch.ones((1,8)), GRID.shape_zyx)
        confidence = fine.replace_features(torch.tensor([[.99]]))
        surface = SurfaceProposals(torch.tensor([[[4.1,4.1,4.1], [5.,5.,5.]]]),
                                   torch.tensor([[.9,.1]]))
        evidence = Stage1Output({"s1":fine}, confidence, {"s1":radar_cell}, {}, surface)
        domain = make_candidates(evidence, GRID, confidence_threshold=.25,
                                 expansion_zyx=(0,0,0))
        self.assertEqual(domain.counts["seed_source"], "predicted_lidar_surface")
        self.assertEqual(domain.coordinates.tolist(), [[0,8,8,8]])
        self.assertEqual(len(domain.confidence), 1)
        self.assertAlmostEqual(float(domain.confidence[0]), .9, places=5)

    def test_legacy_checkpoint_retains_old_representation(self):
        legacy = configuration(surface_proposals_per_site=0,
                               surface_geometry_weight=0.,surface_confidence_weight=0.)
        data = legacy.as_dict()
        for key in list(data):
            if key.startswith("surface_"):
                data.pop(key)
        self.assertEqual(config_from_dict(data).surface_proposals_per_site, 0)
        model = RadarLidarStage1(legacy)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/"radar.pth"
            export_radar_only(checkpoint, model)
            restored = load_radar_only(checkpoint)
            self.assertIsNone(restored.surface_head)

    def test_clean_lidar_view_is_uncapped_and_raw_radar_overlay_is_off(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"view.html"
            xyz = np.array([[float(index),0.,0.] for index in range(6)],np.float32)
            counts = save_viewer(path,frame_id="00000",epoch=1,radar=xyz,lidar=xyz,
                                 sites=xyz,confidence=np.ones(6,np.float32),
                                 limits=(-1,10,-1,1,-1,1),max_points=2,
                                 trained=True,calibrated=False)
            self.assertEqual(counts["Radar"]["shown"], 2)
            self.assertEqual(counts["Clean LiDAR"]["shown"], 6)
            html = path.read_text(encoding="utf-8")
            self.assertIn('id="overlay-radar" type="checkbox">',html)
            payload = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>',html).group(1))
            self.assertEqual(len(payload["clouds"][1]["xyz"]), 6)
            ply = Path(directory)/"clean.ply"
            write_ply(ply,xyz)
            loaded,_ = read_ply(ply)
            np.testing.assert_array_equal(loaded,xyz)


if __name__ == "__main__":
    unittest.main()
