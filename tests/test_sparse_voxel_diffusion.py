import unittest
from unittest.mock import patch

import numpy as np
import torch

from models.two_stage_reconstruction_head.diffusion_process import (
    SparseVoxelDiffusionBaseline,
    SparseVoxelDiffusionConfig,
    SoftVoxelChamferLoss,
    build_sparse_voxel_example,
    collate_sparse_voxel_examples,
    voxel_set_metrics_at_distance,
)
from models.two_stage_reconstruction_head.diffusion_process.sparse_voxel_data import SparseVoxelBatch
from voxelization import (
    HardVoxelizer,
    OracleFaultSelector3DConfig,
    VoxelGridConfig,
    build_voxel_fault_targets,
    select_oracle_fault_regions_3d,
)


class SparseVoxelDiffusionTests(unittest.TestCase):
    def setUp(self):
        self.grid = VoxelGridConfig(
            x_range=(0.0, 4.0), y_range=(0.0, 4.0), z_range=(0.0, 4.0),
            voxel_size=(1.0, 1.0, 1.0),
        )
        # The first clean point is missing; the second faulty point is synthetic.
        self.clean = np.asarray([[0.2, 0.2, 0.2, 0.1]], dtype=np.float32)
        self.faulty = np.asarray([[2.2, 2.2, 2.2, 0.3]], dtype=np.float32)
        self.targets = build_voxel_fault_targets(
            self.clean, self.faulty, np.asarray([-1]), self.grid
        )
        self.selection = select_oracle_fault_regions_3d(
            self.targets.repair_mask,
            self.targets.remove_mask,
            self.grid,
            OracleFaultSelector3DConfig(halo_m=1.0, grouping_radius_m=4.0),
        )
        names = ("x", "y", "z", "intensity")
        voxelizer = HardVoxelizer(self.grid)
        self.faulty_voxels = voxelizer.voxelize(self.faulty, names)
        self.radar_voxels = voxelizer.voxelize(
            np.asarray([[0.2, 0.2, 0.2, 0.5]], dtype=np.float32), names
        )

    def _batch(self):
        example = build_sparse_voxel_example(
            faulty_lidar=self.faulty_voxels,
            radar=self.radar_voxels,
            targets=self.targets,
            selection=self.selection,
            component=self.selection.components[0],
            grid=self.grid,
        )
        return collate_sparse_voxel_examples([example])

    def test_candidate_lattice_keeps_missing_repair_voxel(self):
        batch = self._batch()
        coordinates = batch.coords_zyx[0, batch.valid_mask[0, :, 0].bool()]
        self.assertTrue(torch.any(torch.all(coordinates == torch.tensor([0, 0, 0]), dim=1)))
        self.assertGreater(int(batch.editable_mask.sum()), 0)

    def test_condition_does_not_reveal_exact_repair_or_remove_voxels(self):
        batch = self._batch()
        valid = batch.valid_mask.bool()
        self.assertEqual(batch.condition_features.shape[-1], 2)
        self.assertTrue(torch.equal(batch.editable_mask[valid], torch.ones_like(batch.editable_mask[valid])))
        self.assertGreater(int(((batch.target_occupancy < 0.5) & valid).sum()), 0)
        self.assertGreater(int(((batch.target_occupancy > 0.5) & valid).sum()), 0)

    def test_loss_matches_requested_baseline_and_backpropagates(self):
        batch = self._batch()
        model = SparseVoxelDiffusionBaseline(
            SparseVoxelDiffusionConfig(
                grid_dimensions_zyx=self.grid.dimensions_zyx,
                hidden_dim=16,
                num_blocks=2,
                training_timesteps=16,
                lambda_chamfer=0.7,
                chamfer_chunk_size=16,
            )
        )
        output = model(batch, torch.tensor([3]))
        expected = (
            output["diffusion_loss"] + output["bce_loss"]
            + 0.7 * output["chamfer_loss"]
        )
        self.assertTrue(torch.allclose(output["loss"], expected))
        output["loss"].backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_zero_chamfer_weight_skips_pairwise_loss(self):
        batch = self._batch()
        model = SparseVoxelDiffusionBaseline(
            SparseVoxelDiffusionConfig(
                grid_dimensions_zyx=self.grid.dimensions_zyx,
                hidden_dim=16,
                num_blocks=1,
                training_timesteps=16,
                lambda_chamfer=0.0,
            )
        )
        with patch.object(model.chamfer_loss, "forward", side_effect=AssertionError("Chamfer ran")):
            output = model(batch, torch.tensor([3]))
        self.assertEqual(float(output["chamfer_loss"]), 0.0)
        self.assertTrue(torch.allclose(output["loss"], output["diffusion_loss"] + output["bce_loss"]))

    def test_sampling_keeps_non_editable_voxels_equal_to_faulty_input(self):
        batch = self._batch()
        model = SparseVoxelDiffusionBaseline(
            SparseVoxelDiffusionConfig(
                grid_dimensions_zyx=self.grid.dimensions_zyx,
                hidden_dim=16,
                num_blocks=1,
                training_timesteps=8,
                chamfer_chunk_size=16,
            )
        )
        output = model.sample(batch, sampling_steps=3)
        fixed = (batch.valid_mask > 0.5) & (batch.editable_mask < 0.5)
        self.assertTrue(torch.equal(
            output["occupancy_probability"][fixed], batch.faulty_occupancy[fixed]
        ))

    def test_capped_chamfer_backpropagates_through_sampled_candidates(self):
        torch.manual_seed(7)
        coords = torch.arange(100, dtype=torch.float32)[:, None].repeat(1, 3)
        probabilities = torch.full((100,), 0.5, requires_grad=True)
        target = torch.zeros(100)
        target[:50] = 1.0
        loss = SoftVoxelChamferLoss(chunk_size=8, max_points=16)._one(
            coords, probabilities, target
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(int((probabilities.grad != 0).sum()), 0)
        self.assertLessEqual(int((probabilities.grad != 0).sum()), 16)

    def test_large_voxel_metric_uses_exact_nearest_neighbours(self):
        count = 1200
        x = torch.arange(count, dtype=torch.float32) * 0.2
        xyz = torch.stack((x, torch.zeros_like(x), torch.zeros_like(x)), dim=-1)[None]
        occupied = torch.ones((1, count, 1))
        batch = SparseVoxelBatch(
            coords_zyx=torch.zeros((1, count, 3), dtype=torch.long),
            coords_xyz_m=xyz,
            condition_features=torch.zeros((1, count, 2)),
            target_occupancy=occupied,
            faulty_occupancy=occupied,
            editable_mask=occupied,
            valid_mask=occupied,
        )
        metrics = voxel_set_metrics_at_distance(occupied, batch, distance_m=0.2)
        self.assertAlmostEqual(float(metrics["f1_at_0_2m"]), 1.0)
        self.assertAlmostEqual(float(metrics["iou_at_0_2m"]), 1.0)

    def test_voxel_metric_excludes_preserved_context(self):
        occupied = torch.tensor([[[1.0], [0.0]]])
        target = torch.ones((1, 2, 1))
        batch = SparseVoxelBatch(
            coords_zyx=torch.tensor([[[0, 0, 0], [0, 0, 1]]]),
            coords_xyz_m=torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
            condition_features=torch.zeros((1, 2, 2)),
            target_occupancy=target,
            faulty_occupancy=occupied,
            editable_mask=torch.tensor([[[0.0], [1.0]]]),
            valid_mask=torch.ones((1, 2, 1)),
        )
        metrics = voxel_set_metrics_at_distance(occupied, batch, distance_m=0.2)
        self.assertEqual(float(metrics["f1_at_0_2m"]), 0.0)
        self.assertEqual(float(metrics["iou_at_0_2m"]), 0.0)


if __name__ == "__main__":
    unittest.main()
