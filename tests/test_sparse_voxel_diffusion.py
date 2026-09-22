import unittest

import numpy as np
import torch

from models.two_stage_reconstruction_head.diffusion_process import (
    SparseVoxelDiffusionBaseline,
    SparseVoxelDiffusionConfig,
    build_sparse_voxel_example,
    collate_sparse_voxel_examples,
)
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


if __name__ == "__main__":
    unittest.main()
