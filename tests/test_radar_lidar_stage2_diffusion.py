"""Diffusion supervision and reverse-process invariants without MinkowskiEngine."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from models.radar_lidar_stage1.config import VoxelGrid
from models.radar_lidar_stage2_diffusion.config import DiffusionConfig
from models.radar_lidar_stage2_diffusion.losses import clean_state, denoising_loss, training_state
from models.radar_lidar_stage2_diffusion.model import ResidualSparseDiffusion
from models.radar_lidar_stage2_diffusion.schedule import CosineSchedule
from models.radar_lidar_stage2_diffusion.train import _denoise
from models.radar_lidar_stage2_residual.config import ResidualStage2Config
from models.radar_lidar_stage2_residual.targets import ResidualTargets
from models.radar_lidar_stage2_residual.targets import make_residual_targets
from scripts.smoke_stage2_minkowski import _stage1


def _target():
    return ResidualTargets(
        addition=torch.tensor([True, False, False, False]),
        observed=torch.tensor([False, True, False, False]),
        known_free=torch.tensor([False, False, True, False]),
        offsets_normalized=torch.tensor([[.2, -.1, .3], [0., 0., 0.],
                                         [0., 0., 0.], [0., 0., 0.]]),
        clean=None,
    )


def test_unknown_has_no_loss_and_positive_offset_has_gradient():
    target = _target()
    clean = clean_state(target)
    assert clean[:, 0].tolist() == [1., -1., -1., 0.]
    predicted = torch.zeros((4, 4), requires_grad=True)
    denoising_loss(predicted, clean, target, DiffusionConfig())["total"].backward()
    assert predicted.grad[0, 0] < 0
    assert predicted.grad[1, 0] > 0
    assert predicted.grad[2, 0] > 0
    assert torch.equal(predicted.grad[3], torch.zeros(4))
    assert predicted.grad[0, 1:].abs().sum() > 0
    assert torch.equal(predicted.grad[1, 1:], torch.zeros(3))


def test_unknown_training_latent_is_random_and_loss_masked():
    target = _target()
    state = training_state(target, generator=torch.Generator().manual_seed(5))
    assert state[3, 0].abs() == 1
    assert torch.all(state[3, 1:].abs() <= 1)
    assert torch.equal(state[:3], clean_state(target)[:3])
    prediction = torch.zeros_like(state, requires_grad=True)
    denoising_loss(prediction, state, target, DiffusionConfig())["total"].backward()
    assert torch.equal(prediction.grad[3], torch.zeros(4))


def test_schedule_reaches_noise_and_reverse_preserves_fixed_domain():
    schedule = CosineSchedule(16)
    assert schedule.alpha_bar[0] > schedule.alpha_bar[-1]
    assert schedule.alpha_bar[-1] < .01
    clean = torch.tensor([[1., .2, -.4, .6], [-1., 0., 0., 0.]])
    noise = torch.tensor([[.3, -.7, 1., .2], [-.3, .4, .5, -.8]])
    state = schedule.mix(clean, noise, torch.full((2,), 15, dtype=torch.long))
    for t in range(15, -1, -1):
        state = schedule.reverse_step(state, clean, t)
    assert torch.allclose(state, clean)


def test_diffusion_config_round_trip_and_rejects_deterministic_switch(tmp_path):
    config = DiffusionConfig(timesteps=8)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.as_dict()), encoding="utf-8")
    assert DiffusionConfig.from_json(path) == config
    assert config.diffusion_enabled is True
    try:
        DiffusionConfig(diffusion_enabled=False)
    except ValueError:
        pass
    else:
        raise AssertionError("Diffusion run unexpectedly accepted diffusion off")
    try:
        DiffusionConfig(residual=ResidualStage2Config(diffusion_enabled=True))
    except NotImplementedError:
        pass  # The old deterministic config must reject a misleading switch.
    else:
        raise AssertionError("Deterministic Stage II unexpectedly accepted diffusion")


def test_sparse_diffusion_data_flow_with_coordinate_preserving_engine_stub():
    # The real MinkowskiEngine forward/backward is checked on the professor GPU.
    # This check verifies the entire input/target/sampling boundary on the CPU.
    class Sparse:
        coordinate_builds = 0
        coordinate_reuses = 0

        def __init__(self, *, features, coordinates=None, coordinate_map_key=None,
                     coordinate_manager=None):
            self.F = features
            if coordinates is not None:
                Sparse.coordinate_builds += 1
                self.coordinate_map_key = "input"
                self.coordinate_manager = {"input": coordinates}
            else:
                Sparse.coordinate_reuses += 1
                self.coordinate_map_key = coordinate_map_key
                self.coordinate_manager = coordinate_manager
            self.C = self.coordinate_manager[self.coordinate_map_key]

    class IdentityUNet(nn.Module):
        def __init__(self, channels):
            super().__init__()

        def forward(self, sparse):
            count = len(sparse.F)
            return sparse, {"input": count, "output": count}

    grid = VoxelGrid((0., 0., 0.), (8., 8., 8.), (.5, .5, .5))
    base = ResidualStage2Config(channels=(8, 12, 16, 24), conditioning_dim=8,
                                query_radii_m=(.75, 1.5, 3., 6.), max_neighbors=4,
                                max_candidate_sites=3000)
    config = DiffusionConfig(residual=base, timesteps=4)
    evidence = _stage1(grid, "cpu")
    clean = torch.tensor([[[2.25, 2.25, 2.25, .1], [2.75, 2.25, 2.75, .2]]])
    faulty = clean[:, :1].clone()
    region = {"full_scan": True}
    with (patch("models.radar_lidar_stage2_diffusion.model._require_me"),
          patch("models.radar_lidar_stage2_diffusion.model.ME", SimpleNamespace(SparseTensor=Sparse)),
          patch("models.radar_lidar_stage2_diffusion.model.SparseUNet", IdentityUNet)):
        model = ResidualSparseDiffusion((4, 4, 4, 4), config)
        context = model.prepare(evidence, grid, faulty, torch.ones((1, 1), dtype=torch.bool),
                                [region])
        target = make_residual_targets(context.domain, clean, torch.ones((1, 2), dtype=torch.bool),
                                       context.coverage, free_ray_tolerance_m=.15)
        assert target.addition.any() and target.observed.any()
        losses, counts = _denoise(model, context, target, 1)
        assert counts["input"] == counts["output"] == len(context.domain.coordinates)
        losses["total"].backward()
        assert model.clean_head.weight.grad is not None
        model.eval()
        with torch.no_grad():
            Sparse.coordinate_builds = Sparse.coordinate_reuses = 0
            first = model.sample(context, frame_seeds=[1701])
            second = model.sample(context, frame_seeds=[1701])
        assert Sparse.coordinate_builds == 2
        assert Sparse.coordinate_reuses == 2 * (config.timesteps - 1)
        assert torch.equal(first.addition_probability, second.addition_probability)
        assert not bool((first.addition_probability[~context.coverage.may_add] > .5).any())
        assert first.metadata["diffusion_enabled"] is True
