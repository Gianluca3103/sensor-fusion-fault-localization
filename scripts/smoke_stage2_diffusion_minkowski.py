"""Synthetic Stage-II diffusion forward, backward, sampling and reload check."""

from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import tempfile

import torch

from models.radar_lidar_stage1.config import VoxelGrid
from models.radar_lidar_stage2_diffusion.config import DiffusionConfig
from models.radar_lidar_stage2_diffusion.losses import training_state, denoising_loss
from models.radar_lidar_stage2_diffusion.model import ResidualSparseDiffusion
from models.radar_lidar_stage2_residual.config import ResidualStage2Config
from models.radar_lidar_stage2_residual.targets import make_residual_targets
from scripts.smoke_stage2_minkowski import _stage1


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(19)
    grid = VoxelGrid((0., 0., 0.), (8., 8., 8.), (.5, .5, .5))
    base = ResidualStage2Config(channels=(8, 12, 16, 24), conditioning_dim=8,
                                query_radii_m=(.75, 1.5, 3., 6.), max_neighbors=4,
                                expansion_zyx=(1, 1, 1), max_candidate_sites=3000)
    config = DiffusionConfig(residual=base, timesteps=4)
    stage1 = _stage1(grid, device)
    model = ResidualSparseDiffusion((4, 4, 4, 4), config).to(device).train()
    clean = torch.tensor([[[2.25, 2.25, 2.25, .1], [2.75, 2.25, 2.75, .2]]], device=device)
    faulty = clean[:, :1].clone()
    faulty_valid = torch.ones((1, 1), dtype=torch.bool, device=device)
    region = {"x_range": [0., 8.], "y_range": [0., 8.],
              "min_range_m": 0., "max_range_m": 20.}
    context = model.prepare(stage1, grid, faulty, faulty_valid, [region])
    target = make_residual_targets(context.domain, clean,
                                   torch.ones((1, 2), dtype=torch.bool, device=device),
                                   context.coverage, free_ray_tolerance_m=.15)
    assert target.addition.sum() >= 1 and target.observed.sum() >= 1
    truth = training_state(target)
    t = torch.tensor([2], dtype=torch.long, device=device)
    noisy = model.schedule.mix(truth, torch.randn_like(truth), t[context.domain.coordinates[:, 0]])
    predicted, counts = model.predict_clean(context, noisy, t)
    assert counts["input"] == counts["output"] == len(context.domain.coordinates)
    loss = denoising_loss(predicted, truth, target, config)["total"]
    assert bool(torch.isfinite(loss))
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)
    assert all(sites.features.grad is None for sites in stage1.features.values())
    if device == "cuda":
        model.zero_grad(set_to_none=True)
        # Conditioning contains trainable projections. Rebuild its graph before
        # the second backward pass, as training does for each new batch.
        mixed_context = model.prepare(stage1, grid, faulty, faulty_valid, [region])
        assert torch.equal(context.domain.coordinates, mixed_context.domain.coordinates)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            mixed, _ = model.predict_clean(mixed_context, noisy, t)
        mixed_loss = denoising_loss(mixed.float(), truth, target, config)["total"]
        assert bool(torch.isfinite(mixed_loss))
        mixed_loss.backward()
        assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
    model.eval()
    with torch.no_grad():
        generated = model.sample(context, generator=torch.Generator(device=device).manual_seed(41))
        generated_three = model.sample(context,
                                       generator=torch.Generator(device=device).manual_seed(41),
                                       sampling_steps=3)
        assert generated_three.metadata["sampling_steps"] == 3
        assert bool(torch.isfinite(generated_three.addition_probability).all())
        with (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
              if device == "cuda" else nullcontext()):
            mixed_sample = model.sample(context, generator=torch.Generator(device=device).manual_seed(41))
            mixed_three = model.sample(context,
                                       generator=torch.Generator(device=device).manual_seed(41),
                                       sampling_steps=3)
        assert bool(torch.isfinite(mixed_sample.addition_probability).all())
        assert bool(torch.isfinite(mixed_three.addition_probability).all())
        assert not bool((generated.addition_probability[~context.coverage.may_add] > .5).any())
        assert bool(torch.isfinite(generated.reconstructed_points_xyz).all())
        with tempfile.TemporaryDirectory() as temporary:
            file = Path(temporary) / "diffusion.pt"
            torch.save(model.state_dict(), file)
            restored = ResidualSparseDiffusion((4, 4, 4, 4), config).to(device).eval()
            restored.load_state_dict(torch.load(file, map_location=device, weights_only=True))
            new_context = restored.prepare(stage1, grid, faulty,
                                           torch.ones((1, 1), dtype=torch.bool, device=device), [region])
            reproduced = restored.sample(new_context,
                                         generator=torch.Generator(device=device).manual_seed(41))
            assert torch.allclose(generated.addition_probability,
                                  reproduced.addition_probability, rtol=1e-5, atol=1e-6)
    print(json.dumps({"device": device, "diffusion_enabled": True,
                      "candidate_sites": len(context.domain.coordinates),
                      "positive_sites": int(target.addition.sum()),
                      "observed_sites": int(target.observed.sum()),
                      "sampling_steps": config.timesteps, "finite_gradients": True,
                      "bf16_finite": device == "cuda",
                      "checkpoint_reproduces_output": True}, indent=2))


if __name__ == "__main__":
    main()
