"""Synthetic CUDA smoke test for deterministic Stage-II forward/backward."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import torch

from models.radar_lidar_stage1.config import VoxelGrid
from models.radar_lidar_stage1.model import Stage1Output
from models.radar_lidar_stage1.sparse import SparseSites
from models.radar_lidar_stage2.config import Stage2Config
from models.radar_lidar_stage2.losses import reconstruction_loss
from models.radar_lidar_stage2.reconstruction_model import RadarLidarStage2
from models.radar_lidar_stage2.voxel_target import make_targets


def _stage1(grid: VoxelGrid, device: str) -> Stage1Output:
    levels = {}
    for i in range(4):
        stride = 2**i
        coordinates = torch.tensor([[0,z,y,x] for z in range(4//stride, 8//stride)
                                    for y in range(4//stride, 8//stride)
                                    for x in range(4//stride, 8//stride)],
                                   device=device, dtype=torch.long)
        features = torch.randn(len(coordinates), 4, device=device, requires_grad=True)
        levels[f"s{i+1}"] = SparseSites(coordinates, features, grid.scale_shape(stride), stride)
    confidence = levels["s1"].replace_features(torch.full((len(levels["s1"].coords),1), .9, device=device))
    return Stage1Output(levels, confidence, {name: sites.coords for name,sites in levels.items()}, {})


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(7)
    grid = VoxelGrid((0.,0.,0.), (8.,8.,8.), (.5,.5,.5))
    config = Stage2Config(channels=(8,12,16,24), conditioning_dim=8,
                          query_radii_m=(.75,1.5,3.,6.), max_neighbors=4,
                          expansion_zyx=(1,1,1), max_candidate_sites=3000)
    evidence = _stage1(grid, device)
    model = RadarLidarStage2((4,4,4,4), config).to(device)
    model.train()
    clean = torch.tensor([[[2.25,2.25,2.25,.1], [2.75,2.75,2.75,.2]]], device=device)
    valid = torch.ones((1,2), dtype=torch.bool, device=device)
    output = model(evidence, grid)
    target = make_targets(output.domain, clean, valid)
    losses = reconstruction_loss(output, target, config)
    assert len(output.candidate_coordinates) == len(output.occupancy_logits)
    assert output.metadata["sites"]["input"] == output.metadata["sites"]["output"]
    assert output.metadata["diffusion_enabled"] is False
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)
    assert all(sites.features.grad is None for sites in evidence.features.values())
    model.eval()
    with torch.no_grad():
        before = model(evidence, grid).occupancy_logits
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "stage2.pt"
            torch.save(model.state_dict(), path)
            reloaded = RadarLidarStage2((4,4,4,4), config).to(device).eval()
            reloaded.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            after = reloaded(evidence, grid).occupancy_logits
        assert torch.allclose(before, after, rtol=1e-5, atol=1e-6)
    report = {"device": device, "stage1_frozen": True, "diffusion_enabled": False,
              "candidate_sites": len(output.candidate_coordinates),
              "sites": output.metadata["sites"], "positive_sites": int(target.occupied.sum()),
              "known_free_sites": int(target.known_free.sum()),
              "loss": float(losses["total"]),
              "gradients_finite": True, "checkpoint_reproduces_output": True}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
