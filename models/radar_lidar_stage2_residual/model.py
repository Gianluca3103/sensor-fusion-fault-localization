"""Sparse additive completion conditioned on radar and surviving faulty LiDAR."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.radar_lidar_stage1.model import Stage1Output
from models.radar_lidar_stage1.sparse import encode_keys
from models.radar_lidar_stage2.candidate_domain import CandidateDomain, make_candidates
from models.radar_lidar_stage2.sparse_unet import ME, SparseUNet, _require_me
from models.radar_lidar_stage2.stage1_conditioning import Stage1Conditioning
from models.radar_lidar_stage2.voxel_target import decode_centroids
from .config import ResidualStage2Config
from .coverage import FaultyCoverage, faulty_coverage


@dataclass
class ResidualOutput:
    domain: CandidateDomain
    coverage: FaultyCoverage
    addition_logits: torch.Tensor
    addition_probability: torch.Tensor
    predicted_offsets: torch.Tensor
    reconstructed_points_xyz: torch.Tensor
    reconstructed_confidence: torch.Tensor
    metadata: dict


class ResidualRadarLidarStage2(nn.Module):
    def __init__(self, stage1_channels: tuple[int, ...], config: ResidualStage2Config):
        super().__init__()
        _require_me()
        self.config = config
        self.conditioner = Stage1Conditioning(stage1_channels, config)
        width = config.channels[0]
        self.faulty_encoder = nn.Sequential(nn.Linear(7, width), nn.LayerNorm(width), nn.SiLU())
        self.input_fuse = nn.Sequential(nn.Linear(width * 2, width), nn.LayerNorm(width), nn.SiLU())
        self.unet = SparseUNet(config.channels)
        self.addition_head = nn.Linear(width, 1)
        self.offset_head = nn.Linear(width, 3)

    def forward(self, stage1: Stage1Output, grid, faulty: torch.Tensor,
                faulty_valid: torch.Tensor, regions: list[dict], *,
                occupancy_threshold: float | None = None) -> ResidualOutput:
        cfg = self.config
        domain = make_candidates(stage1, grid, confidence_threshold=cfg.confidence_threshold,
                                 expansion_zyx=cfg.expansion_zyx, max_sites=cfg.max_candidate_sites)
        coverage = faulty_coverage(domain, faulty, faulty_valid, regions,
                                   ray_tolerance_m=cfg.ray_tolerance_m)
        n = len(domain.coordinates)
        if not n:
            zero = self.addition_head.weight.sum() * 0
            empty = zero.expand(0)
            return ResidualOutput(domain, coverage, empty, empty, zero.expand(0, 3),
                                  zero.expand(0, 3), empty, {"sites": {"input": 0}})
        radar_features = self.conditioner(stage1, domain)
        faulty_features = self.faulty_encoder(coverage.features)
        features = self.input_fuse(torch.cat((radar_features, faulty_features), dim=1))
        sparse = ME.SparseTensor(features=features.contiguous(), coordinates=domain.coordinates.int())
        decoded, counts = self.unet(sparse)
        target_keys = encode_keys(domain.coordinates, grid.shape_zyx)
        result_keys = encode_keys(decoded.C.long().to(domain.coordinates.device), grid.shape_zyx)
        order = torch.argsort(result_keys)
        if not torch.equal(result_keys[order], target_keys):
            raise AssertionError("Residual U-Net output differs from radar candidate domain")
        hidden = decoded.F[order]
        logits = self.addition_head(hidden).squeeze(-1)
        probability = torch.sigmoid(logits)
        offsets = .5 * torch.tanh(self.offset_head(hidden))
        threshold = cfg.occupancy_threshold if occupancy_threshold is None else occupancy_threshold
        selected = (probability > threshold) & coverage.may_add
        physical = decode_centroids(domain, offsets)
        return ResidualOutput(domain, coverage, logits, probability, offsets,
                              physical[selected], domain.confidence[selected],
                              {"sites": counts, **domain.counts, "selected": int(selected.sum()),
                               "blocked": int(coverage.blocked.sum()), "diffusion_enabled": False})
