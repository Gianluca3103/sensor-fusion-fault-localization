"""Radar-only sparse reconstruction; clean LiDAR is never a forward input."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.radar_lidar_stage1.model import Stage1Output
from models.radar_lidar_stage1.sparse import encode_keys
from .candidate_domain import CandidateDomain, make_candidates
from .config import Stage2Config
from .stage1_conditioning import Stage1Conditioning
from .sparse_unet import ME, SparseUNet, _require_me
from .voxel_target import decode_centroids


@dataclass
class Stage2Output:
    candidate_coordinates: torch.Tensor
    occupancy_logits: torch.Tensor
    occupancy_probability: torch.Tensor
    predicted_offsets: torch.Tensor
    reconstructed_points_xyz: torch.Tensor
    reconstructed_confidence: torch.Tensor
    confidence: torch.Tensor
    metadata: dict
    domain: CandidateDomain


class RadarLidarStage2(nn.Module):
    def __init__(self, stage1_channels: tuple[int, ...], config: Stage2Config):
        super().__init__()
        _require_me()
        self.config = config
        self.conditioner = Stage1Conditioning(stage1_channels, config)
        self.unet = SparseUNet(config.channels)
        self.occupancy_head = nn.Linear(config.channels[0], 1)
        self.offset_head = nn.Linear(config.channels[0], 3)

    def forward(self, stage1: Stage1Output, grid, *, ablation: str = "real",
                replacement_stage1: Stage1Output | None = None,
                confidence_threshold: float | None = None,
                occupancy_threshold: float | None = None) -> Stage2Output:
        cfg = self.config
        domain = make_candidates(stage1, grid,
                                 confidence_threshold=cfg.confidence_threshold if confidence_threshold is None else confidence_threshold,
                                 expansion_zyx=cfg.expansion_zyx, max_sites=cfg.max_candidate_sites)
        n = len(domain.coordinates)
        if n == 0:
            zero = self.occupancy_head.weight.sum() * 0
            empty = zero.expand(0)
            return Stage2Output(domain.coordinates, empty, empty, zero.expand(0, 3),
                                zero.expand(0, 3), empty, domain.confidence,
                                {"sites": {"input": 0}, **domain.counts, "diffusion_enabled": False}, domain)
        features = self.conditioner(stage1, domain, ablation=ablation,
                                    replacement_stage1=replacement_stage1)
        sparse = ME.SparseTensor(features=features.contiguous(), coordinates=domain.coordinates.int())
        decoded, counts = self.unet(sparse)
        target_keys = encode_keys(domain.coordinates, grid.shape_zyx)
        result_keys = encode_keys(decoded.C.long().to(domain.coordinates.device), grid.shape_zyx)
        order = torch.argsort(result_keys)
        if not torch.equal(result_keys[order], target_keys):
            raise AssertionError("Sparse U-Net output differs from fixed candidate domain")
        hidden = decoded.F[order]
        logits = self.occupancy_head(hidden).squeeze(-1)
        probability = torch.sigmoid(logits)
        offsets = 0.5 * torch.tanh(self.offset_head(hidden))
        threshold = cfg.occupancy_threshold if occupancy_threshold is None else occupancy_threshold
        selected = probability > threshold
        physical = decode_centroids(domain, offsets)
        return Stage2Output(domain.coordinates, logits, probability, offsets,
                            physical[selected], domain.confidence[selected], domain.confidence,
                            {"sites": counts, **domain.counts, "diffusion_enabled": False,
                             "occupancy_threshold": threshold}, domain)
