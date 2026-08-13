"""Global self-attention over occupied Radar PointPillars tokens."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RadarPillarAttentionConfig:
    """Configuration for the isolated Radar PillarAttention ablation."""

    enabled: bool = False
    attention_dim: int = 128
    num_heads: int = 4
    ff_dim: int = 256
    num_blocks: int = 1
    dropout: float = 0.0

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("radar_pillar_attention.enabled must be boolean")
        for name in ("attention_dim", "num_heads", "ff_dim", "num_blocks"):
            if getattr(self, name) < 1:
                raise ValueError(f"radar_pillar_attention.{name} must be positive")
        if self.attention_dim % self.num_heads:
            raise ValueError(
                "radar_pillar_attention.attention_dim must be divisible by "
                "num_heads"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("radar_pillar_attention.dropout must be in [0,1)")

    def to_dict(self) -> dict:
        return asdict(self)


class RadarPillarAttentionBlock(nn.Module):
    """One post-normalized global self-attention and FFN block."""

    def __init__(self, config: RadarPillarAttentionConfig):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            config.attention_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.attention_norm = nn.LayerNorm(config.attention_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.attention_dim, config.ff_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, config.attention_dim),
        )
        self.feed_forward_dropout = nn.Dropout(config.dropout)
        self.feed_forward_norm = nn.LayerNorm(config.attention_dim)

    def forward(
        self,
        features: torch.Tensor,
        *,
        return_attention_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attended, weights = self.self_attention(
            features,
            features,
            features,
            need_weights=return_attention_weights,
            average_attn_weights=False,
        )
        features = self.attention_norm(
            features + self.attention_dropout(attended)
        )
        update = self.feed_forward(features)
        features = self.feed_forward_norm(
            features + self.feed_forward_dropout(update)
        )
        return features, weights


class RadarPillarAttention(nn.Module):
    """Apply global attention independently to each sample's occupied pillars."""

    def __init__(
        self,
        input_channels: int,
        config: RadarPillarAttentionConfig,
    ):
        super().__init__()
        config.validate()
        if input_channels < 1:
            raise ValueError("input_channels must be positive")
        self.input_channels = int(input_channels)
        self.config = config
        self.input_projection = nn.Linear(
            input_channels, config.attention_dim
        )
        self.blocks = nn.ModuleList(
            RadarPillarAttentionBlock(config)
            for _ in range(config.num_blocks)
        )
        self.output_projection = nn.Linear(
            config.attention_dim, input_channels
        )

    def forward(
        self,
        pillar_features: torch.Tensor,
        pillar_coordinates: torch.Tensor,
        batch_size: int,
        *,
        return_attention_weights: bool = False,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        if pillar_features.ndim != 2 or pillar_features.shape[1] != self.input_channels:
            raise ValueError(
                "pillar_features must have shape [P,C], got "
                f"{tuple(pillar_features.shape)}"
            )
        if pillar_coordinates.shape != (pillar_features.shape[0], 3):
            raise ValueError(
                "pillar_coordinates must have shape [P,3] as "
                "[batch,row,column]"
            )
        if pillar_coordinates.dtype != torch.long:
            raise ValueError("pillar_coordinates must use torch.long indices")
        if pillar_coordinates.device != pillar_features.device:
            raise ValueError("pillar features and coordinates must share a device")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if len(pillar_coordinates):
            batches = pillar_coordinates[:, 0]
            if int(batches.min()) < 0 or int(batches.max()) >= batch_size:
                raise ValueError("pillar batch indices are outside the batch")
        else:
            batches = pillar_coordinates.new_empty(0)

        attended_features = None
        token_counts = torch.bincount(batches, minlength=batch_size)
        attention_by_sample: list[tuple[torch.Tensor, ...]] = []
        for sample_index in range(batch_size):
            sample_indices = torch.nonzero(
                batches == sample_index, as_tuple=False
            ).flatten()
            if not len(sample_indices):
                if return_attention_weights:
                    attention_by_sample.append(tuple())
                continue
            sample = self.input_projection(
                pillar_features.index_select(0, sample_indices)
            ).unsqueeze(0)
            block_weights = []
            for block in self.blocks:
                sample, weights = block(
                    sample,
                    return_attention_weights=return_attention_weights,
                )
                if weights is not None:
                    block_weights.append(weights.squeeze(0))
            sample = self.output_projection(sample.squeeze(0))
            if attended_features is None:
                attended_features = torch.empty(
                    pillar_features.shape,
                    dtype=sample.dtype,
                    device=sample.device,
                )
            attended_features.index_copy_(0, sample_indices, sample)
            if return_attention_weights:
                attention_by_sample.append(tuple(block_weights))

        if attended_features is None:
            attended_features = pillar_features.new_empty(
                pillar_features.shape
            )
        debug: dict[str, object] = {
            "token_counts": token_counts,
            "attention_pair_counts": token_counts.square(),
            "occupied_fraction": token_counts.to(pillar_features.dtype)
            / (320 * 320),
            "feature_dimension": pillar_features.new_full(
                (batch_size,), self.input_channels
            ),
            "attention_heads": pillar_features.new_full(
                (batch_size,), self.config.num_heads
            ),
        }
        if return_attention_weights:
            debug["attention_weights"] = tuple(attention_by_sample)
        return attended_features, debug
