"""Dense HRNet-style backbone for full-resolution BEV reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from ..encoders import _group_count


@dataclass(frozen=True)
class HRNetConfig:
    """Lightweight W16-like HRNet configuration for a 320x320 BEV."""

    base_channels: int = 16
    num_stages: int = 4
    blocks_per_stage: int = 2
    residual_blocks_per_branch: int = 2
    dropout: float = 0.0

    def validate(self) -> None:
        if self.base_channels < 1:
            raise ValueError("hrnet.base_channels must be positive")
        if not 1 <= self.num_stages <= 4:
            raise ValueError("hrnet.num_stages must be in [1,4]")
        if self.blocks_per_stage < 1:
            raise ValueError("hrnet.blocks_per_stage must be positive")
        if self.residual_blocks_per_branch < 1:
            raise ValueError(
                "hrnet.residual_blocks_per_branch must be positive"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("hrnet.dropout must be in [0,1)")

    @property
    def branch_channels(self) -> tuple[int, ...]:
        return tuple(
            self.base_channels * (2**index)
            for index in range(self.num_stages)
        )

    def to_dict(self) -> dict:
        return asdict(self)


class HRNetResidualBlock(nn.Module):
    """Project residual features using the repository's GN/SiLU convention."""

    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout else nn.Identity(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.activation(tensor + self.main(tensor))


def _residual_branch(
    channels: int,
    block_count: int,
    dropout: float,
) -> nn.Sequential:
    return nn.Sequential(
        *(HRNetResidualBlock(channels, dropout) for _ in range(block_count))
    )


class HRNetTransition(nn.Module):
    """Append one learned half-resolution branch without changing old streams."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, branches: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        outputs = list(branches)
        outputs.append(self.downsample(branches[-1]))
        return outputs


class _LowToHighTransform(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        )

    def forward(
        self,
        tensor: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        return F.interpolate(
            self.project(tensor),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )


def _high_to_low_transform(
    channels: Sequence[int],
    source_index: int,
    target_index: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_channels = channels[source_index]
    for step in range(source_index + 1, target_index + 1):
        next_channels = channels[step]
        layers.extend(
            (
                nn.Conv2d(
                    current_channels,
                    next_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(_group_count(next_channels), next_channels),
            )
        )
        if step != target_index:
            layers.append(nn.SiLU(inplace=True))
        current_channels = next_channels
    return nn.Sequential(*layers)


class HRNetFusion(nn.Module):
    """Fuse every active input resolution into every output resolution."""

    def __init__(self, channels: Sequence[int]):
        super().__init__()
        self.channels = tuple(channels)
        rows = []
        for target_index, target_channels in enumerate(self.channels):
            transforms = []
            for source_index, source_channels in enumerate(self.channels):
                if source_index == target_index:
                    transform = nn.Identity()
                elif source_index > target_index:
                    transform = _LowToHighTransform(
                        source_channels, target_channels
                    )
                else:
                    transform = _high_to_low_transform(
                        self.channels, source_index, target_index
                    )
                transforms.append(transform)
            rows.append(nn.ModuleList(transforms))
        self.transforms = nn.ModuleList(rows)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, branches: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if len(branches) != len(self.channels):
            raise ValueError("HRNet fusion received the wrong number of branches")
        outputs = []
        for target_index, transforms in enumerate(self.transforms):
            target_size = branches[target_index].shape[-2:]
            fused = None
            for source_index, transform in enumerate(transforms):
                source = branches[source_index]
                if isinstance(transform, _LowToHighTransform):
                    contribution = transform(source, target_size)
                else:
                    contribution = transform(source)
                    if contribution.shape[-2:] != target_size:
                        contribution = F.interpolate(
                            contribution,
                            size=target_size,
                            mode="bilinear",
                            align_corners=False,
                        )
                fused = contribution if fused is None else fused + contribution
            assert fused is not None
            outputs.append(self.activation(fused))
        return outputs


class HRNetModule(nn.Module):
    """Independent residual processing followed by all-to-all fusion."""

    def __init__(
        self,
        channels: Sequence[int],
        residual_blocks_per_branch: int,
        dropout: float,
    ):
        super().__init__()
        self.branches = nn.ModuleList(
            _residual_branch(channels_value, residual_blocks_per_branch, dropout)
            for channels_value in channels
        )
        self.fusion = HRNetFusion(channels)

    def forward(self, branches: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        processed = [
            branch(features)
            for branch, features in zip(self.branches, branches)
        ]
        return self.fusion(processed)


class HRNetBackbone(nn.Module):
    """Maintain a learned full-resolution stream through the complete model."""

    def __init__(self, in_channels: int, config: HRNetConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.channels = config.branch_channels
        self.initial_projection = nn.Sequential(
            nn.Conv2d(in_channels, self.channels[0], 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(self.channels[0]), self.channels[0]),
            nn.SiLU(inplace=True),
        )
        self.stage1 = _residual_branch(
            self.channels[0],
            config.residual_blocks_per_branch,
            config.dropout,
        )

        self.transitions = nn.ModuleList()
        self.stages = nn.ModuleList()
        for stage_index in range(1, config.num_stages):
            self.transitions.append(
                HRNetTransition(
                    self.channels[stage_index - 1],
                    self.channels[stage_index],
                )
            )
            # Version 1 uses one Stage-2 fusion and repeated Stage-3/4 fusion.
            module_count = 1 if stage_index == 1 else config.blocks_per_stage
            self.stages.append(
                nn.Sequential(
                    *(
                        HRNetModule(
                            self.channels[: stage_index + 1],
                            config.residual_blocks_per_branch,
                            config.dropout,
                        )
                        for _ in range(module_count)
                    )
                )
            )

        concatenated_channels = sum(self.channels)
        self.final_fusion = nn.Sequential(
            nn.Conv2d(concatenated_channels, 64, 1, bias=False),
            nn.GroupNorm(_group_count(64), 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(32), 32),
            nn.SiLU(inplace=True),
        )
        self.out_channels = 32
        self.concatenated_channels = concatenated_channels

    def forward(
        self, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        high_resolution = self.stage1(self.initial_projection(tensor))
        branches = [high_resolution]
        debug = {"hrnet_stage_1_branch_0": high_resolution}

        for stage_offset, (transition, stage) in enumerate(
            zip(self.transitions, self.stages), start=2
        ):
            branches = transition(branches)
            branches = stage(branches)
            for branch_index, features in enumerate(branches):
                debug[
                    f"hrnet_stage_{stage_offset}_branch_{branch_index}"
                ] = features

        target_size = branches[0].shape[-2:]
        resized = [branches[0]]
        resized.extend(
            F.interpolate(
                features,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            for features in branches[1:]
        )
        concatenated = torch.cat(resized, dim=1)
        final_features = self.final_fusion(concatenated)
        debug["hrnet_final_concatenated"] = concatenated
        debug["hrnet_final_features"] = final_features
        return final_features, debug
