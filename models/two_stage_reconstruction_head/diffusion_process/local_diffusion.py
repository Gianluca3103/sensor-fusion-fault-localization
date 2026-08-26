"""Local coarse-anchored residual-flow refiner for LiDAR reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F

from ..encoders import _group_count
from ..reconstruction_inputs import ReconstructionInputs
from .diffusion_process import (
    BEVChannelNormalization,
    MaskedFlowMSELoss,
    ResidualChannelNormalization,
    residual_target,
)


SUPPORTED_SAMPLING_STEPS = frozenset({1, 3, 5, 10, 25, 50})
FINE_DIFFUSION_ARCHITECTURE_VERSION = 11


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal timestep embedding used by the fine refiner."""

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = int(dimension)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timestep.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = timestep.float()[:, None] * frequencies[None]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
        if embedding.shape[1] < self.dimension:
            embedding = F.pad(embedding, (0, self.dimension - embedding.shape[1]))
        return embedding


@dataclass(frozen=True)
class FineDiffusionConfig:
    """Configuration for the coarse-anchored local residual-flow refiner."""

    enabled: bool = True
    bypass_coarse_reconstruction: bool = False
    lidar_channels: int = 3
    radar_channels: int = 4
    use_pointpillars_conditioning: bool = False
    lidar_pillar_channels: int = 64
    radar_pillar_channels: int = 64
    transformer_spatial_input_mode: str = "zero_residual"
    hidden_dim: int = 64
    num_heads: int = 4
    num_transformer_blocks: int = 4
    window_size: int = 8
    use_shifted_windows: bool = True
    use_global_faulty_context: bool = True
    global_context_dim: int = 128
    diffusion_prediction_type: str = "residual_flow"
    training_timesteps: int = 1000
    sampling_steps: int = 3
    lambda_diffusion: float = 1.0
    lambda_exact_reconstruction: float = 1.0
    lambda_degradation: float = 1.0
    lambda_residual_regularization: float = 0.05
    residual_regularization_mode: str = "cumulative_absolute"
    residual_regularization_decay_epochs: int = 0
    occupancy_loss_mode: str = "standard_bce"
    occupancy_threshold: float = 0.5
    correction_group_weight: float = 1.0
    preservation_group_weight: float = 0.5
    operation_add_weight: float = 0.21
    operation_remove_weight: float = 0.59
    operation_preserve_occupied_weight: float = 0.20
    operation_preserve_empty_weight: float = 1.00
    dropout: float = 0.0
    denominator_epsilon: float = 1.0e-8
    minimum_residual_std: float = 1.0e-4

    def validate(self) -> None:
        if not self.enabled:
            raise ValueError("Fine diffusion must be enabled")
        for name in (
            "lidar_channels",
            "radar_channels",
            "lidar_pillar_channels",
            "radar_pillar_channels",
            "hidden_dim",
            "num_heads",
            "num_transformer_blocks",
            "window_size",
            "global_context_dim",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not isinstance(self.use_pointpillars_conditioning, bool):
            raise ValueError("use_pointpillars_conditioning must be boolean")
        if self.transformer_spatial_input_mode not in (
            "zero_residual",
            "current_lidar",
        ):
            raise ValueError(
                "transformer_spatial_input_mode must be zero_residual or "
                "current_lidar"
            )
        if self.use_pointpillars_conditioning and self.bypass_coarse_reconstruction:
            raise ValueError(
                "PointPillars-conditioned fine diffusion cannot bypass the "
                "frozen coarse model that supplies its encoders"
            )
        if self.diffusion_prediction_type != "residual_flow":
            raise ValueError(
                "Fine refinement requires residual_flow prediction; Gaussian "
                "epsilon prediction is intentionally unsupported"
            )
        if self.training_timesteps < 2:
            raise ValueError("training_timesteps must be at least 2")
        if self.sampling_steps not in SUPPORTED_SAMPLING_STEPS:
            raise ValueError(
                "sampling_steps must be one of 1, 3, 5, 10, 25, or 50"
            )
        for name in (
            "lambda_diffusion",
            "lambda_exact_reconstruction",
            "lambda_degradation",
            "lambda_residual_regularization",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.residual_regularization_mode not in (
            "cumulative_absolute",
            "per_step_excess",
        ):
            raise ValueError(
                "residual_regularization_mode must be cumulative_absolute "
                "or per_step_excess"
            )
        if self.residual_regularization_decay_epochs < 0:
            raise ValueError(
                "residual_regularization_decay_epochs must be non-negative"
            )
        if self.occupancy_loss_mode not in (
            "standard_bce",
            "operation_balanced",
            "weighted_operation",
        ):
            raise ValueError(
                "occupancy_loss_mode must be standard_bce, operation_balanced, "
                "or weighted_operation"
            )
        if not 0.0 < self.occupancy_threshold < 1.0:
            raise ValueError("occupancy_threshold must be strictly between 0 and 1")
        for name in ("correction_group_weight", "preservation_group_weight"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.correction_group_weight + self.preservation_group_weight <= 0.0:
            raise ValueError("at least one occupancy group weight must be positive")
        operation_weight_names = (
            "operation_add_weight",
            "operation_remove_weight",
            "operation_preserve_occupied_weight",
            "operation_preserve_empty_weight",
        )
        for name in operation_weight_names:
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if sum(getattr(self, name) for name in operation_weight_names) <= 0.0:
            raise ValueError("at least one weighted-operation weight must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if self.denominator_epsilon <= 0:
            raise ValueError("denominator_epsilon must be positive")
        if self.minimum_residual_std <= 0:
            raise ValueError("minimum_residual_std must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReconstructionCropBatch:
    """Aligned, padded local crops and their full-BEV bounding boxes."""

    tensors: dict[str, torch.Tensor]
    boxes: torch.Tensor
    valid_mask: torch.Tensor
    active_samples: torch.Tensor
    crop_heights: torch.Tensor
    crop_widths: torch.Tensor
    full_height: int
    full_width: int

    def paste(
        self,
        crop: torch.Tensor,
        *,
        channels: int | None = None,
    ) -> torch.Tensor:
        output_channels = channels or crop.shape[1]
        output = crop.new_zeros(
            (crop.shape[0], output_channels, self.full_height, self.full_width)
        )
        for index, box in enumerate(self.boxes.tolist()):
            top, bottom, left, right = box
            height, width = bottom - top, right - left
            if bool(self.active_samples[index]):
                output[index, :, top:bottom, left:right] = crop[
                    index, :output_channels, :height, :width
                ]
        return output


class ReconstructionCropExtractor:
    """Crop the exact repair/halo union and add invalid technical padding."""

    def __init__(self, pad_multiple: int = 8):
        if pad_multiple < 1:
            raise ValueError("pad_multiple must be positive")
        self.pad_multiple = int(pad_multiple)

    @staticmethod
    def _extent(mask: torch.Tensor) -> tuple[int, int, int, int] | None:
        locations = torch.nonzero(mask, as_tuple=False)
        if locations.numel() == 0:
            return None
        rows, columns = locations[:, -2], locations[:, -1]
        return (
            int(rows.min()),
            int(rows.max()) + 1,
            int(columns.min()),
            int(columns.max()) + 1,
        )

    def _box(
        self,
        repair: torch.Tensor,
        halo: torch.Tensor,
    ) -> tuple[tuple[int, int, int, int], bool]:
        crop_mask = torch.maximum(repair, halo)
        crop_extent = self._extent(crop_mask > 0.5)
        if crop_extent is None:
            return (0, 1, 0, 1), False
        return crop_extent, True

    def extract(
        self,
        tensors: Mapping[str, torch.Tensor],
        reconstruction_mask: torch.Tensor,
        halo_mask: torch.Tensor,
    ) -> ReconstructionCropBatch:
        if reconstruction_mask.ndim != 4 or reconstruction_mask.shape[1] != 1:
            raise ValueError("reconstruction_mask must have shape [B,1,H,W]")
        if halo_mask.shape != reconstruction_mask.shape:
            raise ValueError("halo_mask must match reconstruction_mask")
        batch, _one, height, width = reconstruction_mask.shape
        for name, tensor in tensors.items():
            if tensor.ndim != 4 or tensor.shape[0] != batch:
                raise ValueError(f"{name} must have shape [B,C,H,W]")
            if tensor.shape[-2:] != (height, width):
                raise ValueError(f"{name} is not spatially aligned")
        boxes_and_active = [
            self._box(
                reconstruction_mask[index, 0],
                halo_mask[index, 0],
            )
            for index in range(batch)
        ]
        boxes = [item[0] for item in boxes_and_active]
        active = reconstruction_mask.new_tensor(
            [item[1] for item in boxes_and_active], dtype=torch.bool
        )
        crop_heights = [bottom - top for top, bottom, _left, _right in boxes]
        crop_widths = [right - left for _top, _bottom, left, right in boxes]
        padded_height = math.ceil(max(crop_heights) / self.pad_multiple) * self.pad_multiple
        padded_width = math.ceil(max(crop_widths) / self.pad_multiple) * self.pad_multiple
        cropped: dict[str, torch.Tensor] = {}
        for name, tensor in tensors.items():
            output = tensor.new_zeros(
                (batch, tensor.shape[1], padded_height, padded_width)
            )
            for index, box in enumerate(boxes):
                top, bottom, left, right = box
                output[index, :, : bottom - top, : right - left] = tensor[
                    index, :, top:bottom, left:right
                ]
            cropped[name] = output
        valid = reconstruction_mask.new_zeros(
            (batch, 1, padded_height, padded_width)
        )
        for index, (crop_height, crop_width) in enumerate(
            zip(crop_heights, crop_widths)
        ):
            valid[index, :, :crop_height, :crop_width] = 1
        return ReconstructionCropBatch(
            tensors=cropped,
            boxes=torch.tensor(boxes, device=reconstruction_mask.device),
            valid_mask=valid,
            active_samples=active,
            crop_heights=torch.tensor(crop_heights, device=reconstruction_mask.device),
            crop_widths=torch.tensor(crop_widths, device=reconstruction_mask.device),
            full_height=height,
            full_width=width,
        )


def _coordinate_channels(crops: ReconstructionCropBatch) -> torch.Tensor:
    """Return global and local XY coordinates without learned size coupling."""

    batch, _one, padded_height, padded_width = crops.valid_mask.shape
    coordinates = crops.valid_mask.new_zeros((batch, 4, padded_height, padded_width))
    for index, box in enumerate(crops.boxes.tolist()):
        top, bottom, left, right = box
        height, width = bottom - top, right - left
        global_y = torch.linspace(
            -1.0 + 2.0 * top / max(crops.full_height - 1, 1),
            -1.0 + 2.0 * (bottom - 1) / max(crops.full_height - 1, 1),
            height,
            device=coordinates.device,
            dtype=coordinates.dtype,
        )
        global_x = torch.linspace(
            -1.0 + 2.0 * left / max(crops.full_width - 1, 1),
            -1.0 + 2.0 * (right - 1) / max(crops.full_width - 1, 1),
            width,
            device=coordinates.device,
            dtype=coordinates.dtype,
        )
        local_y = torch.linspace(
            -1.0, 1.0, height, device=coordinates.device, dtype=coordinates.dtype
        )
        local_x = torch.linspace(
            -1.0, 1.0, width, device=coordinates.device, dtype=coordinates.dtype
        )
        coordinates[index, 0, :height, :width] = global_x[None, :]
        coordinates[index, 1, :height, :width] = global_y[:, None]
        coordinates[index, 2, :height, :width] = local_x[None, :]
        coordinates[index, 3, :height, :width] = local_y[:, None]
    return coordinates * crops.valid_mask


class GlobalFaultyLidarEncoder(nn.Module):
    """Small strided convolutional encoder returning one scene vector."""

    def __init__(self, input_channels: int, output_dim: int, hidden_dim: int):
        super().__init__()
        widths = (hidden_dim // 2, hidden_dim, hidden_dim)
        layers: list[nn.Module] = []
        current = input_channels
        for width in widths:
            layers.extend(
                (
                    nn.Conv2d(current, width, 3, stride=2, padding=1),
                    nn.GroupNorm(_group_count(width), width),
                    nn.SiLU(inplace=True),
                )
            )
            current = width
        self.encoder = nn.Sequential(*layers)
        self.projection = nn.Linear(current, output_dim)

    def forward(self, trusted_faulty_bev: torch.Tensor) -> torch.Tensor:
        features = self.encoder(trusted_faulty_bev)
        return self.projection(F.adaptive_avg_pool2d(features, 1).flatten(1))


class AuxiliaryConditionEncoder(nn.Module):
    """Encode non-coarse local evidence without altering the coarse BEV."""

    def __init__(self, input_channels: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )

    def forward(self, condition: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return self.encoder(condition) * valid


class AdaptiveLayerNorm2d(nn.Module):
    """Per-cell LayerNorm modulated by timestep/global conditioning."""

    def __init__(self, channels: int, condition_dim: int):
        super().__init__()
        self.normalization = nn.LayerNorm(channels, elementwise_affine=False)
        self.modulation = nn.Linear(condition_dim, 2 * channels)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, tensor: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        normalized = self.normalization(tensor.permute(0, 2, 3, 1)).permute(
            0, 3, 1, 2
        )
        scale, shift = self.modulation(condition).chunk(2, dim=1)
        return normalized * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]


def _partition_windows(
    tensor: torch.Tensor,
    valid: torch.Tensor,
    window_size: int,
    shift: int,
):
    batch, channels, height, width = tensor.shape
    pad_top = pad_left = shift
    padded_height = math.ceil((height + shift) / window_size) * window_size
    padded_width = math.ceil((width + shift) / window_size) * window_size
    pad_bottom = padded_height - height - pad_top
    pad_right = padded_width - width - pad_left
    tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom))
    valid = F.pad(valid, (pad_left, pad_right, pad_top, pad_bottom))
    rows, columns = padded_height // window_size, padded_width // window_size
    windows = (
        tensor.reshape(batch, channels, rows, window_size, columns, window_size)
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(batch * rows * columns, window_size * window_size, channels)
    )
    valid_windows = (
        valid.reshape(batch, 1, rows, window_size, columns, window_size)
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(batch * rows * columns, window_size * window_size)
        > 0.5
    )
    metadata = (
        batch,
        channels,
        height,
        width,
        padded_height,
        padded_width,
        rows,
        columns,
        shift,
    )
    return windows, valid_windows, metadata


def _reverse_windows(windows: torch.Tensor, metadata) -> torch.Tensor:
    (
        batch,
        channels,
        height,
        width,
        padded_height,
        padded_width,
        rows,
        columns,
        shift,
    ) = metadata
    window_size = int(math.sqrt(windows.shape[1]))
    tensor = (
        windows.reshape(batch, rows, columns, window_size, window_size, channels)
        .permute(0, 5, 1, 3, 2, 4)
        .reshape(batch, channels, padded_height, padded_width)
    )
    return tensor[:, :, shift : shift + height, shift : shift + width]


class WindowAttention2d(nn.Module):
    """Windowed attention supporting independent query and key/value maps."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        window_size: int,
        dropout: float,
        *,
        key_value_dim: int | None = None,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=key_value_dim,
            vdim=key_value_dim,
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        valid: torch.Tensor,
        *,
        shift: int = 0,
        key_value_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query_windows, valid_windows, metadata = _partition_windows(
            query, valid, self.window_size, shift
        )
        key_windows, key_valid, _ = _partition_windows(
            key_value,
            valid if key_value_valid is None else key_value_valid,
            self.window_size,
            shift,
        )
        active = valid_windows.any(dim=1) & key_valid.any(dim=1)
        output = torch.zeros_like(query_windows)
        if bool(active.any()):
            attended, _weights = self.attention(
                query_windows[active],
                key_windows[active],
                key_windows[active],
                key_padding_mask=~key_valid[active],
                need_weights=False,
            )
            output[active] = attended.to(output.dtype) * valid_windows[
                active, :, None
            ]
        return _reverse_windows(output, metadata) * valid


class ConvolutionalFFN(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        expanded = 4 * channels
        self.layers = nn.Sequential(
            nn.Conv2d(channels, expanded, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(expanded, expanded, 3, padding=1, groups=expanded),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout else nn.Identity(),
            nn.Conv2d(expanded, channels, 1),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.layers(tensor)


class DiffusionRefinementBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        window_size: int,
        condition_dim: int,
        dropout: float,
        shifted: bool,
        radar_channels: int,
    ):
        super().__init__()
        self.shift = window_size // 2 if shifted else 0
        self.self_norm = AdaptiveLayerNorm2d(hidden_dim, condition_dim)
        self.radar_cross_norm = AdaptiveLayerNorm2d(hidden_dim, condition_dim)
        self.ffn_norm = AdaptiveLayerNorm2d(hidden_dim, condition_dim)
        self.self_attention = WindowAttention2d(
            hidden_dim, num_heads, window_size, dropout
        )
        self.radar_cross_attention = WindowAttention2d(
            hidden_dim,
            num_heads,
            window_size,
            dropout,
            key_value_dim=radar_channels,
        )
        self.ffn = ConvolutionalFFN(hidden_dim, dropout)

    def forward(
        self,
        tensor: torch.Tensor,
        raw_radar: torch.Tensor,
        condition_vector: torch.Tensor,
        valid: torch.Tensor,
        radar_valid: torch.Tensor,
    ) -> torch.Tensor:
        tensor = tensor + self.self_attention(
            self.self_norm(tensor, condition_vector),
            self.self_norm(tensor, condition_vector),
            valid,
            shift=self.shift,
        )
        tensor = tensor + self.radar_cross_attention(
            self.radar_cross_norm(tensor, condition_vector),
            raw_radar,
            valid,
            shift=self.shift,
            key_value_valid=radar_valid,
        )
        tensor = tensor + self.ffn(self.ffn_norm(tensor, condition_vector)) * valid
        return tensor * valid


class LocalResidualDiffusionTransformer(nn.Module):
    """Resolution-preserving local residual-flow velocity predictor."""

    def __init__(self, config: FineDiffusionConfig):
        super().__init__()
        config.validate()
        self.config = config
        coordinates = 4
        auxiliary_channels = config.lidar_channels + 2 + coordinates
        spatial_stem = nn.Conv2d(
            config.lidar_channels + coordinates,
            config.hidden_dim,
            3,
            padding=1,
        )
        if config.transformer_spatial_input_mode == "zero_residual":
            self.residual_stem = spatial_stem
        else:
            self.current_lidar_stem = spatial_stem
        self.lidar_pillar_stem = (
            nn.Conv2d(
                config.lidar_pillar_channels,
                config.hidden_dim,
                3,
                padding=1,
            )
            if config.use_pointpillars_conditioning
            else None
        )
        self.auxiliary_condition_encoder = AuxiliaryConditionEncoder(
            auxiliary_channels, config.hidden_dim
        )
        condition_dim = config.global_context_dim
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(condition_dim),
            nn.Linear(condition_dim, condition_dim),
            nn.SiLU(inplace=True),
            nn.Linear(condition_dim, condition_dim),
        )
        if config.use_global_faulty_context:
            self.global_encoder = GlobalFaultyLidarEncoder(
                config.lidar_channels,
                config.global_context_dim,
                config.hidden_dim,
            )
        else:
            self.global_encoder = None
        self.blocks = nn.ModuleList(
            DiffusionRefinementBlock(
                config.hidden_dim,
                config.num_heads,
                config.window_size,
                condition_dim,
                config.dropout,
                config.use_shifted_windows and index % 2 == 1,
                (
                    config.radar_pillar_channels
                    if config.use_pointpillars_conditioning
                    else config.radar_channels
                ),
            )
            for index in range(config.num_transformer_blocks)
        )
        self.output_head = nn.Sequential(
            nn.GroupNorm(_group_count(config.hidden_dim), config.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(config.hidden_dim, config.lidar_channels, 3, padding=1),
        )
        # An untrained refiner is an identity mapping: zero velocity means the
        # output remains exactly the frozen coarse reconstruction.
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def global_context(
        self,
        faulty_lidar_bev: torch.Tensor,
        reconstruction_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trusted = faulty_lidar_bev * (1.0 - reconstruction_mask)
        if self.global_encoder is None:
            embedding = faulty_lidar_bev.new_zeros(
                (faulty_lidar_bev.shape[0], self.config.global_context_dim)
            )
        else:
            embedding = self.global_encoder(trusted)
        return trusted, embedding

    def forward(
        self,
        spatial_state: torch.Tensor,
        trusted_faulty: torch.Tensor,
        radar: torch.Tensor,
        lidar_pillars: torch.Tensor | None,
        radar_pillars: torch.Tensor | None,
        reconstruction_mask: torch.Tensor,
        halo_mask: torch.Tensor,
        valid: torch.Tensor,
        coordinates: torch.Tensor,
        timestep: torch.Tensor,
        global_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        auxiliary_condition_input = torch.cat(
            (
                trusted_faulty,
                reconstruction_mask,
                halo_mask,
                coordinates,
            ),
            dim=1,
        )
        auxiliary_condition = self.auxiliary_condition_encoder(
            auxiliary_condition_input, valid
        )
        spatial_stem_input = torch.cat((spatial_state, coordinates), dim=1)
        if self.config.use_pointpillars_conditioning:
            if lidar_pillars is None or radar_pillars is None:
                raise ValueError(
                    "Fine diffusion requires both LiDAR and radar PointPillars "
                    "feature maps"
                )
            assert self.lidar_pillar_stem is not None
            lidar_pillar_condition = self.lidar_pillar_stem(lidar_pillars)
            radar_attention = radar_pillars
        else:
            lidar_pillar_condition = torch.zeros_like(auxiliary_condition)
            radar_attention = radar
        spatial_stem = (
            self.residual_stem
            if self.config.transformer_spatial_input_mode == "zero_residual"
            else self.current_lidar_stem
        )
        tensor = (
            spatial_stem(spatial_stem_input)
            + auxiliary_condition
            + lidar_pillar_condition
        ) * valid
        condition_vector = self.time_embedding(timestep) + global_embedding
        radar_valid = reconstruction_mask * valid
        for block in self.blocks:
            tensor = block(
                tensor,
                radar_attention,
                condition_vector,
                valid,
                radar_valid,
            )
        velocity = torch.tanh(self.output_head(tensor))
        active_repair = (reconstruction_mask > 0.5) & (valid > 0.5)
        velocity = torch.where(
            active_repair.expand_as(velocity),
            velocity,
            torch.zeros_like(velocity),
        )
        debug = {
            "radar_cross_attention": radar_attention,
            "radar_cross_attention_valid": radar_valid,
            "lidar_pillar_condition": lidar_pillar_condition,
            "spatial_stem_input": spatial_stem_input,
        }
        if self.config.transformer_spatial_input_mode == "zero_residual":
            debug["current_residual"] = spatial_state
            debug["residual_stem_input"] = spatial_stem_input
        else:
            debug["current_lidar"] = spatial_state
            debug["current_lidar_stem_input"] = spatial_stem_input
        if not self.config.use_pointpillars_conditioning:
            debug["raw_radar_cross_attention"] = radar_attention
        return velocity, auxiliary_condition, debug


def fine_diffusion_architecture_metadata(
    config: FineDiffusionConfig,
) -> dict[str, bool | int | str]:
    return {
        "version": FINE_DIFFUSION_ARCHITECTURE_VERSION,
        "process": "coarse_anchored_residual_flow",
        "initial_state": "frozen_coarse_reconstruction",
        "internal_state": "normalized_current_residual",
        "transformer_spatial_input_mode": config.transformer_spatial_input_mode,
        "transformer_spatial_input": (
            f"{config.transformer_spatial_input_mode}_plus_coordinates_plus_"
            "lidar_pointpillars"
            if config.use_pointpillars_conditioning
            else f"{config.transformer_spatial_input_mode}_plus_coordinates"
        ),
        "spatial_stem_input_channels": config.lidar_channels + 4,
        "transformer_initial_spatial_state": (
            "zero_residual"
            if config.transformer_spatial_input_mode == "zero_residual"
            else "frozen_coarse_reconstruction"
        ),
        "refinement_feedback": (
            "updated_normalized_residual_each_step"
            if config.transformer_spatial_input_mode == "zero_residual"
            else "updated_current_lidar_each_step"
        ),
        "coarse_role": "frozen_reconstruction_baseline_only",
        "pointpillars_conditioning": config.use_pointpillars_conditioning,
        "lidar_pillar_channels": config.lidar_pillar_channels,
        "radar_pillar_channels": config.radar_pillar_channels,
        "lidar_pillar_role": (
            "spatial_stem_additive_condition"
            if config.use_pointpillars_conditioning
            else "disabled"
        ),
        "radar_cross_attention": (
            "radar_pointpillars_postscatter"
            if config.use_pointpillars_conditioning
            else "raw_full_resolution_reconstruction_mask"
        ),
    }


def validate_fine_diffusion_checkpoint_compatibility(
    checkpoint: Mapping, config: FineDiffusionConfig
) -> None:
    expected = config.lidar_channels + 4
    state = checkpoint.get("diffusion_state_dict", {})
    metadata = checkpoint.get("fine_diffusion_architecture")
    architecture_version = (
        metadata.get("version") if isinstance(metadata, Mapping) else None
    )
    if architecture_version == 10:
        expected_mode = "current_lidar"
        stem_name = "transformer.current_lidar_stem.weight"
    elif architecture_version == FINE_DIFFUSION_ARCHITECTURE_VERSION:
        expected_mode = "zero_residual"
        stem_name = "transformer.residual_stem.weight"
    else:
        raise ValueError(
            "Fine Diffusion checkpoint is neither supported v10 current-LiDAR "
            "evaluation nor v11 zero-residual refinement."
        )
    if config.transformer_spatial_input_mode != expected_mode:
        raise ValueError(
            "Fine Diffusion spatial-input mode does not match its checkpoint."
        )
    weight = state.get(stem_name)
    actual = int(weight.shape[1]) if torch.is_tensor(weight) else None
    if actual != expected:
        raise ValueError(
            "Fine Diffusion spatial-stem input mismatch: checkpoint has "
            f"{actual} channels but this model requires {expected}."
        )
    has_lidar_pillar_stem = "transformer.lidar_pillar_stem.weight" in state
    if has_lidar_pillar_stem != config.use_pointpillars_conditioning:
        raise ValueError(
            "Fine Diffusion PointPillars conditioning does not match the "
            "checkpoint architecture. Start a fresh run."
        )


class MaskedExactReconstructionLoss(nn.Module):
    """Exact cell-aligned channel-aware loss inside the repair mask."""

    GROUPS = (
        "add",
        "remove",
        "preserve_occupied",
        "preserve_empty",
    )

    def __init__(
        self,
        epsilon: float = 1.0e-8,
        *,
        occupancy_loss_mode: str = "standard_bce",
        occupancy_threshold: float = 0.5,
        correction_group_weight: float = 1.0,
        preservation_group_weight: float = 0.5,
        operation_add_weight: float = 0.21,
        operation_remove_weight: float = 0.59,
        operation_preserve_occupied_weight: float = 0.20,
        operation_preserve_empty_weight: float = 1.00,
    ):
        super().__init__()
        self.epsilon = float(epsilon)
        self.occupancy_loss_mode = str(occupancy_loss_mode)
        self.occupancy_threshold = float(occupancy_threshold)
        self.balanced_group_weights = {
            "add": float(correction_group_weight),
            "remove": float(correction_group_weight),
            "preserve_occupied": float(preservation_group_weight),
            "preserve_empty": float(preservation_group_weight),
        }
        self.weighted_operation_group_weights = {
            "add": float(operation_add_weight),
            "remove": float(operation_remove_weight),
            "preserve_occupied": float(operation_preserve_occupied_weight),
            "preserve_empty": float(operation_preserve_empty_weight),
        }

    def _occupancy_groups(
        self,
        clean_occupancy: torch.Tensor,
        coarse_occupancy: torch.Tensor,
        selected: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        clean_occupied = clean_occupancy >= self.occupancy_threshold
        coarse_occupied = coarse_occupancy >= self.occupancy_threshold
        return {
            "add": selected & clean_occupied & ~coarse_occupied,
            "remove": selected & ~clean_occupied & coarse_occupied,
            "preserve_occupied": selected & clean_occupied & coarse_occupied,
            "preserve_empty": selected & ~clean_occupied & ~coarse_occupied,
        }

    def forward(
        self,
        refined: torch.Tensor,
        clean: torch.Tensor,
        reconstruction_mask: torch.Tensor,
        coarse: torch.Tensor | None = None,
        *,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.autocast(device_type=refined.device.type, enabled=False):
            refined_float = refined.float()
            clean_float = clean.float()
            mask_float = reconstruction_mask.float()
            occupied = clean_float[:, 0:1].clamp(0.0, 1.0)
            selected = mask_float > 0.5
            refined_occupancy_map = refined_float[:, 0:1]
            refined_occupancy = refined_occupancy_map[selected]
            target_occupancy = occupied[selected]
            if not bool(torch.isfinite(refined_occupancy).all()):
                raise FloatingPointError(
                    "Fine reconstruction occupancy is non-finite inside the "
                    "reconstruction mask"
                )
            if refined_occupancy.numel() == 0:
                zero = refined_float.sum() * 0.0
                components = {
                    "occupancy_loss": zero,
                    "continuous_loss": zero,
                    **{f"occupancy_{name}_loss": zero for name in self.GROUPS},
                    **{
                        f"num_{name}": zero.detach()
                        for name in self.GROUPS
                    },
                }
                return (zero, components) if return_components else zero
            per_cell_bce = F.binary_cross_entropy(
                refined_occupancy.clamp(1.0e-6, 1.0 - 1.0e-6),
                target_occupancy,
                reduction="none",
            )
            group_losses = {
                name: per_cell_bce.new_zeros(()) for name in self.GROUPS
            }
            group_counts = {
                name: per_cell_bce.new_zeros(()) for name in self.GROUPS
            }
            if coarse is not None:
                coarse_occupancy = coarse.detach().float()[:, 0:1]
                groups = self._occupancy_groups(
                    occupied, coarse_occupancy, selected
                )
                selected_groups = {
                    name: group[selected] for name, group in groups.items()
                }
                for name, group in selected_groups.items():
                    count = group.sum(dtype=torch.float32)
                    group_counts[name] = count
                    if bool(count > 0):
                        group_losses[name] = per_cell_bce[group].sum() / count
            if self.occupancy_loss_mode in (
                "operation_balanced",
                "weighted_operation",
            ):
                if coarse is None:
                    raise ValueError(
                        "operation-aware occupancy loss requires detached "
                        "coarse occupancy"
                    )
                group_weights = (
                    self.weighted_operation_group_weights
                    if self.occupancy_loss_mode == "weighted_operation"
                    else self.balanced_group_weights
                )
                numerator = per_cell_bce.new_zeros(())
                active_weight = 0.0
                for name in self.GROUPS:
                    if bool(group_counts[name] > 0):
                        weight = group_weights[name]
                        numerator = numerator + weight * group_losses[name]
                        active_weight += weight
                occupancy_loss = numerator / max(active_weight, self.epsilon)
            else:
                occupancy_loss = per_cell_bce.mean()
        continuous_loss = occupancy_loss.new_zeros(())
        if refined.shape[1] == 1:
            total = occupancy_loss
            components = {
                "occupancy_loss": occupancy_loss,
                "continuous_loss": continuous_loss,
                **{
                    f"occupancy_{name}_loss": group_losses[name]
                    for name in self.GROUPS
                },
                **{f"num_{name}": group_counts[name] for name in self.GROUPS},
            }
            return (total, components) if return_components else total
        continuous_selected = (selected & (occupied > 0.5)).expand(
            -1, refined.shape[1] - 1, -1, -1
        )
        refined_continuous = refined_float[:, 1:][continuous_selected]
        clean_continuous = clean_float[:, 1:][continuous_selected]
        if not bool(torch.isfinite(refined_continuous).all()):
            raise FloatingPointError(
                "Fine reconstruction geometry is non-finite inside occupied "
                "reconstruction-mask cells"
            )
        continuous_loss = (
            F.smooth_l1_loss(
                refined_continuous, clean_continuous, reduction="mean"
            )
            if refined_continuous.numel()
            else occupancy_loss.new_zeros(())
        )
        total = occupancy_loss + continuous_loss
        components = {
            "occupancy_loss": occupancy_loss,
            "continuous_loss": continuous_loss,
            **{
                f"occupancy_{name}_loss": group_losses[name]
                for name in self.GROUPS
            },
            **{f"num_{name}": group_counts[name] for name in self.GROUPS},
        }
        return (total, components) if return_components else total


class MaskedNoDegradationLoss(nn.Module):
    """Penalize refined cells only when they are worse than coarse cells."""

    def __init__(self, epsilon: float = 1.0e-8):
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(
        self,
        refined: torch.Tensor,
        coarse: torch.Tensor,
        clean: torch.Tensor,
        reconstruction_mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.autocast(device_type=refined.device.type, enabled=False):
            refined_float = refined.float()
            coarse_float = coarse.detach().float()
            clean_float = clean.float()
            mask_float = reconstruction_mask.float()
            occupied = clean_float[:, 0:1].clamp(0.0, 1.0)
            selected = mask_float > 0.5
            refined_occupancy = refined_float[:, 0:1][selected]
            coarse_occupancy = coarse_float[:, 0:1][selected]
            target_occupancy = occupied[selected]
            if not bool(torch.isfinite(refined_occupancy).all()):
                raise FloatingPointError(
                    "Fine reconstruction occupancy is non-finite inside the "
                    "reconstruction mask"
                )
            if refined_occupancy.numel() == 0:
                return refined_float.sum() * 0.0

            refined_occupancy_error = F.binary_cross_entropy(
                refined_occupancy.clamp(1.0e-6, 1.0 - 1.0e-6),
                target_occupancy,
                reduction="none",
            )
            coarse_occupancy_error = F.binary_cross_entropy(
                coarse_occupancy.clamp(1.0e-6, 1.0 - 1.0e-6),
                target_occupancy,
                reduction="none",
            )
            occupancy_degradation = F.relu(
                refined_occupancy_error - coarse_occupancy_error
            )
            occupancy_loss = occupancy_degradation.mean()

            if refined.shape[1] == 1:
                return occupancy_loss

            continuous_selected = (selected & (occupied > 0.5)).expand(
                -1, refined.shape[1] - 1, -1, -1
            )
            refined_continuous = refined_float[:, 1:][continuous_selected]
            coarse_continuous = coarse_float[:, 1:][continuous_selected]
            clean_continuous = clean_float[:, 1:][continuous_selected]
            if not bool(torch.isfinite(refined_continuous).all()):
                raise FloatingPointError(
                    "Fine reconstruction geometry is non-finite inside occupied "
                    "reconstruction-mask cells"
                )
            if refined_continuous.numel() == 0:
                return occupancy_loss
            refined_continuous_error = F.smooth_l1_loss(
                refined_continuous, clean_continuous, reduction="none"
            )
            coarse_continuous_error = F.smooth_l1_loss(
                coarse_continuous, clean_continuous, reduction="none"
            )
            continuous_degradation = F.relu(
                refined_continuous_error - coarse_continuous_error
            )
            continuous_loss = continuous_degradation.mean()
            return occupancy_loss + continuous_loss


class FineDiffusionRefiner(nn.Module):
    """Train and integrate masked corrections from a frozen coarse BEV."""

    def __init__(
        self,
        config: FineDiffusionConfig | None = None,
        normalization: BEVChannelNormalization | None = None,
        residual_normalization: ResidualChannelNormalization | None = None,
    ):
        super().__init__()
        self.config = config or FineDiffusionConfig()
        self.config.validate()
        self.crop_extractor = ReconstructionCropExtractor(self.config.window_size)
        self.normalization = normalization or BEVChannelNormalization(
            means=(0.0,) * self.config.lidar_channels,
            stds=(1.0,) * self.config.lidar_channels,
        )
        self.residual_normalization = (
            residual_normalization
            or ResidualChannelNormalization(
                (1.0,) * self.config.lidar_channels,
                minimum_std=self.config.minimum_residual_std,
                source="identity_fallback",
            )
        )
        self.transformer = LocalResidualDiffusionTransformer(self.config)
        self.diffusion_loss = MaskedFlowMSELoss(
            self.config.denominator_epsilon
        )
        self.exact_loss = MaskedExactReconstructionLoss(
            self.config.denominator_epsilon,
            occupancy_loss_mode=self.config.occupancy_loss_mode,
            occupancy_threshold=self.config.occupancy_threshold,
            correction_group_weight=self.config.correction_group_weight,
            preservation_group_weight=self.config.preservation_group_weight,
            operation_add_weight=self.config.operation_add_weight,
            operation_remove_weight=self.config.operation_remove_weight,
            operation_preserve_occupied_weight=(
                self.config.operation_preserve_occupied_weight
            ),
            operation_preserve_empty_weight=(
                self.config.operation_preserve_empty_weight
            ),
        )
        self.degradation_loss = MaskedNoDegradationLoss(
            self.config.denominator_epsilon
        )

    def _residual_regularization_loss(
        self,
        predicted_residual_physical: torch.Tensor,
        reconstruction_mask: torch.Tensor,
    ) -> torch.Tensor:
        denominator = (
            reconstruction_mask.sum() * predicted_residual_physical.shape[1]
            + self.config.denominator_epsilon
        )
        return (
            predicted_residual_physical.abs() * reconstruction_mask
        ).sum() / denominator

    def _per_step_excess_regularization_loss(
        self,
        predicted_step_normalized: torch.Tensor,
        target_step_normalized: torch.Tensor,
        reconstruction_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize only per-step magnitude beyond the required target step."""

        predicted_physical = self.residual_normalization.denormalize(
            predicted_step_normalized
        )
        target_physical = self.residual_normalization.denormalize(
            target_step_normalized
        )
        excess = F.relu(predicted_physical.abs() - target_physical.abs())
        denominator = (
            reconstruction_mask.sum() * predicted_physical.shape[1]
            + self.config.denominator_epsilon
        )
        return (excess * reconstruction_mask).sum() / denominator

    def _bounded_residual_update(
        self,
        current_residual: torch.Tensor,
        predicted_step: torch.Tensor,
        coarse_crop: torch.Tensor,
        repair: torch.Tensor,
    ) -> torch.Tensor:
        selected = repair > 0.5
        updated = torch.where(
            selected.expand_as(current_residual),
            current_residual + predicted_step,
            torch.zeros_like(current_residual),
        )
        physical = self.residual_normalization.denormalize(updated)
        bounded_physical = torch.where(
            selected.expand_as(physical),
            (coarse_crop + physical).clamp(0.0, 1.0) - coarse_crop,
            torch.zeros_like(physical),
        )
        normalized = self.residual_normalization.normalize(bounded_physical)
        return torch.where(
            selected.expand_as(normalized),
            normalized,
            torch.zeros_like(normalized),
        )

    def _validate_inputs(
        self,
        coarse_lidar_bev: torch.Tensor,
        faulty_lidar_bev: torch.Tensor,
        radar_bev: torch.Tensor,
        reconstruction_mask: torch.Tensor,
        halo_mask: torch.Tensor,
        clean_lidar_bev: torch.Tensor | None = None,
    ) -> None:
        if coarse_lidar_bev.ndim != 4:
            raise ValueError("LiDAR BEVs must have shape [B,C,H,W]")
        expected_lidar = (
            coarse_lidar_bev.shape[0],
            self.config.lidar_channels,
            *coarse_lidar_bev.shape[-2:],
        )
        for name, tensor in (
            ("coarse_lidar_bev", coarse_lidar_bev),
            ("faulty_lidar_bev", faulty_lidar_bev),
        ):
            if tuple(tensor.shape) != expected_lidar:
                raise ValueError(f"{name} must have shape {expected_lidar}")
        if clean_lidar_bev is not None and tuple(clean_lidar_bev.shape) != expected_lidar:
            raise ValueError(f"clean_lidar_bev must have shape {expected_lidar}")
        expected_radar = (
            coarse_lidar_bev.shape[0],
            self.config.radar_channels,
            *coarse_lidar_bev.shape[-2:],
        )
        if tuple(radar_bev.shape) != expected_radar:
            raise ValueError(f"radar_bev must have shape {expected_radar}")
        expected_mask = (
            coarse_lidar_bev.shape[0],
            1,
            *coarse_lidar_bev.shape[-2:],
        )
        if reconstruction_mask.shape != expected_mask or halo_mask.shape != expected_mask:
            raise ValueError(f"masks must have shape {expected_mask}")
        tensors = (
            coarse_lidar_bev,
            faulty_lidar_bev,
            radar_bev,
            reconstruction_mask,
            halo_mask,
        )
        if clean_lidar_bev is not None:
            tensors = (*tensors, clean_lidar_bev)
        if any(tensor.device != coarse_lidar_bev.device for tensor in tensors):
            raise ValueError("All fine-diffusion inputs must share a device")
        if any(tensor.dtype != coarse_lidar_bev.dtype for tensor in tensors):
            raise TypeError("All fine-diffusion inputs must share a floating dtype")
        if not all(torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("Fine-diffusion input contains NaN or Inf")

    def _extract(
        self,
        coarse_lidar_bev: torch.Tensor,
        faulty_lidar_bev: torch.Tensor,
        radar_bev: torch.Tensor,
        reconstruction_mask: torch.Tensor,
        halo_mask: torch.Tensor,
        *,
        clean_lidar_bev: torch.Tensor | None = None,
        shared_inputs: ReconstructionInputs | None = None,
    ) -> ReconstructionCropBatch:
        if shared_inputs is None:
            shared_inputs = ReconstructionInputs(
                faulty_lidar_bev=faulty_lidar_bev,
                radar_bev=radar_bev,
                reconstruction_mask=reconstruction_mask,
                healthy_context_mask=torch.zeros_like(reconstruction_mask),
                halo_mask=halo_mask,
            )
        residual_gt = None
        if clean_lidar_bev is not None:
            residual_gt = residual_target(
                clean_lidar_bev, coarse_lidar_bev, reconstruction_mask
            )
        tensors = shared_inputs.fine_crop_tensors(
            coarse_lidar_bev,
            clean_lidar_bev=clean_lidar_bev,
            residual_gt=residual_gt,
        )
        return self.crop_extractor.extract(
            tensors, reconstruction_mask, halo_mask
        )

    def _predict_velocity(
        self,
        residual_t: torch.Tensor,
        crops: ReconstructionCropBatch,
        timestep: torch.Tensor,
        global_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        values = crops.tensors
        if self.config.transformer_spatial_input_mode == "current_lidar":
            spatial_state = values["coarse"] + (
                self.residual_normalization.denormalize(residual_t)
            )
        else:
            spatial_state = residual_t
        velocity_physical, condition, debug = self.transformer(
            spatial_state,
            self.normalization.normalize(values["trusted_faulty"]),
            values["radar"],
            values.get("lidar_pillars"),
            values.get("radar_pillars"),
            values["repair"],
            values["halo"],
            crops.valid_mask,
            _coordinate_channels(crops),
            timestep,
            global_embedding,
        )
        repair = values["repair"] * crops.valid_mask
        velocity_normalized = self.residual_normalization.normalize(
            velocity_physical
        ) * repair
        debug["predicted_velocity_physical"] = velocity_physical
        debug["current_residual"] = residual_t
        return velocity_normalized, condition, debug

    @staticmethod
    def _compose_full(
        crops: ReconstructionCropBatch,
        refined_crop: torch.Tensor,
        faulty_lidar_bev: torch.Tensor,
        reconstruction_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        refined_full = crops.paste(refined_crop)
        final = (
            (1.0 - reconstruction_mask) * faulty_lidar_bev
            + reconstruction_mask * refined_full
        )
        return refined_full, final

    def forward(
        self,
        clean_lidar_bev: torch.Tensor,
        coarse_lidar_bev: torch.Tensor,
        faulty_lidar_bev: torch.Tensor,
        radar_bev: torch.Tensor,
        reconstruction_mask: torch.Tensor,
        halo_mask: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        return_debug: bool = False,
        return_diagnostics: bool = True,
        shared_inputs: ReconstructionInputs | None = None,
        residual_regularization_weight: float | None = None,
    ) -> dict[str, torch.Tensor | ReconstructionCropBatch | dict]:
        self._validate_inputs(
            coarse_lidar_bev,
            faulty_lidar_bev,
            radar_bev,
            reconstruction_mask,
            halo_mask,
            clean_lidar_bev,
        )
        crops = self._extract(
            coarse_lidar_bev,
            faulty_lidar_bev,
            radar_bev,
            reconstruction_mask,
            halo_mask,
            clean_lidar_bev=clean_lidar_bev,
            shared_inputs=shared_inputs,
        )
        repair = crops.tensors["repair"] * crops.valid_mask
        residual_gt_physical = crops.tensors["residual_gt"] * repair
        residual_gt_normalized = self.residual_normalization.normalize(
            residual_gt_physical
        ) * repair
        trusted_global, global_embedding = self.transformer.global_context(
            faulty_lidar_bev, reconstruction_mask
        )
        flow_timesteps = self._sampling_timesteps(
            self.config.sampling_steps, coarse_lidar_bev.device
        )
        residual_x0_normalized = torch.zeros_like(residual_gt_normalized)
        flow_losses = []
        step_residual_regularization_losses = []
        training_intermediates = []
        local_condition = residual_x0_normalized
        transformer_debug = {}
        target_velocity = residual_gt_normalized
        velocity_pred = residual_x0_normalized
        for index, scalar_timestep in enumerate(flow_timesteps):
            timestep = scalar_timestep.expand(coarse_lidar_bev.shape[0])
            velocity_pred, local_condition, transformer_debug = (
                self._predict_velocity(
                    residual_x0_normalized,
                    crops,
                    timestep,
                    global_embedding,
                )
            )
            remaining_steps = len(flow_timesteps) - index
            target_velocity = (
                residual_gt_normalized - residual_x0_normalized.detach()
            ) / float(remaining_steps)
            target_velocity = target_velocity * repair
            flow_losses.append(
                self.diffusion_loss(velocity_pred, target_velocity, repair)
            )
            if self.config.residual_regularization_mode == "per_step_excess":
                step_residual_regularization_losses.append(
                    self._per_step_excess_regularization_loss(
                        velocity_pred,
                        target_velocity,
                        repair,
                    )
                )
            residual_x0_normalized = self._bounded_residual_update(
                residual_x0_normalized,
                velocity_pred,
                crops.tensors["coarse"],
                repair,
            )
            if return_debug:
                training_intermediates.append(
                    residual_x0_normalized.detach().clone()
                )
        predicted_residual_physical = self.residual_normalization.denormalize(
            residual_x0_normalized
        ) * repair
        refined_crop = (
            crops.tensors["coarse"] + predicted_residual_physical
        ).clamp(0.0, 1.0)
        refined_full, final = self._compose_full(
            crops, refined_crop, faulty_lidar_bev, reconstruction_mask
        )
        diffusion_loss = torch.stack(flow_losses).mean()
        exact_loss, exact_components = self.exact_loss(
            refined_crop,
            crops.tensors["clean"],
            repair,
            crops.tensors["coarse"].detach(),
            return_components=True,
        )
        coarse_exact_loss = (
            self.exact_loss(
                crops.tensors["coarse"].detach(),
                crops.tensors["clean"],
                repair,
                crops.tensors["coarse"].detach(),
            )
            if return_diagnostics
            else diffusion_loss.new_zeros(())
        )
        degradation_loss = self.degradation_loss(
            refined_crop, crops.tensors["coarse"], crops.tensors["clean"], repair
        )
        if self.config.residual_regularization_mode == "per_step_excess":
            residual_regularization_loss = torch.stack(
                step_residual_regularization_losses
            ).mean()
        else:
            residual_regularization_loss = self._residual_regularization_loss(
                predicted_residual_physical, repair
            )
        effective_residual_weight = (
            self.config.lambda_residual_regularization
            if residual_regularization_weight is None
            else float(residual_regularization_weight)
        )
        if effective_residual_weight < 0.0:
            raise ValueError("residual regularization weight must be non-negative")
        weighted_residual_regularization_loss = (
            effective_residual_weight * residual_regularization_loss
        )
        total_loss = (
            self.config.lambda_diffusion * diffusion_loss
            + self.config.lambda_exact_reconstruction * exact_loss
            + self.config.lambda_degradation * degradation_loss
            + weighted_residual_regularization_loss
        )
        diagnostics = {}
        if return_diagnostics:
            repair_cells = repair.sum(dim=(1, 2, 3))
            crop_cells = crops.valid_mask.sum(dim=(1, 2, 3)).clamp_min(1)
            diagnostics = {
                "average_reconstruction_mask_area": repair_cells.mean(),
                "average_crop_area": crop_cells.mean(),
                "average_crop_height": crops.crop_heights.float().mean(),
                "average_crop_width": crops.crop_widths.float().mean(),
                "repair_fraction_of_crop": (repair_cells / crop_cells).mean(),
                "halo_fraction_of_crop": (
                    crops.tensors["halo"].sum(dim=(1, 2, 3)) / crop_cells
                ).mean(),
                "refinement_steps": residual_gt_physical.new_tensor(
                    len(flow_timesteps)
                ),
                "residual_gt_physical_abs_mean": (
                    residual_gt_physical.abs().sum()
                    / (repair.sum() * residual_gt_physical.shape[1]).clamp_min(1)
                ),
                "residual_gt_normalized_abs_mean": (
                    residual_gt_normalized.abs().sum()
                    / (repair.sum() * residual_gt_normalized.shape[1]).clamp_min(1)
                ),
                "predicted_residual_physical_abs_mean": (
                    predicted_residual_physical.abs().sum()
                    / (
                        repair.sum() * predicted_residual_physical.shape[1]
                        + self.config.denominator_epsilon
                    )
                ),
                "predicted_residual_normalized_abs_mean": (
                    residual_x0_normalized.abs().sum()
                    / (
                        repair.sum() * residual_x0_normalized.shape[1]
                        + self.config.denominator_epsilon
                    )
                ),
            }
            physical_gt_magnitude = diagnostics["residual_gt_physical_abs_mean"]
            diagnostics["predicted_to_gt_physical_residual_ratio"] = (
                diagnostics["predicted_residual_physical_abs_mean"]
                / (physical_gt_magnitude + self.config.denominator_epsilon)
            )
            selected = repair.expand_as(residual_gt_physical) > 0.5
            for channel in range(self.config.lidar_channels):
                channel_values = residual_gt_physical[:, channel : channel + 1][
                    selected[:, channel : channel + 1]
                ]
                diagnostics[f"residual_gt_physical_std_channel_{channel}"] = (
                    channel_values.float().std(unbiased=False)
                    if channel_values.numel()
                    else residual_gt_physical.new_zeros(())
                )
        output: dict = {
            "loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "exact_reconstruction_loss": exact_loss,
            "exact_occupancy_loss": exact_components["occupancy_loss"],
            "occupancy_operation_loss": exact_components["occupancy_loss"],
            "exact_continuous_loss": exact_components["continuous_loss"],
            **{
                key: value
                for key, value in exact_components.items()
                if key.startswith("occupancy_") or key.startswith("num_")
            },
            "coarse_exact_reconstruction_loss": coarse_exact_loss,
            "degradation_loss": degradation_loss,
            "residual_regularization_loss": residual_regularization_loss,
            "weighted_residual_regularization_loss": (
                weighted_residual_regularization_loss
            ),
            "residual_regularization_weight": total_loss.new_tensor(
                effective_residual_weight
            ),
            "coarse_lidar_bev": coarse_lidar_bev,
            "final_lidar_bev": final,
            "reconstruction_mask": reconstruction_mask,
            "statistics": diagnostics,
        }
        if return_debug:
            output["debug"] = {
                "crops": crops,
                "trusted_global_faulty_lidar": trusted_global,
                "residual_gt_physical": residual_gt_physical,
                "residual_gt_normalized": residual_gt_normalized,
                "flow_timesteps": flow_timesteps,
                "current_residual": residual_x0_normalized,
                "target_velocity": target_velocity,
                "predicted_velocity": velocity_pred,
                "training_intermediate_residuals": training_intermediates,
                "residual_x0_normalized": residual_x0_normalized,
                "predicted_residual_physical": predicted_residual_physical,
                "local_condition": local_condition,
                **transformer_debug,
                "refined_crop": refined_crop,
                "refined_lidar_bev": refined_full,
            }
        return output

    def _sampling_timesteps(
        self, sampling_steps: int, device: torch.device
    ) -> torch.Tensor:
        if sampling_steps not in SUPPORTED_SAMPLING_STEPS:
            raise ValueError(
                "sampling_steps must be one of 1, 3, 5, 10, 25, or 50"
            )
        return torch.linspace(
            0,
            self.config.training_timesteps,
            sampling_steps + 1,
            device=device,
        )[:-1].round().long().clamp_max(self.config.training_timesteps - 1)

    @torch.no_grad()
    def sample(
        self,
        coarse_lidar_bev: torch.Tensor,
        faulty_lidar_bev: torch.Tensor,
        radar_bev: torch.Tensor,
        reconstruction_mask: torch.Tensor,
        halo_mask: torch.Tensor,
        *,
        sampling_steps: int | None = None,
        generator: torch.Generator | None = None,
        return_debug: bool = False,
        shared_inputs: ReconstructionInputs | None = None,
    ) -> dict[str, torch.Tensor | ReconstructionCropBatch | list]:
        """Refine a coarse BEV without requiring clean LiDAR supervision."""

        self._validate_inputs(
            coarse_lidar_bev,
            faulty_lidar_bev,
            radar_bev,
            reconstruction_mask,
            halo_mask,
        )
        crops = self._extract(
            coarse_lidar_bev,
            faulty_lidar_bev,
            radar_bev,
            reconstruction_mask,
            halo_mask,
            shared_inputs=shared_inputs,
        )
        repair = crops.tensors["repair"] * crops.valid_mask
        if int(repair.count_nonzero()) == 0:
            zeros = torch.zeros_like(coarse_lidar_bev)
            return {
                "coarse_lidar_bev": coarse_lidar_bev,
                "predicted_residual": zeros,
                "final_lidar_bev": faulty_lidar_bev.clone(),
                "reconstruction_mask": reconstruction_mask,
            }
        timesteps = self._sampling_timesteps(
            sampling_steps or self.config.sampling_steps,
            coarse_lidar_bev.device,
        )
        start_timestep = timesteps[0].expand(coarse_lidar_bev.shape[0])
        residual_t = torch.zeros(
            (
                coarse_lidar_bev.shape[0],
                self.config.lidar_channels,
                *repair.shape[-2:],
            ),
            device=coarse_lidar_bev.device,
            dtype=coarse_lidar_bev.dtype,
        )
        initial_residual = residual_t.detach().clone()
        _trusted_global, global_embedding = self.transformer.global_context(
            faulty_lidar_bev, reconstruction_mask
        )
        intermediates = []
        residual_x0_normalized = torch.zeros_like(residual_t)
        for scalar_timestep in timesteps:
            timestep = scalar_timestep.expand(coarse_lidar_bev.shape[0])
            velocity_pred, _local_condition, transformer_debug = self._predict_velocity(
                residual_t, crops, timestep, global_embedding
            )
            residual_t = self._bounded_residual_update(
                residual_t,
                velocity_pred,
                crops.tensors["coarse"],
                repair,
            )
            if return_debug:
                intermediates.append(residual_t.detach().clone())
        residual_x0_normalized = residual_t
        predicted_residual_physical = self.residual_normalization.denormalize(
            residual_x0_normalized * repair
        ) * repair
        refined_crop = (
            crops.tensors["coarse"] + predicted_residual_physical
        ).clamp(0.0, 1.0)
        refined_full, final = self._compose_full(
            crops, refined_crop, faulty_lidar_bev, reconstruction_mask
        )
        output: dict = {
            "coarse_lidar_bev": coarse_lidar_bev,
            "predicted_residual": (
                crops.paste(predicted_residual_physical) * reconstruction_mask
            ),
            "refined_lidar_bev": refined_full,
            "final_lidar_bev": final,
            "reconstruction_mask": reconstruction_mask,
            "crop_boxes": crops.boxes,
        }
        if return_debug:
            output["crop_batch"] = crops
            output["intermediate_residuals"] = intermediates
            output["sampling_timesteps"] = timesteps.detach().clone()
            output["sampling_start_progress"] = torch.zeros_like(start_timestep)
            output["initial_residual"] = initial_residual
            output["initial_lidar_bev"] = (
                coarse_lidar_bev
                + crops.paste(initial_residual) * reconstruction_mask
            ).detach()
            output["residual_x0_normalized"] = residual_x0_normalized.detach().clone()
            output.update(transformer_debug)
        return output
