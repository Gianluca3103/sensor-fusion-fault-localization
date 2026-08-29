"""Local coarse-anchored residual-flow refiner for LiDAR reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F

from ..encoders import _group_count
from ..coarse_reconstruction.coarse_loss import (
    CoarseLossConfig,
    MaskedBEVReconstructionLoss,
    ObservabilityWeightingConfig,
)
from ..reconstruction_inputs import ReconstructionInputs
from ..reconstruction_crop import ReconstructionCropBatch, ReconstructionCropExtractor
from .diffusion_process import (
    BEVChannelNormalization,
    MaskedFlowMSELoss,
    ResidualChannelNormalization,
    residual_target,
)
from .basic_diffusion_unet import BasicDiffusionUNet, SinusoidalTimeEmbedding


SUPPORTED_SAMPLING_STEPS = frozenset({1, 3, 5, 6, 10, 25, 50})
FINE_DIFFUSION_TRANSFORMER_ARCHITECTURE_VERSION = 15
FINE_DIFFUSION_TRANSFORMER_LEGACY_ARCHITECTURE_VERSION = 11
FINE_DIFFUSION_UNET_ARCHITECTURE_VERSION = 12
FINE_DIFFUSION_UNET_POINTPILLARS_ARCHITECTURE_VERSION = 13
FINE_DIFFUSION_UNET_FAIR_ARCHITECTURE_VERSION = 14


@dataclass(frozen=True)
class FineDiffusionConfig:
    """Configuration for the coarse-anchored local residual-flow refiner."""

    enabled: bool = True
    bypass_coarse_reconstruction: bool = False
    fine_backbone: str = "transformer"
    lidar_channels: int = 3
    radar_channels: int = 4
    use_pointpillars_conditioning: bool = False
    lidar_pillar_channels: int = 64
    radar_pillar_channels: int = 64
    transformer_spatial_input_mode: str = "zero_residual"
    hidden_dim: int = 64
    attention_dim: int | None = None
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
    soft_iou_epsilon: float = 1.0e-6
    coarse_positive_occupancy_weight: float = 1.1
    coarse_min_empty_observability_weight: float = 0.1
    correction_group_weight: float = 1.0
    preservation_group_weight: float = 0.5
    operation_add_weight: float = 0.21
    operation_remove_weight: float = 0.395
    operation_preserve_occupied_weight: float = 0.395
    operation_preserve_empty_weight: float = 0.21
    dropout: float = 0.0
    denominator_epsilon: float = 1.0e-8
    minimum_residual_std: float = 1.0e-4
    fine_unet_base_channels: int = 64
    fine_unet_channel_multipliers: tuple[int, ...] = (1, 2, 4, 8)
    fine_unet_num_downsamples: int = 3
    fine_unet_resblocks_per_level: int = 2
    fine_unet_include_coarse_input: bool = True
    fine_unet_use_global_faulty_context: bool = False
    fine_min_context_height: int = 1
    fine_min_context_width: int = 1

    def validate(self) -> None:
        if not self.enabled:
            raise ValueError("Fine diffusion must be enabled")
        if self.fine_backbone not in ("transformer", "unet"):
            raise ValueError("fine_backbone must be transformer or unet")
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
        attention_dim = (
            self.hidden_dim if self.attention_dim is None else self.attention_dim
        )
        if attention_dim < 1:
            raise ValueError("attention_dim must be positive when configured")
        if attention_dim % self.num_heads:
            raise ValueError("attention_dim must be divisible by num_heads")
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
        if self.fine_unet_base_channels < 1:
            raise ValueError("fine_unet_base_channels must be positive")
        if self.fine_unet_num_downsamples != 3:
            raise ValueError("Basic Diffusion U-Net requires exactly 3 downsamples")
        if len(self.fine_unet_channel_multipliers) != 4 or any(
            multiplier < 1 for multiplier in self.fine_unet_channel_multipliers
        ):
            raise ValueError(
                "fine_unet_channel_multipliers must contain 4 positive values"
            )
        if self.fine_unet_resblocks_per_level < 1:
            raise ValueError("fine_unet_resblocks_per_level must be positive")
        if not isinstance(self.fine_unet_include_coarse_input, bool):
            raise ValueError("fine_unet_include_coarse_input must be boolean")
        if not isinstance(self.fine_unet_use_global_faulty_context, bool):
            raise ValueError(
                "fine_unet_use_global_faulty_context must be boolean"
            )
        if self.fine_min_context_height < 1 or self.fine_min_context_width < 1:
            raise ValueError("Fine minimum context dimensions must be positive")
        if self.fine_backbone == "unet" and (
            self.fine_min_context_height < 80
            or self.fine_min_context_width < 80
        ):
            raise ValueError(
                "Basic Diffusion U-Net requires at least 80x80 local context"
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
                "sampling_steps must be one of 1, 3, 5, 6, 10, 25, or 50"
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
            "coarse_existing",
            "soft_iou",
        ):
            raise ValueError(
                "occupancy_loss_mode must be standard_bce, operation_balanced, "
                "weighted_operation, coarse_existing, or soft_iou"
            )
        if not 0.0 < self.occupancy_threshold < 1.0:
            raise ValueError("occupancy_threshold must be strictly between 0 and 1")
        if self.soft_iou_epsilon <= 0.0:
            raise ValueError("soft_iou_epsilon must be positive")
        if self.coarse_positive_occupancy_weight < 1.0:
            raise ValueError(
                "coarse_positive_occupancy_weight must be at least 1"
            )
        if not 0.0 <= self.coarse_min_empty_observability_weight <= 1.0:
            raise ValueError(
                "coarse_min_empty_observability_weight must be in [0,1]"
            )
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


@dataclass(frozen=True)
class WindowLayout:
    """Reusable window geometry and validity for one padded crop shape."""

    valid_windows: torch.Tensor
    metadata: tuple[int, int, int, int, int, int, int, int]


@dataclass(frozen=True)
class CrossAttentionCache:
    """Step-invariant radar windows and their projected keys/values."""

    query_layout: WindowLayout
    key_layout: WindowLayout
    active_windows: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class TransformerInferenceCache:
    """All spatial conditioning that is invariant across refinement steps."""

    coordinates: torch.Tensor
    auxiliary_condition: torch.Tensor
    lidar_pillar_condition: torch.Tensor
    radar_attention: torch.Tensor
    radar_valid: torch.Tensor
    self_layouts: tuple[WindowLayout, ...]
    radar_caches: tuple[CrossAttentionCache, ...]


def _window_layout(
    valid: torch.Tensor,
    window_size: int,
    shift: int,
) -> WindowLayout:
    batch, _one, height, width = valid.shape
    pad_top = pad_left = shift
    padded_height = math.ceil((height + shift) / window_size) * window_size
    padded_width = math.ceil((width + shift) / window_size) * window_size
    pad_bottom = padded_height - height - pad_top
    pad_right = padded_width - width - pad_left
    valid = F.pad(valid, (pad_left, pad_right, pad_top, pad_bottom))
    rows, columns = padded_height // window_size, padded_width // window_size
    valid_windows = (
        valid.reshape(batch, 1, rows, window_size, columns, window_size)
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(batch * rows * columns, window_size * window_size)
        > 0.5
    )
    metadata = (
        batch,
        height,
        width,
        padded_height,
        padded_width,
        rows,
        columns,
        shift,
    )
    return WindowLayout(valid_windows=valid_windows, metadata=metadata)


def _partition_tensor(
    tensor: torch.Tensor,
    layout: WindowLayout,
) -> torch.Tensor:
    batch, channels, height, width = tensor.shape
    (
        expected_batch,
        expected_height,
        expected_width,
        padded_height,
        padded_width,
        rows,
        columns,
        shift,
    ) = layout.metadata
    if (batch, height, width) != (
        expected_batch,
        expected_height,
        expected_width,
    ):
        raise ValueError("tensor does not match cached window layout")
    tensor = F.pad(
        tensor,
        (
            shift,
            padded_width - width - shift,
            shift,
            padded_height - height - shift,
        ),
    )
    window_height = padded_height // rows
    window_width = padded_width // columns
    return (
        tensor.reshape(
            batch,
            channels,
            rows,
            window_height,
            columns,
            window_width,
        )
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(batch * rows * columns, -1, channels)
    )


def _partition_windows(
    tensor: torch.Tensor,
    valid: torch.Tensor,
    window_size: int,
    shift: int,
):
    layout = _window_layout(valid, window_size, shift)
    return _partition_tensor(tensor, layout), layout.valid_windows, layout.metadata


def _reverse_windows(windows: torch.Tensor, metadata) -> torch.Tensor:
    (
        batch,
        height,
        width,
        padded_height,
        padded_width,
        rows,
        columns,
        shift,
    ) = metadata
    channels = windows.shape[-1]
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
        attention_dim: int | None = None,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.attention_dim = int(attention_dim or hidden_dim)
        self.head_dim = self.attention_dim // self.num_heads
        self.dropout = float(dropout)
        self.uses_projected_attention = self.attention_dim != self.hidden_dim
        if self.uses_projected_attention:
            source_dim = int(key_value_dim or hidden_dim)
            self.attention = None
            self.query_projection = nn.Linear(hidden_dim, self.attention_dim)
            self.key_projection = nn.Linear(source_dim, self.attention_dim)
            self.value_projection = nn.Linear(source_dim, self.attention_dim)
            self.output_projection = nn.Linear(self.attention_dim, hidden_dim)
        else:
            self.attention = nn.MultiheadAttention(
                hidden_dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
                kdim=key_value_dim,
                vdim=key_value_dim,
            )
            self.query_projection = None
            self.key_projection = None
            self.value_projection = None
            self.output_projection = None
        self._fused_self_weight: torch.Tensor | None = None
        self._fused_self_bias: torch.Tensor | None = None

    def train(self, mode: bool = True):
        if mode:
            self._fused_self_weight = None
            self._fused_self_bias = None
        return super().train(mode)

    def prepare_for_inference(self) -> None:
        """Cache a fused QKV matrix without changing checkpoint parameters."""

        if self.training or not self.uses_projected_attention:
            return
        if self._fused_self_weight is None:
            self._fused_self_weight = torch.cat(
                (
                    self.query_projection.weight,
                    self.key_projection.weight,
                    self.value_projection.weight,
                ),
                dim=0,
            ).detach()
            self._fused_self_bias = torch.cat(
                (
                    self.query_projection.bias,
                    self.key_projection.bias,
                    self.value_projection.bias,
                ),
                dim=0,
            ).detach()

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _channels = tensor.shape
        return tensor.reshape(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def _output_projection(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.uses_projected_attention:
            return self.output_projection(tensor)
        return self.attention.out_proj(tensor)

    def _project_self_qkv(
        self, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.uses_projected_attention:
            self.prepare_for_inference()
            if self._fused_self_weight is not None:
                projected = F.linear(
                    tensor,
                    self._fused_self_weight,
                    self._fused_self_bias,
                )
                return projected.chunk(3, dim=-1)
            return (
                self.query_projection(tensor),
                self.key_projection(tensor),
                self.value_projection(tensor),
            )
        if self.attention.in_proj_weight is not None:
            return F.linear(
                tensor,
                self.attention.in_proj_weight,
                self.attention.in_proj_bias,
            ).chunk(3, dim=-1)
        bias = self.attention.in_proj_bias
        query_bias, key_bias, value_bias = (
            bias.chunk(3) if bias is not None else (None, None, None)
        )
        return (
            F.linear(tensor, self.attention.q_proj_weight, query_bias),
            F.linear(tensor, self.attention.k_proj_weight, key_bias),
            F.linear(tensor, self.attention.v_proj_weight, value_bias),
        )

    def _project_query(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.uses_projected_attention:
            return self.query_projection(tensor)
        bias = self.attention.in_proj_bias
        query_bias = bias[: self.attention.embed_dim] if bias is not None else None
        weight = (
            self.attention.in_proj_weight[: self.attention.embed_dim]
            if self.attention.in_proj_weight is not None
            else self.attention.q_proj_weight
        )
        return F.linear(tensor, weight, query_bias)

    def _project_key_value(
        self, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.uses_projected_attention:
            return self.key_projection(tensor), self.value_projection(tensor)
        embed_dim = self.attention.embed_dim
        bias = self.attention.in_proj_bias
        key_bias = bias[embed_dim : 2 * embed_dim] if bias is not None else None
        value_bias = bias[2 * embed_dim :] if bias is not None else None
        if self.attention.in_proj_weight is not None:
            key_weight = self.attention.in_proj_weight[embed_dim : 2 * embed_dim]
            value_weight = self.attention.in_proj_weight[2 * embed_dim :]
        else:
            key_weight = self.attention.k_proj_weight
            value_weight = self.attention.v_proj_weight
        return (
            F.linear(tensor, key_weight, key_bias),
            F.linear(tensor, value_weight, value_bias),
        )

    def _attend_projected(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        key_valid: torch.Tensor,
    ) -> torch.Tensor:
        query_length = queries.shape[1]
        attended = F.scaled_dot_product_attention(
            self._reshape_heads(queries),
            self._reshape_heads(keys),
            self._reshape_heads(values),
            attn_mask=key_valid[:, None, None, :],
            dropout_p=(self.dropout if self.training else 0.0),
        ).transpose(1, 2).reshape(
            queries.shape[0], query_length, self.attention_dim
        )
        return self._output_projection(attended)

    def forward_self(
        self,
        tensor: torch.Tensor,
        layout: WindowLayout,
    ) -> torch.Tensor:
        """Inference self-attention with one partition and one fused QKV map."""

        windows = _partition_tensor(tensor, layout)
        valid_windows = layout.valid_windows
        active = valid_windows.any(dim=1)
        output = torch.zeros_like(windows)
        active_windows = windows[active]
        queries, keys, values = self._project_self_qkv(active_windows)
        attended = self._attend_projected(
            queries,
            keys,
            values,
            valid_windows[active],
        )
        output[active] = attended.to(output.dtype) * valid_windows[
            active, :, None
        ]
        return _reverse_windows(output, layout.metadata)

    def prepare_cross_attention_cache_from_windows(
        self,
        key_windows: torch.Tensor,
        query_layout: WindowLayout,
        key_layout: WindowLayout,
    ) -> CrossAttentionCache:
        """Project K/V from radar windows shared by blocks with one shift."""

        active = query_layout.valid_windows.any(dim=1) & (
            key_layout.valid_windows.any(dim=1)
        )
        keys, values = self._project_key_value(key_windows[active])
        return CrossAttentionCache(
            query_layout=query_layout,
            key_layout=key_layout,
            active_windows=active,
            keys=keys,
            values=values,
        )

    def prepare_cross_attention_cache(
        self,
        key_value: torch.Tensor,
        query_valid: torch.Tensor,
        key_value_valid: torch.Tensor,
        *,
        shift: int,
    ) -> CrossAttentionCache:
        query_layout = _window_layout(query_valid, self.window_size, shift)
        key_layout = _window_layout(key_value_valid, self.window_size, shift)
        return self.prepare_cross_attention_cache_from_windows(
            _partition_tensor(key_value, key_layout),
            query_layout,
            key_layout,
        )

    def forward_cross(
        self,
        query: torch.Tensor,
        cache: CrossAttentionCache,
    ) -> torch.Tensor:
        """Cross-attend to step-invariant, preprojected radar keys/values."""

        query_windows = _partition_tensor(query, cache.query_layout)
        output = torch.zeros_like(query_windows)
        active = cache.active_windows
        queries = self._project_query(query_windows[active])
        attended = self._attend_projected(
            queries,
            cache.keys,
            cache.values,
            cache.key_layout.valid_windows[active],
        )
        output[active] = attended.to(output.dtype) * (
            cache.query_layout.valid_windows[active, :, None]
        )
        return _reverse_windows(output, cache.query_layout.metadata)

    def _projected_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        key_valid: torch.Tensor,
    ) -> torch.Tensor:
        keys, values = self._project_key_value(key)
        return self._attend_projected(
            self._project_query(query),
            keys,
            values,
            key_valid,
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
            if self.uses_projected_attention:
                attended = self._projected_attention(
                    query_windows[active],
                    key_windows[active],
                    key_valid[active],
                )
            else:
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
        attention_dim: int | None = None,
    ):
        super().__init__()
        self.shift = window_size // 2 if shifted else 0
        self.self_norm = AdaptiveLayerNorm2d(hidden_dim, condition_dim)
        self.radar_cross_norm = AdaptiveLayerNorm2d(hidden_dim, condition_dim)
        self.ffn_norm = AdaptiveLayerNorm2d(hidden_dim, condition_dim)
        self.self_attention = WindowAttention2d(
            hidden_dim,
            num_heads,
            window_size,
            dropout,
            attention_dim=attention_dim,
        )
        self.radar_cross_attention = WindowAttention2d(
            hidden_dim,
            num_heads,
            window_size,
            dropout,
            key_value_dim=radar_channels,
            attention_dim=attention_dim,
        )
        self.ffn = ConvolutionalFFN(hidden_dim, dropout)

    def forward(
        self,
        tensor: torch.Tensor,
        raw_radar: torch.Tensor,
        condition_vector: torch.Tensor,
        valid: torch.Tensor,
        radar_valid: torch.Tensor,
        *,
        self_layout: WindowLayout | None = None,
        radar_cache: CrossAttentionCache | None = None,
    ) -> torch.Tensor:
        normalized = self.self_norm(tensor, condition_vector)
        if self_layout is None:
            self_update = self.self_attention(
                normalized,
                normalized,
                valid,
                shift=self.shift,
            )
        else:
            self_update = self.self_attention.forward_self(
                normalized,
                self_layout,
            )
        tensor = tensor + self_update
        radar_query = self.radar_cross_norm(tensor, condition_vector)
        if radar_cache is None:
            radar_update = self.radar_cross_attention(
                radar_query,
                raw_radar,
                valid,
                shift=self.shift,
                key_value_valid=radar_valid,
            )
        else:
            radar_update = self.radar_cross_attention.forward_cross(
                radar_query,
                radar_cache,
            )
        tensor = tensor + radar_update
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
                config.attention_dim,
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

    def prepare_inference_cache(
        self,
        trusted_faulty: torch.Tensor,
        radar: torch.Tensor,
        lidar_pillars: torch.Tensor | None,
        radar_pillars: torch.Tensor | None,
        reconstruction_mask: torch.Tensor,
        halo_mask: torch.Tensor,
        valid: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> TransformerInferenceCache:
        """Precompute conditioning that is unchanged by residual-flow steps."""

        for block in self.blocks:
            block.self_attention.prepare_for_inference()

        auxiliary_condition_input = torch.cat(
            (trusted_faulty, reconstruction_mask, halo_mask, coordinates),
            dim=1,
        )
        auxiliary_condition = self.auxiliary_condition_encoder(
            auxiliary_condition_input,
            valid,
        )
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
        radar_valid = reconstruction_mask * valid
        layouts_by_shift: dict[int, WindowLayout] = {}
        radar_layouts_by_shift: dict[int, WindowLayout] = {}
        radar_windows_by_shift: dict[int, torch.Tensor] = {}
        for block in self.blocks:
            shift = block.shift
            if shift not in layouts_by_shift:
                layouts_by_shift[shift] = _window_layout(
                    valid,
                    block.self_attention.window_size,
                    shift,
                )
                radar_layouts_by_shift[shift] = _window_layout(
                    radar_valid,
                    block.radar_cross_attention.window_size,
                    shift,
                )
                radar_windows_by_shift[shift] = _partition_tensor(
                    radar_attention,
                    radar_layouts_by_shift[shift],
                )
        self_layouts = tuple(
            layouts_by_shift[block.shift] for block in self.blocks
        )
        radar_caches = tuple(
            block.radar_cross_attention.prepare_cross_attention_cache_from_windows(
                radar_windows_by_shift[block.shift],
                layouts_by_shift[block.shift],
                radar_layouts_by_shift[block.shift],
            )
            for block in self.blocks
        )
        return TransformerInferenceCache(
            coordinates=coordinates,
            auxiliary_condition=auxiliary_condition,
            lidar_pillar_condition=lidar_pillar_condition,
            radar_attention=radar_attention,
            radar_valid=radar_valid,
            self_layouts=self_layouts,
            radar_caches=radar_caches,
        )

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
        inference_cache: TransformerInferenceCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if inference_cache is None:
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
        else:
            coordinates = inference_cache.coordinates
            auxiliary_condition = inference_cache.auxiliary_condition
        spatial_stem_input = torch.cat((spatial_state, coordinates), dim=1)
        if inference_cache is not None:
            lidar_pillar_condition = inference_cache.lidar_pillar_condition
            radar_attention = inference_cache.radar_attention
        elif self.config.use_pointpillars_conditioning:
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
        radar_valid = (
            inference_cache.radar_valid
            if inference_cache is not None
            else reconstruction_mask * valid
        )
        for index, block in enumerate(self.blocks):
            tensor = block(
                tensor,
                radar_attention,
                condition_vector,
                valid,
                radar_valid,
                self_layout=(
                    inference_cache.self_layouts[index]
                    if inference_cache is not None
                    else None
                ),
                radar_cache=(
                    inference_cache.radar_caches[index]
                    if inference_cache is not None
                    else None
                ),
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
) -> dict[str, object]:
    common: dict[str, object] = {
        "process": "coarse_anchored_residual_flow",
        "initial_state": "frozen_coarse_reconstruction",
        "internal_state": "normalized_current_residual",
        "fine_backbone": config.fine_backbone,
        "coarse_role": "frozen_reconstruction_baseline_only",
        "refinement_feedback": "updated_normalized_residual_each_step",
    }
    if config.fine_backbone == "unet":
        fair_architecture = (
            not config.fine_unet_include_coarse_input
            or config.fine_unet_use_global_faulty_context
        )
        sensor_channels = (
            config.lidar_pillar_channels + config.radar_pillar_channels
            if config.use_pointpillars_conditioning
            else config.radar_channels
        )
        return {
            **common,
            "version": (
                FINE_DIFFUSION_UNET_FAIR_ARCHITECTURE_VERSION
                if fair_architecture
                else (
                    FINE_DIFFUSION_UNET_POINTPILLARS_ARCHITECTURE_VERSION
                    if config.use_pointpillars_conditioning
                    else FINE_DIFFUSION_UNET_ARCHITECTURE_VERSION
                )
            ),
            "backbone_name": (
                "basic_diffusion_unet_pointpillars"
                if config.use_pointpillars_conditioning
                else "basic_diffusion_unet"
            ),
            "input_channels": (
                config.lidar_channels
                + (
                    config.lidar_channels
                    if config.fine_unet_include_coarse_input
                    else 0
                )
                + sensor_channels
                + 3
            ),
            "channel_hierarchy": [
                config.fine_unet_base_channels * multiplier
                for multiplier in config.fine_unet_channel_multipliers
            ],
            "num_downsamples": config.fine_unet_num_downsamples,
            "deepest_stride": 2 ** config.fine_unet_num_downsamples,
            "resblocks_per_level": config.fine_unet_resblocks_per_level,
            "minimum_context": [
                config.fine_min_context_height,
                config.fine_min_context_width,
            ],
            "radar_conditioning": (
                "radar_pointpillars_spatial_concatenation"
                if config.use_pointpillars_conditioning
                else "raw_spatial_concatenation"
            ),
            "lidar_conditioning": (
                "lidar_pointpillars_spatial_concatenation"
                if config.use_pointpillars_conditioning
                else "coarse_reconstruction_only"
            ),
            "pointpillars_conditioning": config.use_pointpillars_conditioning,
            "lidar_pillar_channels": config.lidar_pillar_channels,
            "radar_pillar_channels": config.radar_pillar_channels,
            "coarse_visible_to_backbone": config.fine_unet_include_coarse_input,
            "global_faulty_context": (
                config.fine_unet_use_global_faulty_context
            ),
        }
    return {
        **common,
        "version": (
            FINE_DIFFUSION_TRANSFORMER_ARCHITECTURE_VERSION
            if config.attention_dim is not None
            and config.attention_dim != config.hidden_dim
            else FINE_DIFFUSION_TRANSFORMER_LEGACY_ARCHITECTURE_VERSION
        ),
        "hidden_dim": config.hidden_dim,
        "attention_dim": config.attention_dim or config.hidden_dim,
        "num_heads": config.num_heads,
        "num_transformer_blocks": config.num_transformer_blocks,
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
    state = checkpoint.get("diffusion_state_dict", {})
    metadata = checkpoint.get("fine_diffusion_architecture")
    architecture_version = (
        metadata.get("version") if isinstance(metadata, Mapping) else None
    )
    checkpoint_backbone = (
        metadata.get("fine_backbone", "transformer")
        if isinstance(metadata, Mapping)
        else "transformer"
    )
    if checkpoint_backbone != config.fine_backbone:
        raise ValueError(
            "Fine Diffusion backbone does not match its checkpoint. Start a "
            "fresh architecture-ablation run."
        )
    if config.fine_backbone == "unet":
        fair_architecture = (
            not config.fine_unet_include_coarse_input
            or config.fine_unet_use_global_faulty_context
        )
        expected_version = (
            FINE_DIFFUSION_UNET_FAIR_ARCHITECTURE_VERSION
            if fair_architecture
            else (
                FINE_DIFFUSION_UNET_POINTPILLARS_ARCHITECTURE_VERSION
                if config.use_pointpillars_conditioning
                else FINE_DIFFUSION_UNET_ARCHITECTURE_VERSION
            )
        )
        if architecture_version != expected_version:
            raise ValueError("Unsupported Basic Diffusion U-Net checkpoint version")
        sensor_channels = (
            config.lidar_pillar_channels + config.radar_pillar_channels
            if config.use_pointpillars_conditioning
            else config.radar_channels
        )
        expected = (
            config.lidar_channels
            + (
                config.lidar_channels
                if config.fine_unet_include_coarse_input
                else 0
            )
            + sensor_channels
            + 3
        )
        weight = state.get("unet.input_projection.weight")
        actual = int(weight.shape[1]) if torch.is_tensor(weight) else None
        if actual != expected:
            raise ValueError(
                "Basic Diffusion U-Net input mismatch: checkpoint has "
                f"{actual} channels but this model requires {expected}."
            )
        return

    expected = config.lidar_channels + 4
    if architecture_version == 10:
        expected_mode = "current_lidar"
        stem_name = "transformer.current_lidar_stem.weight"
    elif architecture_version in (
        FINE_DIFFUSION_TRANSFORMER_LEGACY_ARCHITECTURE_VERSION,
        FINE_DIFFUSION_TRANSFORMER_ARCHITECTURE_VERSION,
    ):
        expected_mode = "zero_residual"
        stem_name = "transformer.residual_stem.weight"
    else:
        raise ValueError(
            "Fine Diffusion checkpoint is neither supported v10 current-LiDAR, "
            "v11 legacy zero-residual, nor v15 projected-attention refinement."
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
    projected_attention = (
        config.attention_dim is not None
        and config.attention_dim != config.hidden_dim
    )
    if projected_attention != (
        architecture_version == FINE_DIFFUSION_TRANSFORMER_ARCHITECTURE_VERSION
    ):
        raise ValueError(
            "Fine Diffusion attention projection does not match its checkpoint. "
            "Start a fresh architecture-ablation run."
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
        soft_iou_epsilon: float = 1.0e-6,
        correction_group_weight: float = 1.0,
        preservation_group_weight: float = 0.5,
        operation_add_weight: float = 0.21,
        operation_remove_weight: float = 0.395,
        operation_preserve_occupied_weight: float = 0.395,
        operation_preserve_empty_weight: float = 0.21,
        coarse_positive_occupancy_weight: float = 1.1,
        coarse_min_empty_observability_weight: float = 0.1,
    ):
        super().__init__()
        self.epsilon = float(epsilon)
        self.occupancy_loss_mode = str(occupancy_loss_mode)
        self.occupancy_threshold = float(occupancy_threshold)
        self.soft_iou_epsilon = float(soft_iou_epsilon)
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
        self.coarse_existing_loss = MaskedBEVReconstructionLoss(
            CoarseLossConfig(
                epsilon=self.epsilon,
                positive_occupancy_weight=float(
                    coarse_positive_occupancy_weight
                ),
                observability_weighting=ObservabilityWeightingConfig(
                    enabled=True,
                    min_empty_weight=float(
                        coarse_min_empty_observability_weight
                    ),
                ),
            )
        )

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
        observability_confidence: torch.Tensor | None = None,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.occupancy_loss_mode == "coarse_existing":
            if observability_confidence is None:
                raise ValueError(
                    "coarse_existing Fine Diffusion loss requires "
                    "observability_confidence"
                )
            probability = refined[:, 0:1].float().clamp(1.0e-6, 1.0 - 1.0e-6)
            replacement_raw = torch.cat(
                (torch.logit(probability), refined[:, 1:].float()), dim=1
            )
            losses = self.coarse_existing_loss(
                {
                    "replacement_raw": replacement_raw,
                    "reconstruction_mask": reconstruction_mask.float(),
                },
                clean.float(),
                observability_confidence.float(),
            )
            zero = losses["loss"].new_zeros(())
            components = {
                "occupancy_loss": losses["loss_occupancy"],
                "continuous_loss": (
                    losses["loss_density"] + losses["loss_height"]
                ),
                "coarse_occupancy_bce_loss": losses["loss_occupancy_bce"],
                "coarse_occupancy_dice_loss": losses["loss_occupancy_dice"],
                "coarse_density_loss": losses["loss_density"],
                "coarse_height_loss": losses["loss_height"],
                **{f"occupancy_{name}_loss": zero for name in self.GROUPS},
                **{f"num_{name}": zero.detach() for name in self.GROUPS},
            }
            return (
                (losses["loss"], components)
                if return_components
                else losses["loss"]
            )
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
                    "soft_iou": zero,
                    "continuous_loss": zero,
                    **{f"occupancy_{name}_loss": zero for name in self.GROUPS},
                    **{
                        f"num_{name}": zero.detach()
                        for name in self.GROUPS
                    },
                }
                return (zero, components) if return_components else zero
            per_cell_bce = None
            group_losses = {
                name: refined_occupancy.new_zeros(()) for name in self.GROUPS
            }
            group_counts = {
                name: refined_occupancy.new_zeros(()) for name in self.GROUPS
            }
            soft_iou = refined_occupancy.new_zeros(())
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
            if self.occupancy_loss_mode == "soft_iou":
                # Per-sample reduction gives each valid crop equal influence,
                # independent of its reconstruction-mask area.
                probability = refined_occupancy_map.clamp(0.0, 1.0)
                dimensions = tuple(range(1, probability.ndim))
                intersection = (
                    mask_float * probability * occupied
                ).sum(dim=dimensions)
                union = (
                    mask_float
                    * (probability + occupied - probability * occupied)
                ).sum(dim=dimensions)
                valid_samples = mask_float.sum(dim=dimensions) > 0
                per_sample_soft_iou = (
                    intersection + self.soft_iou_epsilon
                ) / (union + self.soft_iou_epsilon)
                soft_iou = (
                    per_sample_soft_iou * valid_samples
                ).sum() / valid_samples.sum().clamp_min(1)
                occupancy_loss = 1.0 - soft_iou
            else:
                per_cell_bce = F.binary_cross_entropy(
                    refined_occupancy.clamp(1.0e-6, 1.0 - 1.0e-6),
                    target_occupancy,
                    reduction="none",
                )
                if coarse is not None:
                    for name, group in selected_groups.items():
                        if bool(group_counts[name] > 0):
                            group_losses[name] = (
                                per_cell_bce[group].sum() / group_counts[name]
                            )
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
            elif self.occupancy_loss_mode != "soft_iou":
                occupancy_loss = per_cell_bce.mean()
        continuous_loss = occupancy_loss.new_zeros(())
        if refined.shape[1] == 1:
            total = occupancy_loss
            components = {
                "occupancy_loss": occupancy_loss,
                "soft_iou": soft_iou,
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
            "soft_iou": soft_iou,
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
        crop_multiple = (
            8 if self.config.fine_backbone == "unet" else self.config.window_size
        )
        self.crop_extractor = ReconstructionCropExtractor(
            crop_multiple,
            minimum_height=self.config.fine_min_context_height,
            minimum_width=self.config.fine_min_context_width,
        )
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
        if self.config.fine_backbone == "transformer":
            self.transformer = LocalResidualDiffusionTransformer(self.config)
            self.unet = None
        else:
            self.transformer = None
            self.unet = BasicDiffusionUNet(
                lidar_channels=self.config.lidar_channels,
                radar_channels=self.config.radar_channels,
                use_pointpillars_conditioning=(
                    self.config.use_pointpillars_conditioning
                ),
                lidar_pillar_channels=self.config.lidar_pillar_channels,
                radar_pillar_channels=self.config.radar_pillar_channels,
                include_coarse_input=self.config.fine_unet_include_coarse_input,
                global_context_dim=(
                    self.config.global_context_dim
                    if self.config.fine_unet_use_global_faulty_context
                    else 0
                ),
                base_channels=self.config.fine_unet_base_channels,
                channel_multipliers=tuple(
                    self.config.fine_unet_channel_multipliers
                ),
                num_downsamples=self.config.fine_unet_num_downsamples,
                resblocks_per_level=self.config.fine_unet_resblocks_per_level,
                dropout=self.config.dropout,
            )
        self.unet_global_encoder = (
            GlobalFaultyLidarEncoder(
                self.config.lidar_channels,
                self.config.global_context_dim,
                self.config.hidden_dim,
            )
            if self.config.fine_backbone == "unet"
            and self.config.fine_unet_use_global_faulty_context
            else None
        )
        self.diffusion_loss = MaskedFlowMSELoss(
            self.config.denominator_epsilon
        )
        self.exact_loss = MaskedExactReconstructionLoss(
            self.config.denominator_epsilon,
            occupancy_loss_mode=self.config.occupancy_loss_mode,
            occupancy_threshold=self.config.occupancy_threshold,
            soft_iou_epsilon=self.config.soft_iou_epsilon,
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
            coarse_positive_occupancy_weight=(
                self.config.coarse_positive_occupancy_weight
            ),
            coarse_min_empty_observability_weight=(
                self.config.coarse_min_empty_observability_weight
            ),
        )
        self.degradation_loss = MaskedNoDegradationLoss(
            self.config.denominator_epsilon
        )

    def configure_inference_bucket(self, bucket_multiple: int | None) -> int:
        """Round technical crop padding to stable inference-shape buckets."""

        native_multiple = (
            8 if self.config.fine_backbone == "unet" else self.config.window_size
        )
        if bucket_multiple is None or int(bucket_multiple) <= 0:
            resolved = native_multiple
        else:
            resolved = math.lcm(native_multiple, int(bucket_multiple))
        self.crop_extractor.pad_multiple = resolved
        return resolved

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
        transformer_inference_cache: TransformerInferenceCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        values = crops.tensors
        if self.config.fine_backbone == "unet":
            if self.unet is None:
                raise RuntimeError("Basic Diffusion U-Net is unavailable")
            velocity_physical, debug = self.unet(
                residual_t,
                self.normalization.normalize(values["coarse"]),
                values["radar"],
                values["repair"],
                values["halo"],
                crops.valid_mask,
                timestep,
                values.get("lidar_pillars"),
                values.get("radar_pillars"),
                global_embedding,
            )
            condition = debug["unet_contextual_input"]
        else:
            if self.transformer is None:
                raise RuntimeError("Fine Diffusion Transformer is unavailable")
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
                inference_cache=transformer_inference_cache,
            )
        repair = values["repair"] * crops.valid_mask
        velocity_normalized = self.residual_normalization.normalize(
            velocity_physical
        ) * repair
        debug["predicted_velocity_physical"] = velocity_physical
        debug["current_residual"] = residual_t
        return velocity_normalized, condition, debug

    def _prepare_transformer_inference_cache(
        self,
        crops: ReconstructionCropBatch,
    ) -> TransformerInferenceCache | None:
        if self.config.fine_backbone != "transformer":
            return None
        if self.transformer is None:
            raise RuntimeError("Fine Diffusion Transformer is unavailable")
        values = crops.tensors
        return self.transformer.prepare_inference_cache(
            self.normalization.normalize(values["trusted_faulty"]),
            values["radar"],
            values.get("lidar_pillars"),
            values.get("radar_pillars"),
            values["repair"],
            values["halo"],
            crops.valid_mask,
            _coordinate_channels(crops),
        )

    def _global_context(
        self,
        faulty_lidar_bev: torch.Tensor,
        reconstruction_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.fine_backbone == "transformer":
            if self.transformer is None:
                raise RuntimeError("Fine Diffusion Transformer is unavailable")
            return self.transformer.global_context(
                faulty_lidar_bev, reconstruction_mask
            )
        trusted = faulty_lidar_bev * (1.0 - reconstruction_mask)
        if self.unet_global_encoder is None:
            embedding = faulty_lidar_bev.new_zeros(
                (faulty_lidar_bev.shape[0], self.config.global_context_dim)
            )
        else:
            embedding = self.unet_global_encoder(trusted)
        return trusted, embedding

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
        trusted_global, global_embedding = self._global_context(
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
            observability_confidence=crops.tensors.get(
                "observability_confidence"
            ),
            return_components=True,
        )
        coarse_exact_loss = (
            self.exact_loss(
                crops.tensors["coarse"].detach(),
                crops.tensors["clean"],
                repair,
                crops.tensors["coarse"].detach(),
                observability_confidence=crops.tensors.get(
                    "observability_confidence"
                ),
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
            "soft_iou": exact_components.get(
                "soft_iou", exact_loss.new_zeros(())
            ),
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
                "sampling_steps must be one of 1, 3, 5, 6, 10, 25, or 50"
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
        _trusted_global, global_embedding = self._global_context(
            faulty_lidar_bev, reconstruction_mask
        )
        transformer_inference_cache = self._prepare_transformer_inference_cache(
            crops
        )
        intermediates = []
        residual_x0_normalized = torch.zeros_like(residual_t)
        for scalar_timestep in timesteps:
            timestep = scalar_timestep.expand(coarse_lidar_bev.shape[0])
            velocity_pred, _local_condition, transformer_debug = self._predict_velocity(
                residual_t,
                crops,
                timestep,
                global_embedding,
                transformer_inference_cache,
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
