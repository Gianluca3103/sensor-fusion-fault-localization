"""Train-split coarse-to-clean residual statistics for fine diffusion."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

import torch


class ResidualStatisticsAccumulator:
    """Streaming per-channel moments and absolute-value quantiles."""

    def __init__(
        self,
        channels: int,
        *,
        zero_threshold: float = 1.0e-6,
        histogram_bins: int = 10_000,
        maximum_absolute_value: float = 1.0,
    ) -> None:
        if channels < 1 or histogram_bins < 2:
            raise ValueError("channels and histogram_bins must be positive")
        if zero_threshold < 0 or maximum_absolute_value <= 0:
            raise ValueError("Residual statistic bounds are invalid")
        self.channels = int(channels)
        self.zero_threshold = float(zero_threshold)
        self.histogram_bins = int(histogram_bins)
        self.maximum_absolute_value = float(maximum_absolute_value)
        self.count = torch.zeros(channels, dtype=torch.int64)
        self.total = torch.zeros(channels, dtype=torch.float64)
        self.square_total = torch.zeros(channels, dtype=torch.float64)
        self.absolute_total = torch.zeros(channels, dtype=torch.float64)
        self.maximum = torch.zeros(channels, dtype=torch.float64)
        self.zero_count = torch.zeros(channels, dtype=torch.int64)
        self.histogram = torch.zeros(
            channels, histogram_bins + 1, dtype=torch.int64
        )

    def update(
        self, residual_physical: torch.Tensor, reconstruction_mask: torch.Tensor
    ) -> None:
        if residual_physical.ndim != 4:
            raise ValueError("residual must have shape [B,C,H,W]")
        if residual_physical.shape[1] != self.channels:
            raise ValueError("residual channel count changed during estimation")
        if reconstruction_mask.shape != (
            residual_physical.shape[0],
            1,
            *residual_physical.shape[-2:],
        ):
            raise ValueError("reconstruction mask shape does not match residual")
        selected = reconstruction_mask > 0.5
        for channel in range(self.channels):
            values = residual_physical[:, channel : channel + 1][selected]
            if values.numel() == 0:
                continue
            values = values.detach().double()
            absolute = values.abs()
            count = values.numel()
            self.count[channel] += count
            self.total[channel] += values.sum().cpu()
            self.square_total[channel] += values.square().sum().cpu()
            self.absolute_total[channel] += absolute.sum().cpu()
            self.maximum[channel] = torch.maximum(
                self.maximum[channel], absolute.max().cpu()
            )
            self.zero_count[channel] += int(
                (absolute <= self.zero_threshold).sum().item()
            )
            indices = (
                absolute.clamp(0.0, self.maximum_absolute_value)
                / self.maximum_absolute_value
                * self.histogram_bins
            ).floor().long()
            self.histogram[channel] += torch.bincount(
                indices.cpu(), minlength=self.histogram_bins + 1
            )

    def _quantile(self, channel: int, probability: float) -> float:
        count = int(self.count[channel])
        if count == 0:
            return 0.0
        target = max(1, int(torch.ceil(torch.tensor(probability * count))))
        index = int(
            torch.searchsorted(
                self.histogram[channel].cumsum(0), torch.tensor(target)
            )
        )
        return (
            min(index, self.histogram_bins)
            / self.histogram_bins
            * self.maximum_absolute_value
        )

    def finalize(self, *, minimum_std: float) -> dict:
        if minimum_std <= 0:
            raise ValueError("minimum_std must be positive")
        channels = []
        for channel in range(self.channels):
            count = int(self.count[channel])
            if count == 0:
                raise ValueError(
                    f"No repair-mask residual values for LiDAR channel {channel}"
                )
            mean = float(self.total[channel] / count)
            variance = max(
                float(self.square_total[channel] / count) - mean * mean, 0.0
            )
            raw_std = variance**0.5
            channels.append(
                {
                    "channel": channel,
                    "mean": mean,
                    "raw_std": raw_std,
                    "effective_std": max(raw_std, minimum_std),
                    "mean_absolute_value": float(
                        self.absolute_total[channel] / count
                    ),
                    "median_absolute_value": self._quantile(channel, 0.50),
                    "p95_absolute_value": self._quantile(channel, 0.95),
                    "p99_absolute_value": self._quantile(channel, 0.99),
                    "maximum_absolute_value": float(self.maximum[channel]),
                    "fraction_approximately_zero": float(
                        self.zero_count[channel] / count
                    ),
                    "sample_count": count,
                }
            )
        return {
            "split": "train",
            "definition": "mask * (clean_lidar_bev - coarse_lidar_bev)",
            "zero_threshold": self.zero_threshold,
            "minimum_residual_std": float(minimum_std),
            "quantile_method": (
                f"absolute-value histogram with {self.histogram_bins} bins"
            ),
            "channels": channels,
            "raw_channel_stds": [item["raw_std"] for item in channels],
            "effective_channel_stds": [
                item["effective_std"] for item in channels
            ],
        }


@torch.inference_mode()
def estimate_training_residual_statistics(
    loader: Iterable,
    *,
    move_batch: Callable[[object], Mapping[str, torch.Tensor]],
    coarse_forward: Callable[[Mapping[str, torch.Tensor]], torch.Tensor],
    channels: int,
    minimum_std: float,
    zero_threshold: float = 1.0e-6,
) -> dict:
    """Estimate normalization from the supplied training loader only."""

    accumulator = ResidualStatisticsAccumulator(
        channels, zero_threshold=zero_threshold
    )
    for raw_batch in loader:
        batch = move_batch(raw_batch)
        coarse = coarse_forward(batch).detach()
        residual = batch["reconstruction_mask"] * (
            batch["clean_lidar_bev"] - coarse
        )
        accumulator.update(residual.float(), batch["reconstruction_mask"])
    return accumulator.finalize(minimum_std=minimum_std)
