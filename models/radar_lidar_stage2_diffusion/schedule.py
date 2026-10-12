"""Cosine variance schedule and deterministic DDIM reverse updates."""

from __future__ import annotations

import math

import torch
from torch import nn


class CosineSchedule(nn.Module):
    def __init__(self, timesteps: int):
        super().__init__()
        if timesteps < 2:
            raise ValueError("At least two timesteps are required")
        steps = torch.arange(timesteps + 1, dtype=torch.float64)
        cumulative = torch.cos(((steps / timesteps + .008) / 1.008) * math.pi / 2).square()
        cumulative = cumulative / cumulative[0]
        beta = (1 - cumulative[1:] / cumulative[:-1]).clamp(1e-5, .999)
        alpha_bar = torch.cumprod(1 - beta, dim=0).float()
        self.register_buffer("alpha_bar", alpha_bar)

    def mix(self, clean: torch.Tensor, noise: torch.Tensor,
            timestep: torch.Tensor) -> torch.Tensor:
        if clean.shape != noise.shape or timestep.shape != (len(clean),):
            raise ValueError("Noise, clean state and timestep shapes differ")
        alpha = self.alpha_bar[timestep].to(clean.dtype)[:, None]
        return alpha.sqrt() * clean + (1 - alpha).sqrt() * noise

    def reverse_step(self, noisy: torch.Tensor, predicted_clean: torch.Tensor,
                     t: int) -> torch.Tensor:
        if noisy.shape != predicted_clean.shape or not 0 <= t < len(self.alpha_bar):
            raise ValueError("Invalid DDIM reverse step")
        if t == 0:
            return predicted_clean
        alpha = self.alpha_bar[t].to(noisy.dtype)
        earlier = self.alpha_bar[t - 1].to(noisy.dtype)
        epsilon = (noisy - alpha.sqrt() * predicted_clean) / (1 - alpha).sqrt()
        return earlier.sqrt() * predicted_clean + (1 - earlier).sqrt() * epsilon
