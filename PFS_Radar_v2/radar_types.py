from __future__ import annotations

from dataclasses import dataclass
import math


class RadarAlignmentUnavailableError(RuntimeError):
    """Raised when a paired K-Radar frame cannot be aligned."""


@dataclass(frozen=True)
class DopplerConfig:
    """K-Radar ego-Doppler compensation and dynamic-point settings."""

    dynamic_threshold_mps: float = 1.0
    doppler_sign: str = "auto"
    sign_inference_min_speed_mps: float = 0.5
    max_abs_velocity_mps: float = 30.0
    doppler_period_mps: float = 3.865182436611008
    dynamic_power_quantile: float = 0.9

    def validate(self) -> None:
        if self.doppler_sign not in {"auto", "1", "-1"}:
            raise ValueError("doppler_sign must be one of: auto, 1, -1")
        positive = {
            "dynamic_threshold_mps": self.dynamic_threshold_mps,
            "sign_inference_min_speed_mps": self.sign_inference_min_speed_mps,
            "max_abs_velocity_mps": self.max_abs_velocity_mps,
            "doppler_period_mps": self.doppler_period_mps,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 <= self.dynamic_power_quantile < 1.0:
            raise ValueError("dynamic_power_quantile must lie in [0, 1)")
