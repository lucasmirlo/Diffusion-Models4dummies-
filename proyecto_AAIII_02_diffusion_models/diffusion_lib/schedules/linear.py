# -*- coding: utf-8 -*-
"""Linear noise schedule for VP diffusion."""

import torch

from .base import NoiseSchedule


class LinearSchedule(NoiseSchedule):
    """Linear noise schedule: β(t) = β_min + (β_max − β_min) · t.

    Integral: B(t) = β_min · t + ½ (β_max − β_min) · t²

    Args:
        beta_min: β at t = 0.  Default 0.1.
        beta_max: β at t = 1.  Default 20.0.
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def integral_beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2
