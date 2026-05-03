# -*- coding: utf-8 -*-
"""Cosine noise schedule for VP diffusion (Nichol & Dhariwal, 2021)."""

import numpy as np
import torch

from .base import NoiseSchedule


class CosineSchedule(NoiseSchedule):
    def __init__(self, s: float = 0.008):
        self.s = s
        self._f0 = float(np.cos(np.pi / 2.0 * s / (1.0 + s)) ** 2)

    def _f(self, t: torch.Tensor) -> torch.Tensor:
        # f(t) = cos²(π/2 · (t+s)/(1+s))
        return torch.cos(np.pi / 2.0 * (t + self.s) / (1.0 + self.s)) ** 2

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        # ᾱ(t) = f(t)/f(0), garantiza ᾱ(0)=1
        return self._f(t) / self._f0

    def integral_beta(self, t: torch.Tensor) -> torch.Tensor:
        # B(t) = -log ᾱ(t)
        return -torch.log(self.alpha_bar(t).clamp(min=1e-8))

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        # β(t) = π/(1+s) · tan(π/2 · (t+s)/(1+s))
        u = np.pi / 2.0 * (t + self.s) / (1.0 + self.s)
        return torch.clamp(np.pi / (1.0 + self.s) * torch.tan(u), max=20.0)
