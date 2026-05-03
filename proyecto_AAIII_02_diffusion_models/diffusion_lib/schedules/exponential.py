# -*- coding: utf-8 -*-
"""Exponential noise schedule for VP diffusion."""

import math
import torch

from .base import NoiseSchedule


class ExponentialSchedule(NoiseSchedule):

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        if beta_min <= 0 or beta_max <= 0:
            raise ValueError("beta_min and beta_max must be strictly positive.")
        if beta_max == beta_min:
            raise ValueError(
                "beta_max must differ from beta_min for an exponential schedule."
            )
        self.beta_min = beta_min
        self.beta_max = beta_max
        # k = ln(β_max / β_min)  — precomputed for numerical efficiency
        self._k = math.log(beta_max / beta_min)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        # β(t) = β_min · exp(k · t)
        return self.beta_min * torch.exp(self._k * t)

    def integral_beta(self, t: torch.Tensor) -> torch.Tensor:
        # B(t) = (β(t) − β_min) / k  =  β_min · (exp(k · t) − 1) / k
        return (self.beta(t) - self.beta_min) / self._k