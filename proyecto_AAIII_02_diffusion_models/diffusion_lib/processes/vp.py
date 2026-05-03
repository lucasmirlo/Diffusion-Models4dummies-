# -*- coding: utf-8 -*-
"""Variance-Preserving (VP) diffusion process — Ornstein-Uhlenbeck."""

import numpy as np
import torch

from .base import DiffusionProcess
from ..schedules.base import NoiseSchedule


class VPProcess(DiffusionProcess):
    """Variance-Preserving SDE (Ornstein-Uhlenbeck process).

    Forward SDE:  dx = −½ β(t) x dt + √β(t) dW
    Marginal:     x_t | x_0 ~ N(α_t x_0, σ_t² I)

                  B(t)  = ∫₀ᵗ β(s) ds          (from the noise schedule)
                  α_t   = exp(−½ B(t))
                  σ_t   = √(1 − exp(−B(t)))

    Prior:        x_T ≈ N(0, I)  (converges when B(T) is large)

    Args:
        schedule: A NoiseSchedule instance providing β(t) and B(t).
        T:        Diffusion horizon.  Default 1.0.
    """

    def __init__(self, schedule: NoiseSchedule, T: float = 1.0):
        self.schedule = schedule
        self.T = T

    # ── Forward process ──────────────────────────────────────────────────────

    def drift_coefficient(self, x_t, t):
        view_shape = [x_t.shape[0]] + [1] * (x_t.dim() - 1)
        beta_t = self.schedule.beta(t).view(*view_shape)
        return -0.5 * beta_t * x_t

    def diffusion_coefficient(self, t):
        return torch.sqrt(self.schedule.beta(t))

    def mu_t(self, x_0, t):
        # t already broadcast-expanded to x_0 shape
        B_t = self.schedule.integral_beta(t)
        return x_0 * torch.exp(-0.5 * B_t)

    def sigma_t(self, t):
        B_t = self.schedule.integral_beta(t)
        return torch.sqrt(torch.clamp(1.0 - torch.exp(-B_t), min=1e-5))

    # ── Prior ────────────────────────────────────────────────────────────────

    def prior_sample(self, shape, device):
        # VP converges to N(0, I) for large B(T)
        return torch.randn(shape, device=device)

    def log_prior(self, x_T, T=None):
        D = x_T[0].numel()
        log_norm = -0.5 * D * np.log(2 * np.pi)
        log_exp = -0.5 * x_T.reshape(x_T.shape[0], -1).pow(2).sum(dim=1)
        return log_norm + log_exp
