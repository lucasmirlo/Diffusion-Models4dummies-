# -*- coding: utf-8 -*-
"""Variance-Exploding (VE) diffusion process — Brownian motion."""

import numpy as np
import torch

from .base import DiffusionProcess


class VEProcess(DiffusionProcess):
    """Variance-Exploding SDE with σ^t diffusion coefficient.

    Forward SDE:  dx = σ^t dW
    Marginal:     x_t | x_0 ~ N(x_0, σ_t² I)
                  σ_t² = (σ^(2t) − 1) / (2 ln σ)
    Prior:        x_T ~ N(0, σ_T² I)

    Args:
        sigma: Base diffusion parameter σ > 1.  Typically 25.0 (Song 2021).
        T:     Diffusion horizon.  Default 1.0.
    """

    def __init__(self, sigma: float = 25.0, T: float = 1.0):
        self.sigma = sigma
        self.T = T

    # ── Forward process ──────────────────────────────────────────────────────

    def drift_coefficient(self, x_t, t):
        return torch.zeros_like(x_t)

    def diffusion_coefficient(self, t):
        # g(t) = σ^t,  shape mirrors t
        return self.sigma ** t

    def mu_t(self, x_0, t):
        # No drift → mean is x_0 regardless of t
        return x_0

    def sigma_t(self, t):
        return torch.sqrt(
            0.5 * (self.sigma ** (2.0 * t) - 1.0) / np.log(self.sigma)
        )

    # ── Prior ────────────────────────────────────────────────────────────────

    def prior_sample(self, shape, device):
        t_T = torch.ones(1, device=device) * self.T
        sigma_T = self.sigma_t(t_T).item()
        return torch.randn(shape, device=device) * sigma_T

    def log_prior(self, x_T, T=None):
        T = T if T is not None else self.T
        device = x_T.device
        sigma_T = self.sigma_t(torch.tensor([T], device=device, dtype=torch.float32)).item()
        D = x_T[0].numel()
        log_norm = -0.5 * D * np.log(2 * np.pi * sigma_T ** 2)
        log_exp = (
            -0.5
            * x_T.reshape(x_T.shape[0], -1).pow(2).sum(dim=1)
            / sigma_T ** 2
        )
        return log_norm + log_exp
