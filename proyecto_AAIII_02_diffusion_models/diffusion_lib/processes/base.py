# -*- coding: utf-8 -*-
"""Abstract base class for diffusion processes."""

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn


class DiffusionProcess(ABC):
    """Abstract base for forward-time diffusion SDEs.

    Subclasses implement the process-specific coefficients (drift, diffusion,
    marginal mean/std, prior).  The shared training loss, perturbation
    reparametrisation, and reverse-drift formula live here.
    """

    # ── Abstract interface ───────────────────────────────────────────────────

    @abstractmethod
    def drift_coefficient(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """f(x_t, t) — drift term of the forward SDE."""

    @abstractmethod
    def diffusion_coefficient(self, t: torch.Tensor) -> torch.Tensor:
        """g(t) — scalar diffusion coefficient, shape [B]."""

    @abstractmethod
    def mu_t(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Mean of the perturbation kernel p(x_t | x_0).
        t has the same shape as x_0 (already broadcast-expanded).
        """

    @abstractmethod
    def sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Std deviation of the perturbation kernel p(x_t | x_0).
        Works element-wise for any shape of t.
        """

    @abstractmethod
    def prior_sample(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """Draw x_T ~ p_T(x)."""

    @abstractmethod
    def log_prior(self, x_T: torch.Tensor, T: float = 1.0) -> torch.Tensor:
        """Log-density log p_T(x_T), shape [B].  Required for BPD."""

    # ── Shared implementations ───────────────────────────────────────────────

    def perturb(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t | x_0 via reparametrisation: x_t = μ_t + σ_t · ε.

        Args:
            x_0: Clean images [B, C, H, W].
            t:   Times [B].

        Returns:
            (x_t, noise): Both of shape [B, C, H, W].
        """
        view_shape = [x_0.shape[0]] + [1] * (x_0.dim() - 1)
        t_exp = t.view(*view_shape)
        noise = torch.randn_like(x_0)
        x_t = self.mu_t(x_0, t_exp) + self.sigma_t(t_exp) * noise
        return x_t, noise

    def reverse_drift(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor,
    ) -> torch.Tensor:
        """Drift of the reverse-time SDE: f(x,t) − g(t)² · s_θ(x,t).

        Args:
            x_t:   Noisy images [B, C, H, W].
            t:     Times [B].
            score: Score estimate s_θ(x_t, t) [B, C, H, W].
        """
        view_shape = [x_t.shape[0]] + [1] * (x_t.dim() - 1)
        g2 = self.diffusion_coefficient(t).view(*view_shape) ** 2
        return self.drift_coefficient(x_t, t) - g2 * score

    def loss_function(
        self,
        score_model: nn.Module,
        x_0: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """Weighted denoising score-matching loss.

        Minimises  E_t E_{x_t|x_0} [ ‖σ_t · s_θ(x_t, t) + ε‖² ]

        which is zero when  s_θ(x_t, t) = ∇_x log p_t(x_t).
        """
        t = torch.rand(x_0.shape[0], device=x_0.device) * (1.0 - eps) + eps
        x_t, noise = self.perturb(x_0, t)

        view_shape = [x_0.shape[0]] + [1] * (x_0.dim() - 1)
        sigma = self.sigma_t(t.view(*view_shape))

        score = score_model(x_t, t)
        per_sample = torch.sum(
            (sigma * score + noise) ** 2,
            dim=list(range(1, x_0.dim())),
        )
        return per_sample.mean()
