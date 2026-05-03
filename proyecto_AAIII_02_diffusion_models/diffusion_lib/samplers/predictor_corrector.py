# -*- coding: utf-8 -*-
"""Predictor-Corrector (PC) sampler (Song et al., 2021)."""

import numpy as np
import torch

from .base import Sampler


class PredictorCorrectorSampler(Sampler):
    """PC sampler: Euler-Maruyama predictor + annealed Langevin corrector.

    At each time step t_n → t_{n+1}:

    Predictor — one reverse Euler-Maruyama step:
        x' = x_n + [f(x_n, t_n) − g(t_n)² · s_θ(x_n, t_n)] Δt
             + g(t_n) √|Δt| · z

    Corrector — M annealed Langevin steps at t_{n+1}:
        ε  = 2 · (snr · ‖z‖ / ‖s_θ‖)²
        x  = x + ε · s_θ(x, t_{n+1}) + √(2ε) · z

    Args:
        n_corrector_steps: Langevin steps per predictor step.  Default 1.
        snr: Signal-to-noise ratio controlling the Langevin step size.
             Default 0.16 (Song 2021).
    """

    def __init__(self, n_corrector_steps: int = 1, snr: float = 0.16):
        self.n_corrector_steps = n_corrector_steps
        self.snr = snr

    def _langevin_step(self, x, t_batch, score_model):
        for _ in range(self.n_corrector_steps):
            grad = score_model(x, t_batch)
            noise = torch.randn_like(x)
            grad_norm = grad.view(x.shape[0], -1).norm(dim=1).mean()
            noise_norm = noise.view(x.shape[0], -1).norm(dim=1).mean()
            step = 2.0 * (self.snr * noise_norm / (grad_norm + 1e-8)) ** 2
            x = x + step * grad + (2.0 * step).sqrt() * noise
        return x

    def sample(
        self,
        score_model,
        process,
        n_images,
        img_shape,
        n_steps,
        device,
        T=1.0,
        eps=1e-3,
    ) -> torch.Tensor:
        x = process.prior_sample((n_images, *img_shape), device)
        dt = (eps - T) / n_steps
        t_vals = torch.linspace(T, eps, n_steps + 1, device=device)
        view_shape = [n_images] + [1] * len(img_shape)

        score_model.eval()
        with torch.no_grad():
            for i in range(n_steps):
                t = t_vals[i].expand(n_images)

                # ── Predictor ─────────────────────────────────────────────
                score = score_model(x, t)
                drift = process.reverse_drift(x, t, score)
                g_t = process.diffusion_coefficient(t).view(*view_shape)
                z = torch.randn_like(x)
                x = x + drift * dt + g_t * np.sqrt(abs(dt)) * z

                # ── Corrector at t_{n+1} ───────────────────────────────────
                t_next = t_vals[i + 1].expand(n_images)
                x = self._langevin_step(x, t_next, score_model)

        return x
