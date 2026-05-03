# -*- coding: utf-8 -*-
"""Euler-Maruyama sampler for the reverse-time SDE."""

import numpy as np
import torch

from .base import Sampler


class EulerMaruyamaSampler(Sampler):
    """Stochastic Euler-Maruyama integrator for the reverse-time SDE.

    Update rule:
        x_{n+1} = x_n + [f(x_n, t_n) − g(t_n)² · s_θ(x_n, t_n)] Δt
                  + g(t_n) √|Δt| · z_n ,   z_n ~ N(0, I)

    where Δt < 0 (integrating from T to ε).
    """

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
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            return_trajectory: If True, return all intermediate steps as a
                tensor of shape (n_images, C, H, W, n_steps+1) suitable for
                ``diffusion_utilities.plot_image_evolution_rgb``.
                If False (default), return only the final images (n_images, C, H, W).
        """
        x = process.prior_sample((n_images, *img_shape), device)
        dt = (eps - T) / n_steps
        t_vals = torch.linspace(T, eps, n_steps + 1, device=device)
        view_shape = [n_images] + [1] * len(img_shape)

        trajectory = [x.clone()] if return_trajectory else None

        score_model.eval()
        with torch.no_grad():
            for i in range(n_steps):
                t = t_vals[i].expand(n_images)
                score = score_model(x, t)
                drift = process.reverse_drift(x, t, score)
                g_t = process.diffusion_coefficient(t).view(*view_shape)
                z = torch.randn_like(x)
                x = x + drift * dt + g_t * np.sqrt(abs(dt)) * z
                if return_trajectory:
                    trajectory.append(x.clone())

        if return_trajectory:
            # [n_steps+1, n_images, C, H, W] → [n_images, C, H, W, n_steps+1]
            return torch.stack(trajectory, dim=0).permute(1, 2, 3, 4, 0)
        return x
