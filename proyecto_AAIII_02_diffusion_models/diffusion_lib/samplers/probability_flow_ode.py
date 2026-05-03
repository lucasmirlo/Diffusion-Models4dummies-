# -*- coding: utf-8 -*-
"""Probability Flow ODE sampler (deterministic)."""

import torch

from .base import Sampler


class ProbabilityFlowODESampler(Sampler):
    """Deterministic probability flow ODE sampler.

    Integrates the ODE  dx/dt = f(x,t) − ½ g(t)² · s_θ(x,t)  from T → ε
    using the Euler method.  No stochastic term — faster and more stable than
    the SDE samplers, and enables exact log-likelihood computation via the
    instantaneous change-of-variables formula (see metrics/bpd.py).
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
                g2 = process.diffusion_coefficient(t).view(*view_shape) ** 2
                # ODE drift = f(x,t) − ½ g² · score  (half the SDE diffusion term)
                ode_drift = process.drift_coefficient(x, t) - 0.5 * g2 * score
                x = x + ode_drift * dt
                if return_trajectory:
                    trajectory.append(x.clone())

        if return_trajectory:
            # [n_steps+1, n_images, C, H, W] → [n_images, C, H, W, n_steps+1]
            return torch.stack(trajectory, dim=0).permute(1, 2, 3, 4, 0)
        return x
