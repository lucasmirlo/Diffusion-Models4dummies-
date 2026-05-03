# -*- coding: utf-8 -*-
"""Legacy Gaussian diffusion-process simulation.

Originally ``diffusion_process.py`` at project root.
@author: <alberto.suarez@uam.es>

This module provides a *trajectory-simulation* API (returns the full
``(times, x_t)`` array) which is distinct from the *generation* API in
``diffusion_lib.processes``.  It is kept here for backward compatibility with
the early demo notebooks.
"""

from __future__ import annotations

from typing import Callable, Union

import numpy as np
import torch
from torch import Tensor


def euler_maruyama_integrator(
    x_0: Tensor,
    t_0: float,
    t_end: float,
    n_steps: int,
    drift_coefficient: Callable[float, float],
    diffusion_coefficient: Callable[float],
    seed: Union[int, None] = None,
) -> Tensor:
    """Euler-Maruyama integrator (approximate).

    Args:
        x_0: Initial images ``(batch_size, n_channels, H, W)``.
        t_0: Start of the integration interval.
        t_end: End of the integration interval.
        n_steps: Number of integration steps.
        drift_coefficient: ``f(x_t, t)`` — drift term of the SDE.
        diffusion_coefficient: ``g(t)`` — diffusion term of the SDE.
        seed: Optional seed for the random number generator.

    Returns:
        ``(times, x_t)`` where ``x_t`` has shape ``(*shape(x_0), n_steps+1)``.

    Examples:
        >>> import numpy as np
        >>> drift_coefficient = lambda x_t, t: - x_t
        >>> diffusion_coefficient = lambda t: torch.ones_like(t)
        >>> x_0 = torch.tensor(np.reshape(np.arange(120), (2, 3, 5, 4)))
        >>> t_0, t_end = 0.0, 3.0
        >>> n_steps = 6
        >>> times, x_t = euler_maruyama_integrator(
        ...     x_0, t_0, t_end, n_steps, drift_coefficient, diffusion_coefficient,
        ... )
        >>> print(times)
        tensor([0.0000, 0.5000, 1.0000, 1.5000, 2.0000, 2.5000, 3.0000])
        >>> print(np.shape(x_t))
        torch.Size([2, 3, 5, 4, 7])
    """
    device = x_0.device
    times = torch.linspace(t_0, t_end, n_steps + 1, device=device)
    dt = times[1] - times[0]

    x_t = torch.tensor(
        np.empty((*np.shape(x_0), len(times))),
        dtype=torch.float32,
        device=device,
    )
    x_t[..., 0] = x_0
    z = torch.randn_like(x_t)
    z[..., -1] = 0.0  # no noise injection at the last step

    for n, t in enumerate(times[:-1]):
        t = torch.ones(x_0.shape[0], device=device) * t
        x_t[..., n + 1] = (
            x_t[..., n]
            + drift_coefficient(x_t[..., n], t) * dt
            + diffusion_coefficient(t).view(-1, 1, 1, 1)
            * np.sqrt(np.abs(dt))
            * z[..., n]
        )

    return times, x_t


class DiffussionProcess:
    """Base class for a diffusion process defined by its SDE coefficients."""

    def __init__(
        self,
        drift_coefficient: Callable[float, float] = lambda x_t, t: 0.0,
        diffusion_coefficient: Callable[float] = lambda t: 1.0,
    ):
        self.drift_coefficient = drift_coefficient
        self.diffusion_coefficient = diffusion_coefficient


class GaussianDiffussionProcess(DiffussionProcess):
    """Gaussian diffusion process with closed-form marginal kernel.

    Example:
        >>> mu, sigma = 1.5, 2.0
        >>> bm = GaussianDiffussionProcess(
        ...     drift_coefficient=lambda x_t, t: mu,
        ...     diffusion_coefficient=lambda t: sigma,
        ...     mu_t=lambda x_0, t: x_0 + mu*t,
        ...     sigma_t=lambda t: np.sqrt(2.0 * t),
        ... )
        >>> print(bm.drift_coefficient(x_t=3.0, t=10.0))
        1.5
        >>> print(bm.diffusion_coefficient(t=10.0))
        2.0
        >>> print(bm.mu_t(x_0=3.0, t=10.0), bm.sigma_t(t=10.0))
        18.0 4.47213595499958
    """

    kind = "Gaussian"

    def __init__(
        self,
        drift_coefficient: Callable[float, float] = lambda x_t, t: 0.0,
        diffusion_coefficient: Callable[float] = lambda t: 1.0,
        mu_t: Callable[float, float] = lambda x_0, t: x_0,
        sigma_t: Callable[float] = lambda t: np.sqrt(t),
    ):
        self.drift_coefficient = drift_coefficient
        self.diffusion_coefficient = diffusion_coefficient
        self.mu_t = mu_t
        self.sigma_t = sigma_t

    def loss_function(
        self,
        score_model,
        x_0: torch.Tensor,
        eps: float = 1.0e-5,
    ):
        """Weighted denoising score-matching loss."""
        t = torch.rand(x_0.shape[0], device=x_0.device) * (1.0 - eps) + eps
        z = torch.randn_like(x_0)
        view_shape = [x_0.shape[0]] + [1] * (x_0.dim() - 1)
        t_expanded = t.view(*view_shape)
        mu = self.mu_t(x_0, t_expanded)
        sigma = self.sigma_t(t_expanded)
        x_t = mu + sigma * z
        s = score_model(x_t, t)
        dim_to_sum = list(range(1, x_0.dim()))
        squared_norm = torch.sum(((sigma * s) + z) ** 2, dim=dim_to_sum)
        loss = torch.mean(squared_norm)
        return loss


if __name__ == "__main__":
    import doctest
    doctest.testmod()
