# -*- coding: utf-8 -*-
"""Bits-per-dimension via the probability flow ODE."""

import numpy as np
import torch
import torch.nn as nn


def compute_bpd(
    score_model: nn.Module,
    process,
    x_0: torch.Tensor,
    n_steps: int = 100,
    n_hutchinson: int = 1,
    T: float = 1.0,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Bits-per-dimension for x_0 via the probability flow ODE.

    Encodes x_0 → x_T by running the ODE *forward* (t: eps → T) while
    accumulating the log-density change through Hutchinson's trace estimator:

        log p_0(x_0) = log p_T(x_T) + ∫_eps^T div[f_ode(x_t, t)] dt

    where  f_ode(x,t) = f(x,t) − ½ g(t)² · s_θ(x,t)
    and    div[f_ode] ≈ εᵀ [∂f_ode/∂x]ᵀ ε,  ε ~ N(0, I)  (Hutchinson).

    Args:
        score_model:    Trained score network s_θ(x, t).
        process:        A DiffusionProcess instance (VEProcess or VPProcess).
        x_0:            Input images [B, C, H, W].
        n_steps:        ODE integration steps (more → more accurate, slower).
        n_hutchinson:   Hutchinson noise vectors per step (1 is standard).
        T:              Diffusion horizon.
        eps:            Lower time bound for numerical stability.

    Returns:
        bpd: Tensor [B] — bits per dimension for each image (lower is better).
    """
    device = x_0.device
    B = x_0.shape[0]
    D = x_0[0].numel()

    dt = (T - eps) / n_steps
    t_vals = torch.linspace(eps, T, n_steps + 1, device=device)

    x = x_0.detach().clone()
    log_prob_delta = torch.zeros(B, device=device)
    view_shape = [B] + [1] * (x_0.dim() - 1)
    dim_sum = list(range(1, x_0.dim()))

    score_model.eval()

    for i in range(n_steps):
        t = t_vals[i].expand(B)

        x_req = x.detach().requires_grad_(True)

        score = score_model(x_req, t)
        g2 = process.diffusion_coefficient(t).view(*view_shape) ** 2
        f_t = process.drift_coefficient(x_req, t)
        ode_drift = f_t - 0.5 * g2 * score

        # Hutchinson divergence: E_ε[εᵀ Jᵀ ε] = Tr(J) = div(f_ode)
        div = torch.zeros(B, device=device)
        for _ in range(n_hutchinson):
            epsilon = torch.randn_like(x)
            vjp = torch.autograd.grad(
                ode_drift,
                x_req,
                grad_outputs=epsilon,
                retain_graph=True,
            )[0]
            div += (epsilon * vjp).sum(dim=dim_sum)
        div /= n_hutchinson

        # d/dt log p_t = −div(f_ode)  →  log p_0 = log p_T + ∫ div dt
        log_prob_delta += div * dt
        x = x + ode_drift.detach() * dt

    log_p_T = process.log_prior(x.detach(), T=T)
    log_p_0 = log_p_T + log_prob_delta
    bpd = -log_p_0 / (D * np.log(2))
    return bpd
