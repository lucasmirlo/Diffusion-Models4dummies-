# -*- coding: utf-8 -*-
"""Inpainting sampler via replacement / RePaint-style reverse SDE."""

import math

import torch


class ImputationSampler:
    """Reverse-SDE inpainting via pixel replacement at each step.

    At every reverse step the known pixels are resampled from the
    forward marginal at the current noise level:

        x_t[known] = mu_t(x0, t) + sigma_t(t) * z,   z ~ N(0,I)

    This keeps the known region consistent with the forward process
    while the score model fills in the unknown region.

    Parameters
    ----------
    n_resample : int
        Number of forward/backward cycles per step (RePaint style).
        ``1`` is the standard single-step replacement; values > 1
        improve boundary coherence at the cost of compute.
    """

    def __init__(self, n_resample: int = 1) -> None:
        self.n_resample = n_resample

    @torch.no_grad()
    def inpaint(
        self,
        score_model,
        process,
        x_known: torch.Tensor,
        mask: torch.Tensor,
        n_steps: int = 500,
        device: torch.device = None,
        T: float = 0.999,
        eps: float = 1e-3,
    ) -> torch.Tensor:
        """Inpaint the unknown pixels of ``x_known``.

        Parameters
        ----------
        score_model : callable
            Score model ``s_θ(x, t)`` or a ``CFGWrapper``.
        process : DiffusionProcess
            The forward SDE (VE or VP).
        x_known : torch.Tensor
            Ground-truth images ``(B, C, H, W)`` in [0, 1].
        mask : torch.Tensor
            Binary mask ``(B, 1, H, W)`` or ``(B, C, H, W)``.
            ``1`` = known pixel (will be preserved), ``0`` = to infer.
        n_steps : int
            Number of reverse integration steps.
        device : torch.device, optional
            Target device.  Defaults to ``x_known.device``.
        T : float
            Diffusion horizon.
        eps : float
            Stopping time (avoids numerical instability near t = 0).

        Returns
        -------
        torch.Tensor
            Inpainted images, same shape as ``x_known``.
        """
        if device is None:
            device = x_known.device
        x_known = x_known.to(device)
        mask = mask.to(device).float()

        B, C, H, W = x_known.shape
        img_shape = (C, H, W)
        view_shape = [B] + [1] * len(img_shape)

        x = process.prior_sample((B, *img_shape), device)
        dt = (eps - T) / n_steps
        t_vals = torch.linspace(T, eps, n_steps + 1, device=device)

        if hasattr(score_model, 'eval'):
            score_model.eval()

        for i in range(n_steps):
            t_cur = t_vals[i].expand(B)
            t_nxt = t_vals[i + 1].expand(B)

            for r in range(self.n_resample):
                # ── Reverse step on the full image ─────────────────────────
                score = score_model(x, t_cur)
                drift = process.reverse_drift(x, t_cur, score)
                g_t = process.diffusion_coefficient(t_cur).view(*view_shape)
                z = torch.randn_like(x)
                x_pred = x + drift * dt + g_t * math.sqrt(abs(dt)) * z

                # ── Replace observed pixels with correctly-noised original ─
                x_known_noisy, _ = process.perturb(x_known, t_nxt)
                x = mask * x_known_noisy + (1.0 - mask) * x_pred

                # RePaint: re-noise before next inner iteration
                if self.n_resample > 1 and r < self.n_resample - 1:
                    x_cur_noisy, _ = process.perturb(x, t_cur)
                    x = x_cur_noisy

        # Pin known pixels exactly at t ≈ 0 (remove any residual noise)
        x = mask * x_known + (1.0 - mask) * x
        return x
