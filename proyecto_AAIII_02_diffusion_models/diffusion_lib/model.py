# -*- coding: utf-8 -*-
"""High-level GenerativeDiffusionModel that composes process, sampler, and score network."""

import torch
import torch.nn as nn

from .processes.base import DiffusionProcess
from .samplers.base import Sampler


class GenerativeDiffusionModel:
    """Composable generative model: DiffusionProcess × Sampler × score network.

    Provides a unified interface for training, sampling, and evaluation
    regardless of which process (VE/VP), sampler (EM/PC/ODE), or noise
    schedule is used.

    Example::

        from diffusion_lib import (
            VEProcess, EulerMaruyamaSampler, GenerativeDiffusionModel
        )

        process = VEProcess(sigma=25.0)
        sampler = EulerMaruyamaSampler()
        gm = GenerativeDiffusionModel(process, sampler, score_net, device)

        loss    = gm.compute_loss(x_batch)
        samples = gm.sample(n_images=8, img_shape=(1, 28, 28))
        bpd     = gm.compute_bpd(x_batch)

    Args:
        process:     A DiffusionProcess (VEProcess or VPProcess).
        sampler:     A Sampler (Euler-Maruyama, PC, or Probability-Flow ODE).
        score_model: Trained (or in-training) score network s_θ(x, t).
        device:      Torch device.
        T:           Diffusion horizon — must match training.
        eps:         Lower time bound for numerical stability.
    """

    def __init__(
        self,
        process: DiffusionProcess,
        sampler: Sampler,
        score_model: nn.Module,
        device: torch.device = torch.device("cpu"),
        T: float = 1.0,
        eps: float = 1e-3,
    ):
        self.process = process
        self.sampler = sampler
        self.score_model = score_model.to(device)
        self.device = device
        self.T = T
        self.eps = eps

    # ── Training ─────────────────────────────────────────────────────────────

    def compute_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """Denoising score-matching loss on a mini-batch x_0."""
        return self.process.loss_function(
            self.score_model, x_0.to(self.device)
        )

    # ── Sampling ─────────────────────────────────────────────────────────────

    def sample(
        self,
        n_images: int,
        img_shape: tuple,
        n_steps: int = 500,
        **kwargs,
    ) -> torch.Tensor:
        """Generate n_images samples.

        Args:
            n_images:  Number of images to generate.
            img_shape: Shape of one image, e.g. (1, 28, 28) or (3, 32, 32).
            n_steps:   Reverse integration steps.
            **kwargs:  Extra arguments forwarded to the sampler, e.g.
                       ``return_trajectory=True`` for EM and ODE samplers.

        Returns:
            Tensor [n_images, *img_shape], or [n_images, C, H, W, n_steps+1]
            when return_trajectory=True.
        """
        return self.sampler.sample(
            score_model=self.score_model,
            process=self.process,
            n_images=n_images,
            img_shape=img_shape,
            n_steps=n_steps,
            device=self.device,
            T=self.T,
            eps=self.eps,
            **kwargs,
        )

    # ── Metrics ──────────────────────────────────────────────────────────────

    def compute_bpd(
        self,
        x_0: torch.Tensor,
        n_steps: int = 100,
        n_hutchinson: int = 1,
    ) -> torch.Tensor:
        """Bits-per-dimension for a batch of images via the probability flow ODE.

        Args:
            x_0:           Images [B, C, H, W].
            n_steps:       ODE integration steps (more = more accurate).
            n_hutchinson:  Hutchinson noise vectors per step.

        Returns:
            Tensor [B] — BPD per image.
        """
        from .metrics.bpd import compute_bpd
        return compute_bpd(
            score_model=self.score_model,
            process=self.process,
            x_0=x_0.to(self.device),
            n_steps=n_steps,
            n_hutchinson=n_hutchinson,
            T=self.T,
            eps=self.eps,
        )

    def compute_fid(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        batch_size: int = 32,
    ) -> float:
        """Fréchet Inception Distance between real and generated images."""
        from .metrics.fid_is import compute_fid
        return compute_fid(real_images, fake_images, self.device, batch_size)

    def compute_is(
        self,
        images: torch.Tensor,
        batch_size: int = 32,
        n_splits: int = 10,
    ) -> tuple[float, float]:
        """Inception Score (mean, std) for a set of generated images."""
        from .metrics.fid_is import compute_is
        return compute_is(images, self.device, batch_size, n_splits)
