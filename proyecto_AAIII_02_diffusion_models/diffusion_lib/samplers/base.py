# -*- coding: utf-8 -*-
"""Abstract sampler for the reverse diffusion process."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Sampler(ABC):
    """Abstract base for reverse-process samplers.

    All samplers share the same interface: given a trained score model and a
    diffusion process, integrate the reverse SDE (or ODE) from T → ε to
    produce synthetic images.
    """

    @abstractmethod
    def sample(
        self,
        score_model: nn.Module,
        process,
        n_images: int,
        img_shape: tuple,
        n_steps: int,
        device: torch.device,
        T: float = 1.0,
        eps: float = 1e-3,
    ) -> torch.Tensor:
        """Generate images by integrating the reverse process.

        Args:
            score_model: Trained score network s_θ(x, t).
            process:     A DiffusionProcess instance.
            n_images:    Number of images to generate.
            img_shape:   Shape of a single image, e.g. (1, 28, 28).
            n_steps:     Number of integration steps.
            device:      Torch device.
            T:           Diffusion horizon (start of reverse integration).
            eps:         Lower time bound for numerical stability.

        Returns:
            Tensor of shape (n_images, *img_shape).
        """
