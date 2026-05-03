# -*- coding: utf-8 -*-
"""Abstract noise schedule for VP processes."""

from abc import ABC, abstractmethod

import torch


class NoiseSchedule(ABC):
    """Abstract noise schedule β(t) used by VPProcess.

    Subclasses provide β(t) and its cumulative integral B(t) = ∫₀ᵗ β(s) ds.
    """

    @abstractmethod
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Instantaneous noise level β(t), same shape as t."""

    @abstractmethod
    def integral_beta(self, t: torch.Tensor) -> torch.Tensor:
        """Cumulative noise B(t) = ∫₀ᵗ β(s) ds, same shape as t."""
