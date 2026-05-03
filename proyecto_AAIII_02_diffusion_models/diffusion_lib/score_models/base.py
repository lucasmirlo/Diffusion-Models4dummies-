# -*- coding: utf-8 -*-
"""Abstract base class for all score networks."""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BaseScoreModel(nn.Module, ABC):
    """Score network s_θ(x, t [, y]).

    Both unconditional and conditional score networks inherit from this class
    so that samplers can treat them interchangeably.

    Parameters
    ----------
    x : torch.Tensor
        Noisy images, shape ``(B, C, H, W)``.
    t : torch.Tensor
        Diffusion times, shape ``(B,)``.
    class_label : torch.Tensor, optional
        Integer class labels, shape ``(B,)``.  ``None`` means unconditional.
    """

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        class_label: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the estimated score s_θ(x, t [, y]).

        Returns
        -------
        torch.Tensor
            Same shape as ``x``.
        """
