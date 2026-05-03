# -*- coding: utf-8 -*-
"""Classifier-Free Guidance wrapper for conditional score models."""

from typing import Union

import torch


class CFGWrapper:
    """Wraps a conditional score model to produce a CFG-guided score.

    The guided score is computed with the Ho et al. formulation::

        s_guided = s_uncond + w * (s_cond - s_uncond)

    which is equivalent to the Song-style ``(1+w)*s_cond - w*s_uncond``
    when ``w_Song = w_Ho - 1``.

    Parameters
    ----------
    model : CondUNetScoreModel
        A conditional score model whose ``forward(x, t, class_label)``
        accepts an optional integer label tensor.
    class_label : int or torch.Tensor
        Target class.  An ``int`` is broadcast to the batch dimension at
        call time; a 1-D ``torch.Tensor`` is used as-is.
    cfg_scale : float
        Guidance scale ``w``.  ``w=0`` → unconditional; larger values
        increase class fidelity at the cost of sample diversity.
    """

    def __init__(
        self,
        model,
        class_label: Union[int, torch.Tensor],
        cfg_scale: float = 3.0,
    ) -> None:
        self.model = model
        self.y = class_label
        self.w = cfg_scale

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        if isinstance(self.y, torch.Tensor):
            y = self.y.to(x.device)
        else:
            y = torch.full((B,), self.y, dtype=torch.long, device=x.device)

        s_uncond = self.model(x, t, class_label=None)
        s_cond = self.model(x, t, class_label=y)
        return s_uncond + self.w * (s_cond - s_uncond)

    def eval(self) -> "CFGWrapper":
        self.model.eval()
        return self

    def train(self) -> "CFGWrapper":
        self.model.train()
        return self

    def to(self, device) -> "CFGWrapper":
        self.model.to(device)
        if isinstance(self.y, torch.Tensor):
            self.y = self.y.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device
