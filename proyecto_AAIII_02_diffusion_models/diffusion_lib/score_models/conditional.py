# -*- coding: utf-8 -*-
"""Conditional U-Net score model with class embedding for CFG."""

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .base import BaseScoreModel


class _GaussianRFF(nn.Module):
    def __init__(self, embed_dim: int, scale: float = 30.0) -> None:
        super().__init__()
        self.rff_weights = nn.Parameter(
            torch.randn(embed_dim // 2) * scale,
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = x[:, None] * self.rff_weights[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class _Dense(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(x)[..., None, None]


class _ScoreNetCore(nn.Module):
    """3-channel U-Net core whose forward accepts a pre-computed class embedding.

    Stored as ``self.score_net`` inside :class:`CondUNetScoreModel` so that
    ``state_dict`` keys match the existing ``cfg_score_model_*.pth`` checkpoints.
    """

    def __init__(
        self,
        marginal_prob_std,
        channels: List[int],
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.marginal_prob_std = marginal_prob_std

        self.embed = nn.Sequential(
            _GaussianRFF(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        # Encoder
        self.conv1 = nn.Conv2d(3, channels[0], 3, stride=1, padding=1, bias=False)
        self.dense1 = _Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1, bias=False)
        self.dense2 = _Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1, bias=False)
        self.dense3 = _Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])

        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1, bias=False)
        self.dense4 = _Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoder
        self.tconv4 = nn.ConvTranspose2d(
            channels[3], channels[2], 3, stride=2, padding=1, output_padding=1, bias=False
        )
        self.dense5 = _Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])

        self.tconv3 = nn.ConvTranspose2d(
            channels[2] + channels[2], channels[1], 3, stride=2, padding=1, output_padding=1, bias=False
        )
        self.dense6 = _Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])

        self.tconv2 = nn.ConvTranspose2d(
            channels[1] + channels[1], channels[0], 3, stride=2, padding=1, output_padding=1, bias=False
        )
        self.dense7 = _Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])

        self.tconv1 = nn.ConvTranspose2d(
            channels[0] + channels[0], 3, 3, stride=1, padding=1, bias=False
        )

        self.act = lambda x: x * torch.sigmoid(x)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, class_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        embed = self.act(self.embed(t))
        if class_emb is not None:
            embed = embed + class_emb

        h1 = self.act(self.gnorm1(self.conv1(x) + self.dense1(embed)))
        h2 = self.act(self.gnorm2(self.conv2(h1) + self.dense2(embed)))
        h3 = self.act(self.gnorm3(self.conv3(h2) + self.dense3(embed)))
        h4 = self.act(self.gnorm4(self.conv4(h3) + self.dense4(embed)))

        h = self.act(self.tgnorm4(self.tconv4(h4) + self.dense5(embed)))
        h = self.act(self.tgnorm3(self.tconv3(torch.cat([h, h3], dim=1)) + self.dense6(embed)))
        h = self.act(self.tgnorm2(self.tconv2(torch.cat([h, h2], dim=1)) + self.dense7(embed)))
        h = self.tconv1(torch.cat([h, h1], dim=1))

        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h


class CondUNetScoreModel(BaseScoreModel):
    """Conditional U-Net score model for 3-channel 32×32 images.

    Extends the base U-Net with a learnable class embedding compatible with
    classifier-free guidance (CFG).  A special *null token* (index
    ``n_classes``) acts as the unconditional class ``∅``.

    The ``state_dict`` layout is identical to the ``ConditionalScoreNet``
    defined inline in ``cfg_diffusion_models_SVHN.ipynb``, so existing
    ``cfg_score_model_*.pth`` checkpoints load without modification.

    Parameters
    ----------
    marginal_prob_std : callable
        ``sigma_t(t)`` — standard deviation of the perturbation kernel.
    n_classes : int
        Number of real classes (10 for SVHN digits 0–9).
    channels : list of int
        U-Net channel widths.
    embed_dim : int
        Dimension of time and class embedding vectors.
    null_token : int, optional
        Index of the null token ``∅``.  Defaults to ``n_classes``.
    """

    def __init__(
        self,
        marginal_prob_std,
        n_classes: int = 10,
        channels: List[int] = None,
        embed_dim: int = 256,
        null_token: Optional[int] = None,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]

        self.null_token: int = n_classes if null_token is None else null_token

        # U-Net core — stored as `score_net` to match checkpoint key prefix
        self.score_net = _ScoreNetCore(marginal_prob_std, channels, embed_dim)

        # Class conditioning layers
        self.class_embed = nn.Embedding(n_classes + 1, embed_dim)
        self.class_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        class_label: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Noisy images ``(B, 3, H, W)``.
        t : torch.Tensor
            Diffusion times ``(B,)``.
        class_label : torch.Tensor, optional
            Integer labels ``(B,)``.  Pass ``None`` to use the null token
            (unconditional forward pass for CFG).
        """
        if class_label is None:
            class_label = torch.full(
                (x.shape[0],), self.null_token, dtype=torch.long, device=x.device
            )
        class_emb = self.class_proj(self.class_embed(class_label))
        return self.score_net(x, t, class_emb=class_emb)
