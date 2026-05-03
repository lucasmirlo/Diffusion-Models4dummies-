# -*- coding: utf-8 -*-
"""Integration tests for GenerativeDiffusionModel (training loop + sampling)."""

import torch
import pytest

from diffusion_lib import (
    VEProcess,
    VPProcess,
    LinearSchedule,
    EulerMaruyamaSampler,
    GenerativeDiffusionModel,
    UNetScoreModel,
)


# ---------------------------------------------------------------------------
# Fixtures — tiny models so tests run fast on CPU
# ---------------------------------------------------------------------------

IMG_SHAPE = (1, 28, 28)
DEVICE    = torch.device("cpu")


@pytest.fixture
def ve_model():
    process     = VEProcess(sigma=25.0)
    score_model = UNetScoreModel(marginal_prob_std=process.sigma_t)
    sampler     = EulerMaruyamaSampler()
    return GenerativeDiffusionModel(process, sampler, score_model, device=DEVICE)


@pytest.fixture
def vp_model():
    schedule    = LinearSchedule()
    process     = VPProcess(schedule=schedule)
    score_model = UNetScoreModel(marginal_prob_std=process.sigma_t)
    sampler     = EulerMaruyamaSampler()
    return GenerativeDiffusionModel(process, sampler, score_model, device=DEVICE)


# ---------------------------------------------------------------------------
# compute_loss
# ---------------------------------------------------------------------------

class TestComputeLoss:

    def test_loss_is_scalar(self, ve_model):
        """compute_loss must return a scalar tensor."""
        x0   = torch.randn(4, *IMG_SHAPE)
        loss = ve_model.compute_loss(x0)
        assert loss.ndim == 0, "Loss should be a 0-dim scalar"

    def test_loss_is_positive(self, ve_model):
        """Score-matching loss is a sum of squares — always non-negative."""
        x0   = torch.randn(4, *IMG_SHAPE)
        loss = ve_model.compute_loss(x0)
        assert loss.item() >= 0.0

    def test_loss_decreases_after_one_step(self, vp_model):
        """After one gradient step the loss should decrease."""
        torch.manual_seed(42)
        x0        = torch.randn(8, *IMG_SHAPE)
        optimiser = torch.optim.Adam(
            vp_model.score_model.parameters(), lr=1e-3
        )

        loss_before = vp_model.compute_loss(x0)

        optimiser.zero_grad()
        loss_before.backward()
        optimiser.step()

        loss_after = vp_model.compute_loss(x0)
        assert loss_after.item() < loss_before.item(), (
            f"Loss did not decrease: {loss_before.item():.4f} → {loss_after.item():.4f}"
        )

    def test_loss_vp_is_scalar(self, vp_model):
        """Same scalar check for the VP variant."""
        x0   = torch.randn(4, *IMG_SHAPE)
        loss = vp_model.compute_loss(x0)
        assert loss.ndim == 0


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------

class TestSample:

    def test_sample_shape_ve(self, ve_model):
        """Sampled batch must have shape (n_images, *img_shape)."""
        n = 4
        out = ve_model.sample(n_images=n, img_shape=IMG_SHAPE, n_steps=10)
        assert out.shape == torch.Size([n, *IMG_SHAPE])

    def test_sample_shape_vp(self, vp_model):
        """Same shape check for the VP process."""
        n = 4
        out = vp_model.sample(n_images=n, img_shape=IMG_SHAPE, n_steps=10)
        assert out.shape == torch.Size([n, *IMG_SHAPE])

    def test_sample_is_finite(self, ve_model):
        """Sampled pixels must not contain NaN or Inf."""
        out = ve_model.sample(n_images=2, img_shape=IMG_SHAPE, n_steps=10)
        assert torch.isfinite(out).all(), "Sample contains NaN or Inf values"
