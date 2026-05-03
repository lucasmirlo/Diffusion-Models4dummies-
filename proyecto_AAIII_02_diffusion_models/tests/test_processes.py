# -*- coding: utf-8 -*-
"""Tests for diffusion_lib.processes (VEProcess and VPProcess)."""

import torch
import pytest

from diffusion_lib import VEProcess, VPProcess, LinearSchedule


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ve():
    return VEProcess(sigma=25.0, T=1.0)


@pytest.fixture
def vp():
    return VPProcess(schedule=LinearSchedule(), T=1.0)


# ---------------------------------------------------------------------------
# VEProcess
# ---------------------------------------------------------------------------

class TestVEProcess:

    def test_sigma_t_monotone(self, ve):
        """σ_t must grow monotonically: the VE process explodes variance."""
        t = torch.linspace(0.01, 1.0, 50)
        sigma = ve.sigma_t(t)
        assert (sigma[1:] > sigma[:-1]).all(), "VE σ_t is not monotonically increasing"

    def test_mu_t_equals_x0(self, ve):
        """VE has zero drift: μ_t(x_0, t) = x_0 for all t."""
        x0 = torch.randn(4, 1, 8, 8)
        t = torch.rand(4, 1, 1, 1)
        mu = ve.mu_t(x0, t)
        assert torch.allclose(mu, x0), "VE μ_t should equal x_0 (no drift)"

    def test_perturb_shape(self, ve):
        """perturb() must return tensors with the same shape as x_0."""
        x0 = torch.randn(4, 1, 8, 8)
        t = torch.rand(4) * 0.99 + 0.01
        x_t, noise = ve.perturb(x0, t)
        assert x_t.shape == x0.shape
        assert noise.shape == x0.shape

    def test_prior_sample_shape(self, ve):
        """prior_sample must return a tensor with the requested shape."""
        shape = (8, 1, 8, 8)
        sample = ve.prior_sample(shape, device=torch.device("cpu"))
        assert sample.shape == torch.Size(shape)

    def test_diffusion_coefficient_positive(self, ve):
        """g(t) = σ^t must be strictly positive for all t ∈ (0, 1]."""
        t = torch.linspace(0.01, 1.0, 50)
        g = ve.diffusion_coefficient(t)
        assert (g > 0).all()

    def test_drift_coefficient_zero(self, ve):
        """VE drift is identically zero."""
        x = torch.randn(4, 1, 8, 8)
        t = torch.rand(4)
        drift = ve.drift_coefficient(x, t)
        assert torch.all(drift == 0)


# ---------------------------------------------------------------------------
# VPProcess
# ---------------------------------------------------------------------------

class TestVPProcess:

    def test_sigma_t_monotone(self, vp):
        """σ_t must grow monotonically: more noise is injected over time."""
        t = torch.linspace(0.01, 1.0, 50)
        sigma = vp.sigma_t(t)
        assert (sigma[1:] > sigma[:-1]).all(), "VP σ_t is not monotonically increasing"

    def test_sigma_t_boundary(self, vp):
        """σ_t(0) ≈ 0  and  σ_t(1) ≈ 1 for the VP process."""
        t_start = torch.tensor([1e-4])
        t_end = torch.tensor([1.0])
        assert vp.sigma_t(t_start).item() < 0.05, "VP σ_t(0) should be near 0"
        assert vp.sigma_t(t_end).item() > 0.95,   "VP σ_t(1) should be near 1"

    def test_mu_t_decays(self, vp):
        """μ_t(x_0, t) = α_t · x_0 must decay toward 0 as t → 1."""
        x0 = torch.ones(1, 1, 1, 1)
        t_early = torch.tensor([[[[0.01]]]])
        t_late  = torch.tensor([[[[1.0 ]]]])
        mu_early = vp.mu_t(x0, t_early).item()
        mu_late  = vp.mu_t(x0, t_late).item()
        assert mu_early > mu_late, "VP μ_t should decay with t"
        assert mu_late < 0.1,     "VP μ_t(1) should be close to 0"

    def test_perturb_shape(self, vp):
        """perturb() must return tensors with the same shape as x_0."""
        x0 = torch.randn(4, 3, 8, 8)
        t = torch.rand(4) * 0.99 + 0.01
        x_t, noise = vp.perturb(x0, t)
        assert x_t.shape == x0.shape
        assert noise.shape == x0.shape

    def test_prior_sample_shape(self, vp):
        """prior_sample must return a tensor with the requested shape."""
        shape = (8, 3, 8, 8)
        sample = vp.prior_sample(shape, device=torch.device("cpu"))
        assert sample.shape == torch.Size(shape)
