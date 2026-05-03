# -*- coding: utf-8 -*-
"""Tests for diffusion_lib.schedules (LinearSchedule and CosineSchedule)."""

import torch
import pytest

from diffusion_lib import LinearSchedule, CosineSchedule


# ---------------------------------------------------------------------------
# LinearSchedule
# ---------------------------------------------------------------------------

class TestLinearSchedule:

    def test_beta_at_boundaries(self):
        """β(0) = β_min  and  β(1) = β_max."""
        sched = LinearSchedule(beta_min=0.1, beta_max=20.0)
        assert torch.isclose(sched.beta(torch.tensor(0.0)), torch.tensor(0.1), atol=1e-5)
        assert torch.isclose(sched.beta(torch.tensor(1.0)), torch.tensor(20.0), atol=1e-5)

    def test_beta_monotone(self):
        """β(t) must be strictly increasing for β_min < β_max."""
        sched = LinearSchedule()
        t = torch.linspace(0.0, 1.0, 50)
        beta = sched.beta(t)
        assert (beta[1:] > beta[:-1]).all()

    def test_integral_beta_at_zero(self):
        """B(0) = 0 (no noise accumulated at the start)."""
        sched = LinearSchedule()
        assert torch.isclose(
            sched.integral_beta(torch.tensor(0.0)), torch.tensor(0.0), atol=1e-6
        )

    def test_integral_beta_monotone(self):
        """B(t) must be strictly increasing."""
        sched = LinearSchedule()
        t = torch.linspace(0.0, 1.0, 50)
        B = sched.integral_beta(t)
        assert (B[1:] > B[:-1]).all()

    def test_integral_beta_matches_beta(self):
        """Numerical integral of β should match integral_beta up to step error."""
        sched = LinearSchedule()
        t_vals = torch.linspace(0.0, 1.0, 1000)
        dt = t_vals[1] - t_vals[0]
        numerical = (sched.beta(t_vals[:-1]) * dt).cumsum(0)[-1].item()
        analytic  = sched.integral_beta(torch.tensor(1.0)).item()
        assert abs(numerical - analytic) < 0.05


# ---------------------------------------------------------------------------
# CosineSchedule
# ---------------------------------------------------------------------------

class TestCosineSchedule:

    def test_alpha_bar_at_zero(self):
        """ᾱ(0) must be exactly 1 by construction."""
        sched = CosineSchedule()
        alpha_0 = sched.alpha_bar(torch.tensor(0.0)).item()
        assert abs(alpha_0 - 1.0) < 1e-5, f"ᾱ(0) = {alpha_0}, expected ≈ 1"

    def test_alpha_bar_at_one(self):
        """ᾱ(1) must be close to 0 (almost all signal removed)."""
        sched = CosineSchedule()
        alpha_1 = sched.alpha_bar(torch.tensor(1.0)).item()
        assert alpha_1 < 0.01, f"ᾱ(1) = {alpha_1}, expected ≈ 0"

    def test_alpha_bar_monotone_decreasing(self):
        """ᾱ(t) must decrease monotonically from 1 to 0."""
        sched = CosineSchedule()
        t = torch.linspace(0.0, 1.0, 50)
        alpha = sched.alpha_bar(t)
        assert (alpha[1:] < alpha[:-1]).all()

    def test_integral_beta_at_zero(self):
        """B(0) = -log ᾱ(0) = 0."""
        sched = CosineSchedule()
        B_0 = sched.integral_beta(torch.tensor(0.0)).item()
        assert abs(B_0) < 1e-4, f"B(0) = {B_0}, expected ≈ 0"

    def test_integral_beta_monotone(self):
        """B(t) must be strictly increasing."""
        sched = CosineSchedule()
        t = torch.linspace(0.01, 1.0, 50)
        B = sched.integral_beta(t)
        assert (B[1:] > B[:-1]).all()

    def test_beta_positive(self):
        """β(t) must be strictly positive for all t ∈ (0, 1)."""
        sched = CosineSchedule()
        t = torch.linspace(0.01, 0.99, 50)
        beta = sched.beta(t)
        assert (beta > 0).all()
