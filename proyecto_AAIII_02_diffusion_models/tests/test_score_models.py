# -*- coding: utf-8 -*-
"""Tests for diffusion_lib.score_models."""

import pytest
import torch

from diffusion_lib import (
    VEProcess,
    UNetScoreModel,
    CondUNetScoreModel,
    CFGWrapper,
)


def _ve_sigma_t():
    process = VEProcess(sigma=25.0)
    return process.sigma_t


# ---------------------------------------------------------------------------
# UNetScoreModel (unconditional, 1-channel 28×28)
# ---------------------------------------------------------------------------

class TestUNetScoreModel:
    def setup_method(self):
        self.sigma_t = _ve_sigma_t()
        self.model = UNetScoreModel(marginal_prob_std=self.sigma_t)
        self.model.eval()

    def test_output_shape(self):
        x = torch.randn(4, 1, 28, 28)
        t = torch.rand(4) * 0.99 + 0.01
        with torch.no_grad():
            out = self.model(x, t)
        assert out.shape == x.shape

    def test_ignores_class_label(self):
        x = torch.randn(2, 1, 28, 28)
        t = torch.rand(2) * 0.99 + 0.01
        y = torch.zeros(2, dtype=torch.long)
        with torch.no_grad():
            out_no_label = self.model(x, t, class_label=None)
            out_with_label = self.model(x, t, class_label=y)
        assert torch.allclose(out_no_label, out_with_label)


# ---------------------------------------------------------------------------
# CondUNetScoreModel (conditional, 3-channel 32×32)
# ---------------------------------------------------------------------------

class TestCondUNetScoreModel:
    def setup_method(self):
        self.sigma_t = _ve_sigma_t()
        self.model = CondUNetScoreModel(
            marginal_prob_std=self.sigma_t, n_classes=10
        )
        self.model.eval()

    def test_output_shape_with_label(self):
        x = torch.randn(4, 3, 32, 32)
        t = torch.rand(4) * 0.99 + 0.01
        y = torch.randint(0, 10, (4,))
        with torch.no_grad():
            out = self.model(x, t, class_label=y)
        assert out.shape == x.shape

    def test_output_shape_no_label(self):
        x = torch.randn(4, 3, 32, 32)
        t = torch.rand(4) * 0.99 + 0.01
        with torch.no_grad():
            out = self.model(x, t, class_label=None)
        assert out.shape == x.shape

    def test_cond_differs_from_uncond(self):
        torch.manual_seed(0)
        x = torch.randn(2, 3, 32, 32)
        t = torch.full((2,), 0.5)
        y = torch.zeros(2, dtype=torch.long)
        with torch.no_grad():
            s_uncond = self.model(x, t, class_label=None)
            s_cond = self.model(x, t, class_label=y)
        assert not torch.allclose(s_uncond, s_cond)


# ---------------------------------------------------------------------------
# CFGWrapper
# ---------------------------------------------------------------------------

class TestCFGWrapper:
    def setup_method(self):
        self.sigma_t = _ve_sigma_t()
        self.cond_model = CondUNetScoreModel(
            marginal_prob_std=self.sigma_t, n_classes=10
        )
        self.cond_model.eval()

    def test_output_shape(self):
        wrapper = CFGWrapper(self.cond_model, class_label=3, cfg_scale=3.0)
        x = torch.randn(4, 3, 32, 32)
        t = torch.rand(4) * 0.99 + 0.01
        with torch.no_grad():
            out = wrapper(x, t)
        assert out.shape == x.shape

    def test_zero_scale_equals_uncond(self):
        wrapper = CFGWrapper(self.cond_model, class_label=5, cfg_scale=0.0)
        x = torch.randn(2, 3, 32, 32)
        t = torch.full((2,), 0.3)
        with torch.no_grad():
            cfg_out = wrapper(x, t)
            uncond_out = self.cond_model(x, t, class_label=None)
        assert torch.allclose(cfg_out, uncond_out)

    def test_to_and_eval_return_self(self):
        wrapper = CFGWrapper(self.cond_model, class_label=7, cfg_scale=2.0)
        assert wrapper.eval() is wrapper
        assert wrapper.train() is wrapper
        assert wrapper.to("cpu") is wrapper
