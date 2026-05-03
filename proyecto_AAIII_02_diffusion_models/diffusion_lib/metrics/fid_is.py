# -*- coding: utf-8 -*-
"""Fréchet Inception Distance (FID) and Inception Score (IS)."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights

# ── Shared InceptionV3 loader ────────────────────────────────────────────────

_inception_cache: nn.Module | None = None


def _get_inception(device: torch.device) -> nn.Module:
    global _inception_cache
    if _inception_cache is None:
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
        model.eval()
        _inception_cache = model
    return _inception_cache.to(device)


def _preprocess(images: torch.Tensor) -> torch.Tensor:
    """Resize to 299×299 and normalise to ImageNet stats."""
    images = images.float()
    if images.shape[1] == 1:        # grayscale → RGB
        images = images.repeat(1, 3, 1, 1)
    images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
    images = torch.clamp(images, 0.0, 1.0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


@torch.no_grad()
def _extract_features(
    images: torch.Tensor,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pool3 features [N, 2048] and softmax probs [N, 1000]."""
    model = _get_inception(device)
    pool3_feats, softmax_probs = [], []

    for start in range(0, len(images), batch_size):
        batch = _preprocess(images[start: start + batch_size]).to(device)

        pool3_out: dict = {}

        def _hook(module, input, output):
            pool3_out["feat"] = output.squeeze(-1).squeeze(-1)

        hook = model.avgpool.register_forward_hook(_hook)
        logits = model(batch)
        hook.remove()

        if isinstance(logits, tuple):   # aux outputs present
            logits = logits[0]

        pool3_feats.append(pool3_out["feat"].cpu())
        softmax_probs.append(torch.softmax(logits, dim=1).cpu())

    return torch.cat(pool3_feats, dim=0), torch.cat(softmax_probs, dim=0)


# ── FID ─────────────────────────────────────────────────────────────────────

def compute_fid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 32,
) -> float:
    """Fréchet Inception Distance between two image sets.

    FID = ‖μ_r − μ_f‖² + Tr(Σ_r + Σ_f − 2 (Σ_r Σ_f)^½)

    Lower FID → generated images are closer to the real distribution.

    Args:
        real_images: Tensor [N, C, H, W] with pixel values in [0, 1].
        fake_images: Tensor [M, C, H, W] with pixel values in [0, 1].
        device:      Device for InceptionV3 inference.
        batch_size:  Mini-batch size for feature extraction.

    Returns:
        fid: Scalar float (lower is better).
    """
    from scipy.linalg import sqrtm

    feats_r, _ = _extract_features(real_images, device, batch_size)
    feats_f, _ = _extract_features(fake_images, device, batch_size)

    mu_r = feats_r.mean(0).numpy()
    mu_f = feats_f.mean(0).numpy()
    sigma_r = np.cov(feats_r.numpy(), rowvar=False)
    sigma_f = np.cov(feats_f.numpy(), rowvar=False)

    diff = mu_r - mu_f
    cov_sqrt, _ = sqrtm(sigma_r @ sigma_f, disp=False)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    fid = float(diff @ diff + np.trace(sigma_r + sigma_f - 2.0 * cov_sqrt))
    return fid


# ── IS ──────────────────────────────────────────────────────────────────────

def compute_is(
    images: torch.Tensor,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 32,
    n_splits: int = 10,
) -> tuple[float, float]:
    """Inception Score of a set of generated images.

    IS = exp( E_x[ KL( p(y|x) ‖ p(y) ) ] )

    where  p(y|x) is the InceptionV3 class distribution for image x
    and    p(y) = E_x[p(y|x)]  is the marginal class distribution.

    Higher IS → generated images are both diverse and class-discriminative.

    Args:
        images:     Tensor [N, C, H, W] with pixel values in [0, 1].
        device:     Device for InceptionV3 inference.
        batch_size: Mini-batch size.
        n_splits:   Number of data splits for mean/std estimation.

    Returns:
        (is_mean, is_std): Mean and standard deviation of IS over splits.
    """
    _, probs = _extract_features(images, device, batch_size)
    probs = probs.numpy()
    n = len(probs)
    split_size = n // n_splits

    scores = []
    for k in range(n_splits):
        part = probs[k * split_size: (k + 1) * split_size]
        p_y = part.mean(axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-8) - np.log(p_y + 1e-8))
        scores.append(float(np.exp(kl.sum(axis=1).mean())))

    return float(np.mean(scores)), float(np.std(scores))
