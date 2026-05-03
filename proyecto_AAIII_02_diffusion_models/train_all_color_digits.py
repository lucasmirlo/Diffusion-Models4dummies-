# -*- coding: utf-8 -*-
"""
Train + sample all (process x sampler) permutations on the SVHN colour-digits dataset.

Processes : VE-Brownian | VP-Linear | VP-Cosine | VP-Exponential        (4)
Samplers  : Euler-Maruyama | Predictor-Corrector | Probability-Flow ODE (3)
------------------------------------------------------------------------
Training runs : 4   (sampler has no effect at training time)
Sampling runs : 4 x 3 = 12

Outputs:
  color_digits_checkpoints/color_digits_{process}.pth   -- one per process
  color_digits_samples/{process}_{sampler}.png          -- one per permutation

Usage:
  python train_all_color_digits.py

Re-run safely: existing checkpoints are loaded, not retrained.
"""

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
os.chdir(PROJECT_DIR)
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import scipy.io
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid

from diffusion_lib import (
    UNetScoreModelColor as ScoreNetColor,
    VEProcess, VPProcess,
    LinearSchedule, CosineSchedule, ExponentialSchedule,
    EulerMaruyamaSampler, PredictorCorrectorSampler, ProbabilityFlowODESampler,
    GenerativeDiffusionModel,
)

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS      = 100
BATCH_SIZE  = 64
LR          = 1e-3
N_STEPS     = 500        # reverse integration steps at sampling time
N_IMAGES    = 16         # images per permutation (displayed as 4x4 grid)
IMG_SHAPE   = (3, 32, 32)
DATA_PATH   = Path('digits_true_color_AI/data/train_32x32.mat')
CKPT_DIR    = Path('color_digits_checkpoints')
SAMPLES_DIR = Path('Figuras')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
CKPT_DIR.mkdir(exist_ok=True)
SAMPLES_DIR.mkdir(exist_ok=True)


# ── Dataset ───────────────────────────────────────────────────────────────────
def load_svhn(path: Path) -> TensorDataset:
    """Load SVHN .mat file into a TensorDataset with images in [0, 1], (N, C, H, W)."""
    mat = scipy.io.loadmat(path)
    X = np.transpose(mat['X'], (3, 2, 0, 1)).astype(np.float32) / 255.0
    return TensorDataset(torch.from_numpy(X))


dataset = load_svhn(DATA_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
print(f'SVHN: {len(dataset)} images  |  shape {dataset[0][0].shape}')


# ── Registries ────────────────────────────────────────────────────────────────
PROCESSES = {
    'VE-Brownian':    VEProcess(sigma=25.0),
    'VP-Linear':      VPProcess(schedule=LinearSchedule()),
    'VP-Cosine':      VPProcess(schedule=CosineSchedule()),
    'VP-Exponential': VPProcess(schedule=ExponentialSchedule()),
}

SAMPLERS = {
    'EM':  EulerMaruyamaSampler(),
    'PC':  PredictorCorrectorSampler(),
    'ODE': ProbabilityFlowODESampler(),
}


# ── Training ──────────────────────────────────────────────────────────────────
def train_model(process, net: ScoreNetColor, ckpt_path: Path, T: float = 1.0) -> None:
    """Score-matching training loop; saves checkpoint on completion."""
    gm = GenerativeDiffusionModel(process, EulerMaruyamaSampler(), net, device, T=T)
    optimizer = optim.Adam(net.parameters(), lr=LR)
    net.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for (x,) in loader:
            loss = gm.compute_loss(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(loader)
        print(f'  Epoch {epoch + 1:3d}/{EPOCHS}  loss={avg:.4f}')
    torch.save(net.state_dict(), ckpt_path)
    print(f'  Checkpoint saved -> {ckpt_path}')


# ── Train or load each process ────────────────────────────────────────────────
trained_nets: dict[str, tuple] = {}

for proc_name, process in PROCESSES.items():
    ckpt_path = CKPT_DIR / f'color_digits_{proc_name}.pth'
    net = ScoreNetColor(marginal_prob_std=process.sigma_t).to(device)
    T = 0.999 if proc_name == 'VP-Cosine' else 1.0

    print(f'\n[{proc_name}]')
    if ckpt_path.exists():
        print(f'  Loading checkpoint: {ckpt_path}')
        net.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
    else:
        print(f'  Training for {EPOCHS} epochs ...')
        train_model(process, net, ckpt_path, T=T)

    trained_nets[proc_name] = (process, net, T)


# ── Sample all 12 permutations ────────────────────────────────────────────────
def save_grid(samples: torch.Tensor, title: str, out_path: Path) -> None:
    """Save a (N, C, H, W) tensor as a PNG image grid."""
    grid = make_grid(samples.clamp(0, 1).cpu(), nrow=4)
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.title(title, fontsize=11)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()


print('\n[Sampling] 4 processes x 3 samplers = 12 permutations')
results = []

for proc_name, (process, net, T) in trained_nets.items():
    for samp_name, sampler in SAMPLERS.items():
        tag = f'{proc_name}_{samp_name}'
        out_path = SAMPLES_DIR / f'{tag}.png'
        print(f'  {tag} ...', end=' ', flush=True)
        gm = GenerativeDiffusionModel(process, sampler, net, device, T=T)
        with torch.no_grad():
            samples = gm.sample(N_IMAGES, IMG_SHAPE, n_steps=N_STEPS)

        save_grid(samples, f'{proc_name} + {samp_name}', out_path)
        print(f'-> {out_path}')
        results.append((proc_name, samp_name, out_path))


# ── Summary table ─────────────────────────────────────────────────────────────
print('\n' + '─' * 62)
print(f'{"Process":<18}  {"Sampler":<6}  Output')
print('─' * 62)
for proc, samp, path in results:
    print(f'{proc:<18}  {samp:<6}  {path}')
print('─' * 62)
print(f'Done. {len(results)} images saved to {SAMPLES_DIR}/')
