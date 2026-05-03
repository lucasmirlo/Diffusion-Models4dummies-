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
    VEProcess, VPProcess,
    LinearSchedule, CosineSchedule, ExponentialSchedule,
    EulerMaruyamaSampler, PredictorCorrectorSampler, ProbabilityFlowODESampler,
    GenerativeDiffusionModel, CondUNetScoreModel
)

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS      = 200
BATCH_SIZE  = 64
LR          = 1e-3
N_STEPS     = 500        # reverse integration steps at sampling time
N_IMAGES    = 16         # images per permutation (displayed as 4x4 grid)
IMG_SHAPE   = (3, 32, 32)
DATA_PATH   = Path('digits_true_color_AI/data/train_32x32.mat')
CKPT_DIR    = Path('color_digits_cond_checkpoints')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
CKPT_DIR.mkdir(exist_ok=True)



# ── Dataset ───────────────────────────────────────────────────────────────────
def load_svhn(path: Path) -> TensorDataset:
    """Load SVHN .mat file into a TensorDataset with images in [0, 1], (N, C, H, W)."""
    mat = scipy.io.loadmat(path)
    X = np.transpose(mat['X'], (3, 2, 0, 1)).astype(np.float32) / 255.0
    y = mat['y'].flatten().astype(np.int64)
    y[y == 10] = 0
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


dataset = load_svhn(DATA_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
print(f'SVHN: {len(dataset)} images  |  shape {dataset[0][0].shape}')


# ── Registries ────────────────────────────────────────────────────────────────
PROCESSES = {

    'VP-Exponential': VPProcess(schedule=ExponentialSchedule()),
}


# ── Training ──────────────────────────────────────────────────────────────────
def train_model(process, net: CondUNetScoreModel, ckpt_path: Path, T: float = 1.0) -> None:
    """Score-matching training loop; saves checkpoint on completion."""
    optimizer = optim.Adam(net.parameters(), lr=LR)
    net.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for (x, labels) in loader:
            x, labels = x.to(device), labels.to(device)
            mask = torch.rand(x.shape[0], device=device) < 0.1
            labels[mask] = net.null_token
            t = torch.rand(x.shape[0], device=device) * (1.0 - 1e-5) + 1e-5
            x_t, noise = process.perturb(x, t)
            view_shape = [x.shape[0]] + [1] * (x.dim() - 1)
            sigma = process.sigma_t(t.view(*view_shape))
            score = net(x_t, t, class_label=labels)
            per_sample = torch.sum((sigma * score + noise) ** 2, dim=list(range(1, x.dim())))
            loss = per_sample.mean()
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
    net = CondUNetScoreModel(marginal_prob_std=process.sigma_t).to(device)
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