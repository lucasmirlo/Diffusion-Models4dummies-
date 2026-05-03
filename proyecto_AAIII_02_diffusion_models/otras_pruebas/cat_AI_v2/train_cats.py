# -*- coding: utf-8 -*-
"""
Diffusion Model — Custom Cat Dataset (64x64)
Ejecutar desde la carpeta del proyecto:
    python train_cats.py
"""

import sys
import os
from pathlib import Path

# ── Rutas ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(r'C:\Users\lucan\Desktop\practicas3segundocuatri\apaut3\proyecto\proyecto_AAIII_02_diffusion_models')
os.chdir(PROJECT_DIR)

for path_entry in (PROJECT_DIR, PROJECT_DIR.parent):
    if str(path_entry) not in sys.path:
        sys.path.append(str(path_entry))

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
from functools import partial

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torchvision.transforms import ToTensor
from PIL import Image
import tqdm

import diffusion_process as dfp
import cat_AI_v2.score_model as sm

# ── Config ────────────────────────────────────────────────────────────────────
CAT_FOLDER  = 'cat_AI_v2/cats'
BATCH_SIZE  = 256
LR          = 1e-3
N_EPOCHS    = 2000
SAVE_EVERY  = 100          # guardar checkpoint cada N epochs
CHECKPOINT  = 'cat64_diffusion_{:d}_epochs.pth'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Usando device: {device}')

# ── Dataset ───────────────────────────────────────────────────────────────────
class CatFolderDataset(Dataset):
    def __init__(self, folder_path):
        transform = ToTensor()
        paths = sorted(Path(folder_path).glob('*.jpg'))
        print(f'Cargando {len(paths)} imágenes en RAM...')
        self.images = [transform(Image.open(p).convert('RGB')) for p in paths]
        print('Listo.')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], 0


data_train = CatFolderDataset(CAT_FOLDER)
print(f'Imágenes: {len(data_train)}  |  Shape: {data_train[0][0].shape}')

data_loader = DataLoader(
    data_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

# ── Proceso de difusión ───────────────────────────────────────────────────────
sigma = 25.0

def bm_drift_coefficient(x_t, t):
    return torch.zeros_like(x_t)

def bm_diffusion_coefficient(t, sigma=sigma):
    return sigma ** t

def bm_mu_t(x_0, t):
    return x_0

def bm_sigma_t(t, sigma=sigma):
    return torch.sqrt(0.5 * (sigma ** (2 * t) - 1.0) / np.log(sigma))

diffusion_process = dfp.GaussianDiffussionProcess(
    drift_coefficient=bm_drift_coefficient,
    diffusion_coefficient=bm_diffusion_coefficient,
    mu_t=bm_mu_t,
    sigma_t=bm_sigma_t,
)

# ── Score model ───────────────────────────────────────────────────────────────
score_model = torch.nn.DataParallel(
    sm.ScoreNet(marginal_prob_std=partial(bm_sigma_t, sigma=sigma))
)
score_model = score_model.to(device)
print(f'Modelo en: {next(score_model.parameters()).device}')

# ── Training loop ─────────────────────────────────────────────────────────────
optimizer = Adam(score_model.parameters(), lr=LR)

if __name__ == '__main__':
    tqdm_epoch = tqdm.trange(N_EPOCHS, desc='Entrenando')

    for epoch in tqdm_epoch:
        avg_loss = 0.0
        num_items = 0

        for x, _ in data_loader:
            x = x.to(device)
            loss = diffusion_process.loss_function(score_model, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]

        tqdm_epoch.set_description(f'Loss: {avg_loss / num_items:.5f}')

        if (epoch + 1) % SAVE_EVERY == 0:
            path = CHECKPOINT.format(epoch + 1)
            torch.save(score_model.state_dict(), path)
            print(f'\nCheckpoint guardado: {path}')

    # Guardado final
    final_path = CHECKPOINT.format(N_EPOCHS)
    torch.save(score_model.state_dict(), final_path)
    print(f'\nEntrenamiento completado. Modelo guardado en: {final_path}')
