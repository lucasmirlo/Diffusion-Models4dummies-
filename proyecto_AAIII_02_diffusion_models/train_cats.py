# -*- coding: utf-8 -*-
"""
Diffusion Model — Custom Cat Dataset (64x64)
Dos fases de entrenamiento con checkpoints en escala logarítmica:
  · Fase 1 — lr=1e-3, 1500 épocas  (desde cero)
  · Fase 2 — lr=1e-4, 1500 épocas  (fine-tuning desde checkpoint final Fase 1)

Ejecutar desde la carpeta raíz del proyecto:
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
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torchvision.transforms import ToTensor
from PIL import Image
import tqdm

from diffusion_lib import (
    VPProcess,
    ExponentialSchedule,
    EulerMaruyamaSampler,
    GenerativeDiffusionModel,
    UNetScoreModelColor,
)

# ── Config ────────────────────────────────────────────────────────────────────
CAT_FOLDER = 'otras_pruebas/cat_AI_v2/cats'
BATCH_SIZE = 256
N_EPOCHS   = 1500

# Checkpoints en escala logarítmica: ~15 épocas distribuidas de log(1) a log(1500),
# más el epoch final garantizado.
_log_ckpts = np.unique(
    np.round(np.logspace(0, np.log10(N_EPOCHS), 15)).astype(int)
)
LOG_CHECKPOINTS = set(_log_ckpts.tolist()) | {N_EPOCHS}

CKPT_DIR = Path('otras_pruebas/cat_AI_v2/checkpoints')
CKPT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
print(f'Épocas con checkpoint: {sorted(LOG_CHECKPOINTS)}')


# ── Dataset ───────────────────────────────────────────────────────────────────
class CatFolderDataset(Dataset):
    def __init__(self, folder_path: str):
        transform = ToTensor()
        paths = sorted(Path(folder_path).glob('*.jpg'))
        if not paths:
            raise FileNotFoundError(
                f'No se encontraron imágenes .jpg en {folder_path}'
            )
        print(f'Cargando {len(paths)} imágenes en RAM...')
        self.images = [transform(Image.open(p).convert('RGB')) for p in paths]
        print(f'Dataset listo. Shape: {self.images[0].shape}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], 0


# ── Función de entrenamiento ──────────────────────────────────────────────────
def train_phase(
    score_model: torch.nn.Module,
    data_loader: DataLoader,
    lr: float,
    n_epochs: int,
    phase: int,
    checkpoint_epochs: set,
) -> Path:
    """
    Entrena score_model durante n_epochs con Adam(lr=lr).
    Guarda checkpoints en las épocas indicadas por checkpoint_epochs.
    Devuelve la ruta del último checkpoint guardado.
    """
    vp_process      = VPProcess(schedule=ExponentialSchedule())
    diffusion_model = GenerativeDiffusionModel(
        vp_process, EulerMaruyamaSampler(), score_model, DEVICE
    )
    optimizer  = Adam(score_model.parameters(), lr=lr)
    last_ckpt  = None

    print(f'\n{"="*60}')
    print(f' FASE {phase}  |  lr={lr:.0e}  |  {n_epochs} épocas')
    print(f'{"="*60}')

    tqdm_epoch = tqdm.trange(1, n_epochs + 1, desc=f'Fase {phase}')
    for epoch in tqdm_epoch:
        avg_loss  = 0.0
        num_items = 0

        for x, _ in data_loader:
            x = x.to(DEVICE)
            # FIX 1: compute_loss (GenerativeDiffusionModel no tiene loss_function)
            loss = diffusion_model.compute_loss(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss  += loss.item() * x.shape[0]
            num_items += x.shape[0]

        epoch_loss = avg_loss / num_items
        tqdm_epoch.set_description(f'Fase {phase} | loss={epoch_loss:.5f}')

        if epoch in checkpoint_epochs:
            fname = f'cat64_VP-Exp_phase{phase}_lr{lr:.0e}_ep{epoch:04d}.pth'
            ckpt_path = CKPT_DIR / fname
            torch.save(score_model.state_dict(), ckpt_path)
            tqdm_epoch.write(f'  [ckpt] época {epoch:4d} → {fname}')
            last_ckpt = ckpt_path

    print(f'\nFase {phase} completada. Último checkpoint: {last_ckpt}')
    return last_ckpt


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # Dataset
    dataset    = CatFolderDataset(CAT_FOLDER)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == 'cuda'),
    )

    # Proceso VP-Exponencial (necesario para construir el score model con sigma_t)
    vp = VPProcess(schedule=ExponentialSchedule())

    # ── FASE 1: lr = 1e-3, entrenamiento desde cero ───────────────────────────
    score_model = UNetScoreModelColor(
        marginal_prob_std=vp.sigma_t   # FIX 2: argumento obligatorio
    ).to(DEVICE)

    ckpt_phase1 = train_phase(
        score_model       = score_model,
        data_loader       = dataloader,
        lr                = 1e-3,
        n_epochs          = N_EPOCHS,
        phase             = 1,
        checkpoint_epochs = LOG_CHECKPOINTS,
    )

    # ── FASE 2: lr = 1e-4, fine-tuning desde el checkpoint final de Fase 1 ───
    print(f'\nCargando pesos Fase 1 desde: {ckpt_phase1}')
    score_model.load_state_dict(
        torch.load(ckpt_phase1, map_location=DEVICE, weights_only=True)
    )

    ckpt_phase2 = train_phase(
        score_model       = score_model,
        data_loader       = dataloader,
        lr                = 1e-4,
        n_epochs          = N_EPOCHS,
        phase             = 2,
        checkpoint_epochs = LOG_CHECKPOINTS,
    )

    print('\n' + '='*60)
    print(' Entrenamiento completo')
    print(f'  Fase 1 final : {ckpt_phase1}')
    print(f'  Fase 2 final : {ckpt_phase2}')
    print('='*60)
