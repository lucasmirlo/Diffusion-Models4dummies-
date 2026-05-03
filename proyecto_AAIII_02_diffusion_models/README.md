# Proyecto AAIII — Diffusion Models
**Aprendizaje Automático III · Grado en Ciencia e Ingeniería de Datos · UAM**

Score-based generative modelling with stochastic differential equations (SDEs). The project implements, trains and evaluates diffusion models on MNIST and SVHN colour digits, covering VE/VP processes, multiple noise schedules, three samplers, classifier-free guidance and inpainting.

---

## Project structure

```
proyecto_AAIII_02_diffusion_models/
│
├── diffusion_lib/                  # Core Python package
│   ├── processes/                  # VEProcess, VPProcess
│   ├── schedules/                  # LinearSchedule, CosineSchedule, ExponentialSchedule
│   ├── samplers/                   # EulerMaruyama, PredictorCorrector, ProbabilityFlowODE, Imputation
│   ├── score_models/               # UNetScoreModel, UNetScoreModelColor, CondUNetScoreModel, CFGWrapper
│   ├── metrics/                    # BPD (bits-per-dim), FID
│   ├── utils/                      # Visualization helpers
│   ├── legacy/                     # Original trajectory-simulation API (backward compat.)
│   └── model.py                    # GenerativeDiffusionModel (train / sample / BPD)
│
├── project_AAIII_teamCode_lastName1_lastName2.ipynb   # Main project notebook
├── fid_ve_vp.ipynb                 # FID comparison: VE vs VP schedules
├── samplers_vp_exp.ipynb           # Sampler comparison on VP-Cosine
├── conditional_and_imputation.ipynb # CFG conditional generation + inpainting
├── diffusion_lib_use_cases.ipynb   # API usage examples
│
├── train_all_color_digits.py       # Train unconditional models on SVHN colour digits
├── train_all_color_digits_conditional.py  # Train conditional models (CFG)
├── train_cats.py                   # Train cat diffusion model (VP-Exp, 2-phase)
├── cats_metrics.ipynb              # FID / BPD / IS comparison: cat Phase 1 vs Phase 2
│
├── other_checkpoints/              # MNIST checkpoints (BM, VP-Linear/Cosine/Exp)
├── color_digits_checkpoints/       # Unconditional SVHN checkpoints
├── color_digits_cond_checkpoints/  # Conditional SVHN checkpoints
│
├── data/                           # MNIST dataset (auto-downloaded)
│   └── MNIST/raw/train_32x32.mat   # SVHN training data
│
├── Figuras/                        # All output figures (referenced by the LaTeX report)
├── otras_pruebas/                  # Exploratory notebooks (not part of the deliverable)
│   └── cat_AI_v2/                  # Cat dataset, images and exploration notebook
└── tests/                          # Unit tests for diffusion_lib
```

---

## Setup

```bash
# Requires Python >= 3.11
uv sync          # installs all dependencies from uv.lock
# or
pip install -e . --break-system-packages
```

---

## Notebooks — execution order

Run in this order to regenerate all figures in `Figuras/`:

| # | Notebook | Figures produced |
|---|----------|-----------------|
| 1 | `project_AAIII_teamCode_lastName1_lastName2.ipynb` | training curves, generation grids, BPD table |
| 2 | `fid_ve_vp.ipynb` | `fid_ve_vp_comparison.png` |
| 3 | `samplers_vp_exp.ipynb` | `samplers_vp_cos_comparison.png` |
| 4 | `conditional_and_imputation.ipynb` | `cfg_class_grid.png`, `cfg_ablation_w.png`, `cfg_process_comparison.png`, `imputation_masks.png`, `imputation_diversity.png` |
| 5 | `cats_metrics.ipynb` *(after `train_cats.py` finishes)* | `cats_generation_comparison.png`, `cats_metrics_comparison.png` |

---

## diffusion_lib — quick reference

```python
from diffusion_lib import (
    VEProcess, VPProcess,
    LinearSchedule, CosineSchedule, ExponentialSchedule,
    EulerMaruyamaSampler, PredictorCorrectorSampler,
    ProbabilityFlowODESampler, ImputationSampler,
    UNetScoreModel, UNetScoreModelColor,
    CondUNetScoreModel, CFGWrapper,
    GenerativeDiffusionModel,
)

# Example: VP process with cosine schedule + Euler-Maruyama sampler
process = VPProcess(schedule=CosineSchedule())
score   = UNetScoreModel(marginal_prob_std=process.sigma_t).to(device)
model   = GenerativeDiffusionModel(process, EulerMaruyamaSampler(), score, device)

model.train_epoch(train_loader, optimizer)
images = model.sample(n_images=16, img_shape=(1, 28, 28), n_steps=500)
bpd    = model.compute_bpd(x_eval, n_steps=50).mean()
```

---

## Key results (SVHN, N = 500, T = 200 steps, Euler-Maruyama)

| Process | FID ↓ |
|---------|-------|
| VE Brownian | 248.75 |
| VP Linear | 209.25 |
| VP Cosine | 199.70 |
| VP Exponential | **194.46** |