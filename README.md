# SENDAI — Sparse Encoder-Decoder with Nonlinear Dynamics and AI

## What It Does

- Encodes sparse PMU (Phasor Measurement Unit) sensor histories with GRU layers.
- Reconstructs full-grid frequency-deviation fields with a shallow MLP decoder.
- Regularizes LF training with SINDy terms in the GRU latent space, enabling equation discovery.
- Discovers post-hoc symbolic latent dynamics with PySINDy (`dz/dt = f(z)`).
- Corrects high-frequency residuals with a hierarchical stack of 1D-Conv peeling layers.
- Supports forward prediction (inference rollout) by integrating the identified SINDy ODE beyond the observation window.


---

## Pipeline Overview

```
Sparse sensors (10 PMUs)
        │
        ▼
 ┌─────────────┐     Low-pass filter → smooth target
 │  LF SHRED   │  GRU encoder → latent z (dim 8)
 │  + SINDy    │  MLP decoder → full grid (22 PMUs)
 └─────────────┘     SINDy regularization during training
        │
        ▼ residual
 ┌─────────────┐
 │  HF Peeling │  10× (1D-Conv + MLP) correction layers
 │  Layers     │  trained sequentially on residuals
 └─────────────┘
        │
        ▼
 Full reconstruction + SINDy rollout forecast
```

---

## Installation

Requires Python 3.9+. Install PyTorch first for your hardware.

### 1. Install PyTorch

```bash
# CUDA 12.8
pip install torch --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124

# CPU only
pip install torch
```

Verify:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install pysindy
```

### 3. Confirm imports

```bash
python -c "import torch, pysindy, numpy, scipy; print('SENDAI ready')"
```

---

## Data

Place datasets under `data/synthetic/`. The default scenario expects:

```
data/
└── synthetic/
    └── iberian_event/
        ├── measurements_2016.csv               # PMU frequency-deviation time series
        ├── raw_data.npz                   # optional ground-truth latent trajectory
        └── metadata.json                  # sampling rate (fs in Hz)
```

The CSV contains one column per PMU node. Each row is one time step of frequency deviation in Hz.

---

## Quick Start

Open and run `main.ipynb` end-to-end:

```bash
jupyter lab main.ipynb
```

Results are written to `data/synthetic/iberian_event/results/`.

---

## Python API

The pipeline is driven by a `PipelineConfig` dataclass and a `run_pipeline` orchestrator.

```python
from dataclasses import replace
# all code lives in main.ipynb — import the relevant cells or convert to a module

config = PipelineConfig(
    latent_dim=4,
    lags=400,
    lstm_hidden=64,
    lf_epochs=500,
    sindy_poly_order=1,
    sindy_regularization=5.0,
    sindy_threshold=0.0,
    lf_cutoff_hz=0.08,
    n_peel_layers=10,
    hf_epochs=400,
    finetune_epochs=150,
    lambda_sparse=0.1,
    n_pmu_total=22,
    n_pmu_observed=10,
    fs=10.0,
)

results = run_pipeline(config)
```

Key attributes of the returned `results` object:

| Attribute | Description |
|---|---|
| `results.model` | Trained `FullReconstructionModel` (LF + HF) |
| `results.lf_recon` | LF-only reconstruction array, shape `(T, M)` |
| `results.full_recon` | LF + HF reconstruction array, shape `(T, M)` |
| `results.latent_traj` | GRU latent trajectory, shape `(T, latent_dim)` |
| `results.sindy_model` | Fitted PySINDy model (post-hoc identification) |
| `results.metrics` | Dict with RMSE values and SINDy sparsity counts |

---

## Key Components

### `PipelineConfig`

Dataclass holding all hyperparameters. Key fields:

| Parameter | Default | Description |
|---|---|---|
| `lf_mode` | `"sindy_shred"` | `"sindy_shred"` (GRU) or `"vanilla_shred"` (LSTM, no SINDy) |
| `latent_dim` | `4` | GRU hidden state / latent space dimension |
| `lags` | `400` | Sensor history window fed to the GRU |
| `lstm_hidden` | `64` | GRU hidden size |
| `decoder_hidden` | `[256, 256]` | MLP decoder layer sizes |
| `sindy_poly_order` | `1` | Polynomial order of the SINDy library |
| `sindy_regularization` | `5.0` | SINDy loss weight (ramps from 0 after warmup) |
| `sindy_threshold` | `0.0` | STLS pruning threshold |
| `lf_cutoff_hz` | `0.08` | Butterworth low-pass cutoff for LF target smoothing |
| `n_peel_layers` | `10` | Number of HF correction layers |
| `hf_conv_channels` | `[32, 64]` | 1D-Conv channel sizes in each HF layer |
| `lambda_sparse` | `0.1` | HF frequency-sparsity regularization weight |
| `max_freq_hz` | `5.0` | HF out-of-band penalty cutoff (Hz) |
| `fs` | `10.0` | Sampling frequency of the PMU data (Hz) |
| `n_pmu_total` | `22` | Total PMU nodes in the grid |
| `n_pmu_observed` | `10` | Number of observed (sensor) nodes |

### `SHREDNet`

GRU encoder → linear projection to latent space → MLP decoder. In `sindy_shred` mode, holds a learnable SINDy coefficient matrix `Xi` of shape `(library_dim, latent_dim)` with a binary STLS mask.

### `HFPeelLayer`

Single high-frequency correction module: 1D-Conv temporal encoder + per-timestep MLP → full-grid correction. Layers are trained sequentially, each on the residual of all previous layers.

### `FullReconstructionModel`

Wraps one `SHREDNet` (LF) and a `nn.ModuleList` of `HFPeelLayer` objects (HF).

### `DataPreparer`

Handles CSV loading, MinMax scaling, Butterworth low-pass filtering, and construction of `PMUTimeSeriesDataset` (LF) and `HFWindowDataset` (HF) objects.

---

## Training Details

**Stage 1 — LF training (`train_lf`)**

- Adam optimizer, `lr=1e-4`, `ReduceLROnPlateau` scheduler.
- Loss = reconstruction MSE + SINDy regularization (after `sindy_warmup_epochs=20` epochs).
- SINDy term: sub-stepped Euler integration of `dz/dt = Theta(z) @ Xi`; MSE against true next latent state.
- STLS pruning every `sindy_thres_epoch=10` epochs: zero small coefficients, refit survivors via central finite differences.
- Early stopping at `lf_patience=30` epochs.

**Stage 2 — HF residual preparation**

LF predictions are computed on all splits; HF residuals are built as `full_signal - LF_prediction`.

**Stage 3 — HF layer training (`train_hf_layer`, repeated ×10)**

- Each layer trained with frozen LF model and all previous HF layers.
- Loss = 0.5 × sensor MSE + 0.5 × full-grid MSE + `lambda_sparse` × frequency-sparsity loss + `lambda_mag` × magnitude penalty.
- `lambda_sparse` ramps from 0 to `0.1` during training, then drops to `0.002` for fine-tuning.

---

## Outputs

All results are written to `data/synthetic/iberian_event/results/`:

| File | Description |
|---|---|
| `plots/00_training_curves.png` | LF + HF loss curves (log scale); SINDy loss overlaid |
| `plots/01_reconstruction_comparison.png` | Per-PMU reconstruction: truth vs LF vs LF+HF; optional SINDy rollout |
| `plots/01_reconstruction_comparison.csv` | Time-indexed CSV of GT, LF, and LF+HF for all PMUs |
| `plots/02_hf_spectrum.png` | Mean FFT magnitude of HF corrections vs frequency |
| `plots/03_latent_space.png` | GRU latent trajectory vs SINDy-simulated ODE |
| `plots/latent_space_comparison.csv` | Latent trajectories (GRU and SINDy) |
| `metrics.json` | LF RMSE (scaled + Hz), HF RMSE, improvement %, SINDy sparsity |
| `checkpoint.pt` | Full model state dict, config, sensor locations, training history |
| `log_<timestamp>.txt` | Full console output |

---

## References

- **SHRED**: Williams, Jan P., Olivia Zahn, and J. Nathan Kutz. "Sensing with shallow recurrent decoder networks." Proceedings of the Royal Society A 480.2298 (2024): 20240054.
- **SINDy**: Brunton, Steven L., Joshua L. Proctor, and J. Nathan Kutz. "Discovering governing equations from data by sparse identification of nonlinear dynamical systems." Proceedings of the national academy of sciences 113.15 (2016): 3932-3937.
- **SINDy-SHRED**:Gao, Mars Liyao, Jan P. Williams, and J. Nathan Kutz. "Sparse identification of nonlinear dynamics and koopman operators with shallow recurrent decoder networks." Proceedings of the National Academy of Sciences 123.16 (2026): e2508144123.
- **PySINDy**: Zhang, Xingyue, et al. "SENDAI: A Hierarchical Sparse-measurement, EfficieNt Data AssImilation Framework." ICLR 2026 Workshop on Foundation Models for Science: Real-World Impact and Science-First Design. 2026.
