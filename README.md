# Interpolated Diffusion for Particle Maze Trajectories

This repository implements two phases of trajectory diffusion for a Particle Maze domain:

- **Phase 1 (full-sequence refinement)**: Treats "keyframes + interpolation" as a deterministic corruption operator `C_K`. A diffusion model learns to invert this corruption on the residual.
- **Phase 2 (causal / diffusion-forcing style)**: A causal transformer denoiser uses per-timestep diffusion levels and inference-like rollouts.

The code is designed to run on a single RTX 3090 (24GB) using mixed precision, gradient accumulation, and checkpointing where needed.

## Theory / Motivation

We treat "keyframes + interpolation" as a deterministic corruption `C_K` applied to a clean trajectory `x`:

- Choose keyframe index set `K` (mask `M`), always including endpoints.
- Keep `x` at `K`, fill missing steps via interpolation: `y = C_K(x)`.
- Define residual `r = x - y`. Residual is exactly zero at keyframes.

We train a diffusion model on residual `r`, conditioned on `y`, mask `M`, and maze condition `c`.
At inference, we sample residual `r_hat` via reverse diffusion, clamp keyframes (`r[K]=0`), and output `x_hat = y + r_hat`.

Key insight: interpolation is a cheap proposal ("half step"); diffusion corrects it ("full step").
We then test how many reverse steps are needed for high-quality correction.

Phase 2 moves to a causal/diffusion-forcing setup:
- The model is causal in time.
- Each timestep `i` has its own diffusion level `t_i` (per-frame noise).
- Training matches inference-like rollouts with a clean past + noisy future mixture.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### D4RL Maze2d (optional)

To use the D4RL Maze2d environments (`maze2d-umaze-v1`, `maze2d-medium-v1`, `maze2d-large-v1`),
install D4RL and its dependencies separately (MuJoCo + Gym). Suggested workflow:

```bash
# install D4RL extras after torch is installed
pip install -r requirements-d4rl.txt
```

If you already have a working D4RL install, no further changes are needed.

## Project Structure

- `README.md`
- `requirements.txt`
- `configs/`
- `src/`
  - `data/`
    - `maze.py`            # maze generation, SDF computation
    - `astar.py`           # A* path planning on grid
    - `trajectories.py`    # convert grid path -> continuous trajectory length T
    - `dataset.py`         # PyTorch Dataset, on-the-fly generation + caching
  - `corruptions/`
    - `keyframes.py`       # sample keyframe masks, interpolation
  - `diffusion/`
    - `schedules.py`       # beta schedules, alpha_bar, utilities
    - `ddpm.py`            # forward noise, reverse sampler (DDPM + DDIM)
    - `rectified_flow.py`  # optional, placeholder
  - `models/`
    - `encoders.py`        # CNN encoder for maze/SDF and start/goal embed
    - `transformer.py`     # Transformer blocks (bi-dir + causal)
    - `denoiser_fullseq.py`
    - `denoiser_causal.py`
  - `train/`
    - `train_fullseq.py`
    - `train_causal.py`
  - `eval/`
    - `metrics.py`
    - `visualize.py`
  - `sample/`
    - `sample_fullseq.py`
    - `sample_causal.py`
  - `utils/`
    - `seed.py`, `device.py`, `logging.py`, `ema.py`, `checkpoint.py`
- `tests/`

## Phase 1: Full-Sequence Residual Diffusion

### Train

```bash
python -m src.train.train_fullseq --T 64 --N_train 1000 --batch 256 --steps 20000 --use_sdf 1
```

### Train on D4RL Maze2d

```bash
python -m src.train.train_fullseq --dataset d4rl --env_id maze2d-medium-v1 --T 64 --batch 256 --with_velocity 1
```

### Sample

```bash
python -m src.sample.sample_fullseq --ckpt <path> --ddim_steps 50
python -m src.sample.sample_fullseq --ckpt <path> --ddim_steps 10
python -m src.sample.sample_fullseq --ckpt <path> --ddim_steps 2
python -m src.sample.sample_fullseq --ckpt <path> --ddim_steps 1
```

The sampler writes metrics and PNG plots into the output directory and prints a short summary.

## Phase 2: Causal / Diffusion-Forcing-Style Variant

### Train

```bash
python -m src.train.train_causal --T 64 --chunk 16 --N_train 1000 --batch 256 --steps 20000
```

### Train on D4RL Maze2d

```bash
python -m src.train.train_causal --dataset d4rl --env_id maze2d-medium-v1 --T 64 --chunk 16 --batch 256 --with_velocity 1
```

### Sample

```bash
python -m src.sample.sample_causal --ckpt <path> --chunk 16 --ddim_steps 5
```

## Notes

- Mixed precision and gradient accumulation are enabled by default.
- EMA checkpoints are saved when enabled.
- Deterministic seeds are set in `src/utils/seed.py`.
- Minimal unit tests live in `tests/`.
