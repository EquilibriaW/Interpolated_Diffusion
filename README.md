# Interpolated Diffusion for Particle Maze Trajectories

This repository implements a **two-stage generative pipeline** for trajectory diffusion in Particle Maze (and optional D4RL Maze2D):

- **Stage 1 (Keypoint diffusion)**: sample sparse anchors (keypoints) from Gaussian noise using a transformer over **K tokens only**.
- **Stage 2 (Interpolation-corruption denoiser)**: treat anchor density as the discrete corruption level and predict the clean trajectory **x0** (or delta) from an interpolated sequence.

The code is designed to run on a single RTX 3090/4090 using mixed precision, gradient accumulation, and checkpointing.

## Theory / Motivation

We treat **anchor density as the "noise level"** (not Gaussian noise on residuals):

1. Sample nested masks `{M_0 ... M_S}` with increasing anchor counts (M_S is sparsest).
2. Corrupt a clean trajectory by deterministic interpolation:
   - `x_s = Interp(x0 | M_s)`
3. Train a denoiser `f_theta(x_s, M_s, s, cond) -> x0` (or delta = x0 - x_s).

At inference:
1. **Stage 1** samples keypoints `z` at `M_S` from Gaussian noise.
2. Interpolate to full sequence: `x_S = Interp(z | M_S)`.
3. **Stage 2** does a **one-step jump** `x_hat0 = f_theta(x_S, s=S, M_S, cond)` and clamps anchors.

This aligns “high noise ↔ few anchors” and makes the keypoint attention compute cheap: O(K^2).

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
    - `denoiser_keypoints.py`
    - `denoiser_interp_levels.py`
    - `denoiser_interp_levels_causal.py`
  - `train/`
    - `train_keypoints.py`
    - `train_interp_levels.py`
    - `train_interp_levels_causal.py`
    - `train_fullseq.py`   # alias -> train_interp_levels
    - `train_causal.py`    # alias -> train_interp_levels_causal
  - `eval/`
    - `metrics.py`
    - `visualize.py`
  - `sample/`
    - `sample_keypoints.py`
    - `sample_generate.py`
    - `sample_generate_causal.py`
    - `sample_fullseq.py`  # alias -> sample_generate
    - `sample_causal.py`   # alias -> sample_generate_causal
  - `utils/`
    - `seed.py`, `device.py`, `logging.py`, `ema.py`, `checkpoint.py`
- `tests/`

## Stage 1: Keypoint Diffusion (Gaussian DDPM)

### Train

```bash
python -m src.train.train_keypoints --dataset particle --T 64 --K 8 --steps 20000
```

### Train on D4RL Maze2d

```bash
python -m src.train.train_keypoints --dataset d4rl --env_id maze2d-medium-v1 --T 64 --K 8 --steps 20000
```

### Sample keypoints

```bash
python -m src.sample.sample_keypoints --ckpt <path> --T 64 --K 8 --ddim_steps 20
```

## Stage 2: Interpolation-Corruption Denoiser (x0-pred)

### Train

```bash
python -m src.train.train_interp_levels --dataset particle --T 64 --K_min 8 --levels 3 --steps 20000
```

### Optional: Stage-1-anchored bootstrapping (scheduled sampling)

```bash
python -m src.train.train_interp_levels \
  --dataset particle --T 64 --K_min 8 --levels 3 --steps 20000 \
  --bootstrap_stage1_ckpt checkpoints/keypoints/ckpt_final.pt \
  --bootstrap_prob_start 0.0 --bootstrap_prob_end 0.3 --bootstrap_warmup_steps 5000 \
  --bootstrap_prob_cap 0.5
```

`--bootstrap_prob_cap` (default 0.5) limits the fraction of batches/examples that use Stage‑1 anchors during training.

### Train on D4RL Maze2d

```bash
python -m src.train.train_interp_levels --dataset d4rl --env_id maze2d-medium-v1 --T 64 --K_min 8 --levels 3 --steps 20000
```

## End-to-End Generation (keypoints -> interpolate -> one-step denoise)

```bash
python -m src.sample.sample_generate --dataset particle --T 64 --K_min 8 --levels 3 --n_samples 16 --out_dir runs/gen
python -m src.sample.sample_generate --dataset d4rl --env_id maze2d-medium-v1 --T 64 --K_min 8 --levels 3 --n_samples 16 --out_dir runs/gen
```

Clamp policy:
- `--clamp_policy endpoints` (default): clamp only t=0 and t=T-1 after Stage 2
- `--clamp_policy all_anchors`: clamp all anchor indices (legacy behavior)
- `--clamp_policy none`: no clamping
- `--clamp_dims pos|all` controls whether to clamp only position dims (0:2) or all dims

### D4RL visualization

For D4RL Maze2D, samples are plotted in **continuous world coordinates** with maze walls extracted from MuJoCo geom boxes when available.
If MuJoCo wall geoms are not available, the sampler falls back to `maze_map` (if exposed), and then to the grid-based occupancy plot.

### Diffusion-step frames (optional)

To save per‑step frames of the keypoint diffusion process (interpolated at each DDIM step) and export video:

```bash
python -m src.sample.sample_generate \
  --dataset d4rl --env_id maze2d-medium-v1 \
  --T 64 --K_min 8 --levels 3 --n_samples 1 \
  --save_diffusion_frames 1 --frames_stride 1 --frames_include_stage2 1 \
  --export_video mp4 --video_fps 8 \
  --out_dir runs/gen
```

Frames are written under `runs/gen/diffusion_steps/sample_0000/step_*.png`.
If `imageio` is installed, mp4/gif export is handled in Python; otherwise `ffmpeg` is used for mp4 export (if available).

## Causal Variant (optional)

```bash
python -m src.train.train_interp_levels_causal --dataset particle --T 64 --K_min 8 --levels 3 --steps 20000
python -m src.sample.sample_generate_causal --dataset particle --T 64 --K_min 8 --levels 3 --n_samples 16 --out_dir runs/gen_causal
```

To save per‑chunk frames and export a video (causal generation):

```bash
python -m src.sample.sample_generate_causal \
  --dataset d4rl --env_id maze2d-medium-v1 \
  --T 64 --K_min 8 --levels 3 --n_samples 1 \
  --save_chunk_frames 1 --frames_stride 1 \
  --export_video mp4 --video_fps 8 \
  --out_dir runs/gen_causal
```

## DiDeMo (text-conditioned video, 2-5s)

Fetch metadata + download videos (AWS script port):

```bash
python scripts/datasets/didemo/fetch_didemo_metadata.py --out_dir data/didemo
python scripts/datasets/didemo/download_videos_aws.py --download --video_directory data/didemo/videos --data_dir data/didemo
```

Stage-1 keypoint diffusion (text-conditioned, VAE-latent tokens):

```bash
python -m src.train.train_keypoints_didemo --T 16 --K 4 --frame_size 64 --clip_seconds 5 --patch_size 2 --batch 8 --steps 20000
```

Stage-2 interpolation-corruption denoiser:

```bash
python -m src.train.train_interp_levels_didemo --T 16 --K_min 4 --levels 4 --frame_size 64 --clip_seconds 5 --patch_size 2 --batch 4 --steps 20000
```

Optional: precompute VAE latents + CLIP text embeddings (speeds up training):

```bash
python scripts/datasets/didemo/precompute_cache.py --split train --max_items 1000 --batch 8 --shard_size 256 \
  --out_dir data/didemo_cache --data_dir data/didemo --video_dir data/didemo/videos
```

Then train from cache:

```bash
python -m src.train.train_keypoints_didemo --cache_dir data/didemo_cache --use_cached_text 1 --steps 20000
python -m src.train.train_interp_levels_didemo --cache_dir data/didemo_cache --use_cached_text 1 --steps 20000
```

## Notes

- Mixed precision and gradient accumulation are enabled by default.
- EMA checkpoints are saved when enabled.
- Deterministic seeds are set in `src/utils/seed.py`.
- Minimal unit tests live in `tests/`.
