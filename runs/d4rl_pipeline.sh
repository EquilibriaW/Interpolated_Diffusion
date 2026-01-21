#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_PY_MUJOCO_PATH=/workspace/mujoco210
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/workspace/mujoco210/bin"
export MUJOCO_GL=egl
export D4RL_SUPPRESS_IMPORT_ERROR=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -m src.train.train_keypoints \
  --dataset d4rl --env_id maze2d-medium-v1 \
  --T 64 --K 8 --N_train 200 --steps 20000 --use_sdf 1

python -m src.train.train_interp_levels \
  --dataset d4rl --env_id maze2d-medium-v1 \
  --T 64 --K_min 8 --levels 3 --steps 20000 --use_sdf 1 \
  --bootstrap_stage1_ckpt checkpoints/keypoints/ckpt_final.pt \
  --bootstrap_ddim_steps 20 --bootstrap_prob_start 0.0 --bootstrap_prob_end 0.3 --bootstrap_warmup_steps 5000

python -m src.sample.sample_generate \
  --dataset d4rl --env_id maze2d-medium-v1 \
  --ddim_steps 20 \
  --clamp_policy endpoints --clamp_dims pos \
  --save_diffusion_frames 1 --frames_stride 1 --frames_include_stage2 1 \
  --export_video mp4 --video_fps 8 \
  --out_dir runs/gen
