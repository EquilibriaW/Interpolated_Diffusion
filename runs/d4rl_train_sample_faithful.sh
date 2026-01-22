#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_PY_MUJOCO_PATH=/workspace/mujoco210
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/workspace/mujoco210/bin"
export MUJOCO_GL=egl
export D4RL_SUPPRESS_IMPORT_ERROR=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

TRAIN_DATASET_PATH=${1:-outputs/d4rl_prepared_large_T128_train/dataset.npz}
EVAL_DATASET_PATH=${2:-outputs/d4rl_prepared_large_T128_eval/dataset.npz}

python -m src.train.train_keypoints \
  --dataset d4rl_prepared --prepared_path "${TRAIN_DATASET_PATH}" \
  --T 128 --K 8 --steps 20000 --N_train 200 \
  --use_sdf 1 --use_start_goal 1

python -m src.train.train_interp_levels \
  --dataset d4rl_prepared --prepared_path "${TRAIN_DATASET_PATH}" \
  --T 128 --K_min 8 --levels 3 --steps 20000 \
  --use_sdf 1 --use_start_goal 1 \
  --bootstrap_stage1_ckpt checkpoints/keypoints/ckpt_final.pt \
  --bootstrap_ddim_steps 20 --bootstrap_prob_start 0.0 --bootstrap_prob_end 0.3 \
  --bootstrap_warmup_steps 5000

python -m src.sample.sample_generate \
  --dataset d4rl_prepared --prepared_path "${EVAL_DATASET_PATH}" \
  --T 128 --K_min 8 --ddim_steps 20 \
  --clamp_policy endpoints --clamp_dims pos \
  --n_samples 100 --batch 50 --sample_random 1 \
  --out_dir runs/gen_full_faithful

python -m src.sample.sample_generate \
  --dataset d4rl_prepared --prepared_path "${EVAL_DATASET_PATH}" \
  --T 128 --K_min 8 --ddim_steps 20 \
  --ckpt_interp "" --skip_stage2 1 \
  --compare_oracle 1 --plot_keypoints 1 \
  --kp_index_mode uniform \
  --n_samples 100 --batch 50 --sample_random 1 \
  --out_dir runs/gen_oracle_vs_pred

python -m src.sample.sample_generate \
  --dataset d4rl_prepared --prepared_path "${EVAL_DATASET_PATH}" \
  --T 128 --K_min 8 --ddim_steps 20 \
  --compare_oracle 1 --plot_keypoints 1 \
  --kp_index_mode uniform \
  --n_samples 100 --batch 50 --sample_random 1 \
  --out_dir runs/gen_oracle_pred_stage2
