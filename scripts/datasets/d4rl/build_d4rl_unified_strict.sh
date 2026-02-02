#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/datasets/d4rl/build_d4rl_unified_strict.sh
# Override any variable via env, e.g.:
#   NUM_TRAIN=20000 MIN_GOAL_DIST=1.5 bash scripts/datasets/d4rl/build_d4rl_unified_strict.sh

export MUJOCO_PY_MUJOCO_PATH=${MUJOCO_PY_MUJOCO_PATH:-/workspace/Interpolated_Diffusion/mujoco210}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/workspace/Interpolated_Diffusion/mujoco210/bin}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export D4RL_SUPPRESS_IMPORT_ERROR=${D4RL_SUPPRESS_IMPORT_ERROR:-1}

PYTHON=${PYTHON:-.venv/bin/python}
T=${T:-128}

# Sample counts
NUM_TRAIN=${NUM_TRAIN:-10000}
NUM_EVAL=${NUM_EVAL:-1000}

# Strict collision, relaxed length/turns (tunable).
MAX_COLLISION=${MAX_COLLISION:-0.0}
MIN_GOAL_DIST=${MIN_GOAL_DIST:-2.0}
MIN_PATH_LEN=${MIN_PATH_LEN:-4.0}
MIN_TORTUOSITY=${MIN_TORTUOSITY:-1.2}
MIN_TURNS=${MIN_TURNS:-2}
TURN_ANGLE_DEG=${TURN_ANGLE_DEG:-30.0}

# Easy split thresholds (slightly looser).
EASY_MIN_GOAL_DIST=${EASY_MIN_GOAL_DIST:-1.0}
EASY_MIN_PATH_LEN=${EASY_MIN_PATH_LEN:-2.0}
EASY_MIN_TORTUOSITY=${EASY_MIN_TORTUOSITY:-1.1}
EASY_MIN_TURNS=${EASY_MIN_TURNS:-1}
EASY_TURN_ANGLE_DEG=${EASY_TURN_ANGLE_DEG:-30.0}

MAX_RESAMPLE_TRIES=${MAX_RESAMPLE_TRIES:-2000}
HARD_FRACTION=${HARD_FRACTION:-0.5}

ENVS=(maze2d-umaze-v1 maze2d-medium-v1 maze2d-large-v1)

build_split () {
  local split=$1
  local num=$2
  local split_mod=$3
  local split_val=$4
  for env in "${ENVS[@]}"; do
    echo "==> Build ${split} prepared for ${env}"
    PYTHONPATH=. ${PYTHON} src/data/prepare_d4rl_dataset.py \
      --env_id "${env}" \
      --out_dir "outputs/d4rl_prepared_${env}_T${T}_${split}" \
      --num_samples "${num}" --T "${T}" \
      --hard_fraction "${HARD_FRACTION}" \
      --max_collision_rate "${MAX_COLLISION}" \
      --max_resample_tries "${MAX_RESAMPLE_TRIES}" \
      --min_goal_dist "${MIN_GOAL_DIST}" \
      --min_path_len "${MIN_PATH_LEN}" \
      --min_tortuosity "${MIN_TORTUOSITY}" \
      --min_turns "${MIN_TURNS}" \
      --turn_angle_deg "${TURN_ANGLE_DEG}" \
      --easy_min_goal_dist "${EASY_MIN_GOAL_DIST}" \
      --easy_min_path_len "${EASY_MIN_PATH_LEN}" \
      --easy_min_tortuosity "${EASY_MIN_TORTUOSITY}" \
      --easy_min_turns "${EASY_MIN_TURNS}" \
      --easy_turn_angle_deg "${EASY_TURN_ANGLE_DEG}" \
      --episode_split_mod "${split_mod}" \
      --episode_split_val "${split_val}" \
      --require_accept 1
  done
}

build_split train "${NUM_TRAIN}" 10 0
build_split eval "${NUM_EVAL}" 10 1

echo "==> Build unified train dataset (pad-only, no resizing)"
PYTHONPATH=. ${PYTHON} scripts/datasets/d4rl/build_unified_prepared.py \
  --inputs \
    "outputs/d4rl_prepared_maze2d-umaze-v1_T${T}_train/dataset.npz" \
    "outputs/d4rl_prepared_maze2d-medium-v1_T${T}_train/dataset.npz" \
    "outputs/d4rl_prepared_maze2d-large-v1_T${T}_train/dataset.npz" \
  --out_dir "outputs/d4rl_prepared_unified_T${T}_train" \
  --use_sdf 1 --resize_mode pad --pad_scale_mode none

echo "==> Build unified eval dataset (pad-only, no resizing)"
PYTHONPATH=. ${PYTHON} scripts/datasets/d4rl/build_unified_prepared.py \
  --inputs \
    "outputs/d4rl_prepared_maze2d-umaze-v1_T${T}_eval/dataset.npz" \
    "outputs/d4rl_prepared_maze2d-medium-v1_T${T}_eval/dataset.npz" \
    "outputs/d4rl_prepared_maze2d-large-v1_T${T}_eval/dataset.npz" \
  --out_dir "outputs/d4rl_prepared_unified_T${T}_eval" \
  --use_sdf 1 --resize_mode pad --pad_scale_mode none

echo "==> Done"
