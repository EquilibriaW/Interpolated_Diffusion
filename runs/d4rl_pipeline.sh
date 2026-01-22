#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_PY_MUJOCO_PATH=/workspace/mujoco210
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/workspace/mujoco210/bin"
export MUJOCO_GL=egl
export D4RL_SUPPRESS_IMPORT_ERROR=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

./runs/d4rl_build_dataset.sh
./runs/d4rl_train_prepared.sh
