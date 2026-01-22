#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_PY_MUJOCO_PATH=/workspace/mujoco210
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/workspace/mujoco210/bin"
export MUJOCO_GL=egl
export D4RL_SUPPRESS_IMPORT_ERROR=1

python -m src.data.prepare_d4rl_dataset \
  --env_id maze2d-large-v1 \
  --out_dir outputs/d4rl_prepared_large_T128_eval \
  --num_samples 1000 --hard_fraction 0.5 --T 128 --use_sdf 1 --d4rl_flip_y 0 \
  --max_collision_rate 0.0 --max_resample_tries 200 \
  --min_goal_dist 6.0 --min_path_len 12.0 --min_tortuosity 1.8 --min_turns 6 --turn_angle_deg 30 \
  --easy_min_goal_dist 3.0 --easy_min_path_len 6.0 --easy_min_tortuosity 1.2 --easy_min_turns 3 --easy_turn_angle_deg 25 \
  --window_mode episode --goal_mode window_end \
  --episode_split_mod 10 --episode_split_val 0
