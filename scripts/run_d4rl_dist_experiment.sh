#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_PY_MUJOCO_PATH=/workspace/Interpolated_Diffusion/mujoco210
export LD_LIBRARY_PATH=/workspace/Interpolated_Diffusion/mujoco210/bin:${LD_LIBRARY_PATH:-}
export MUJOCO_GL=egl
export D4RL_SUPPRESS_IMPORT_ERROR=1

ENVS=(maze2d-umaze-v1 maze2d-medium-v1 maze2d-large-v1)
T=128
K=8
K_MIN=8
LEVELS=3
STEPS=20000
BATCH=256
USE_SDF=1
N_SAMPLES=32
DETERMINISTIC=0

for env in "${ENVS[@]}"; do
  train_ds="outputs/d4rl_prepared_${env}_T${T}_train/dataset.npz"
  eval_ds="outputs/d4rl_prepared_${env}_T${T}_eval/dataset.npz"
  kp_dir="checkpoints/keypoints_${env}_T${T}"
  s2_dir="checkpoints/interp_levels_${env}_T${T}"
  kp_log="runs/keypoints_${env}_T${T}"
  s2_log="runs/interp_levels_${env}_T${T}"

  echo "==> [${env}] Stage 1 keypoints"
  PYTHONPATH=. .venv/bin/python -m src.train.train_keypoints \
    --dataset d4rl_prepared --prepared_path "${train_ds}" \
    --T ${T} --K ${K} --steps ${STEPS} --batch ${BATCH} --use_sdf ${USE_SDF} \
    --deterministic ${DETERMINISTIC} \
    --ckpt_dir "${kp_dir}" --log_dir "${kp_log}"

  echo "==> [${env}] Stage 2 interp (dist corruption)"
  PYTHONPATH=. .venv/bin/python -m src.train.train_interp_levels \
    --dataset d4rl_prepared --prepared_path "${train_ds}" \
    --T ${T} --K_min ${K_MIN} --levels ${LEVELS} --steps ${STEPS} --batch ${BATCH} --use_sdf ${USE_SDF} \
    --corrupt_mode dist \
    --deterministic ${DETERMINISTIC} \
    --ckpt_dir "${s2_dir}" --log_dir "${s2_log}"

  echo "==> [${env}] Sample train"
  PYTHONPATH=. .venv/bin/python -m src.sample.sample_generate \
    --dataset d4rl_prepared --prepared_path "${train_ds}" \
    --T ${T} --K_min ${K_MIN} --levels ${LEVELS} --use_sdf ${USE_SDF} \
    --ckpt_keypoints "${kp_dir}/ckpt_final.pt" \
    --ckpt_interp "${s2_dir}/ckpt_final.pt" \
    --out_dir "runs/gen_${env}_train" --n_samples ${N_SAMPLES}

  echo "==> [${env}] Sample eval"
  PYTHONPATH=. .venv/bin/python -m src.sample.sample_generate \
    --dataset d4rl_prepared --prepared_path "${eval_ds}" \
    --T ${T} --K_min ${K_MIN} --levels ${LEVELS} --use_sdf ${USE_SDF} \
    --ckpt_keypoints "${kp_dir}/ckpt_final.pt" \
    --ckpt_interp "${s2_dir}/ckpt_final.pt" \
    --out_dir "runs/gen_${env}_eval" --n_samples ${N_SAMPLES}

  echo "==> [${env}] Done"
  echo

done
