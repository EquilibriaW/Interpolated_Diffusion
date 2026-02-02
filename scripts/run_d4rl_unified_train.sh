#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_PY_MUJOCO_PATH=/workspace/Interpolated_Diffusion/mujoco210
export LD_LIBRARY_PATH=/workspace/Interpolated_Diffusion/mujoco210/bin:${LD_LIBRARY_PATH:-}
export MUJOCO_GL=egl
export D4RL_SUPPRESS_IMPORT_ERROR=1

T=${T:-128}
K=${K:-8}
K_MIN=${K_MIN:-8}
LEVELS=${LEVELS:-8}
STEPS=${STEPS:-20000}
BATCH=${BATCH:-256}
USE_SDF=${USE_SDF:-1}
DETERMINISTIC=${DETERMINISTIC:-0}
COND_START_GOAL=${COND_START_GOAL:-0}
CLAMP_ENDPOINTS=${CLAMP_ENDPOINTS:-1}

K_SCHEDULE=${K_SCHEDULE:-geom}
K_GEOM_GAMMA=${K_GEOM_GAMMA:-}

TRAIN_DATA=${TRAIN_DATA:-outputs/d4rl_prepared_unified_T${T}_train/dataset.npz}

CKPT_KP=${CKPT_KP:-checkpoints/keypoints_unified_T${T}}
CKPT_S2=${CKPT_S2:-checkpoints/interp_levels_unified_T${T}}
LOG_KP=${LOG_KP:-runs/keypoints_unified_T${T}}
LOG_S2=${LOG_S2:-runs/interp_levels_unified_T${T}}

common_k_schedule=()
if [[ -n "${K_GEOM_GAMMA}" ]]; then
  common_k_schedule+=(--k_geom_gamma "${K_GEOM_GAMMA}")
fi

echo "==> Train unified stage-1 keypoints"
PYTHONPATH=. .venv/bin/python -m src.train.train_keypoints \
  --dataset d4rl_prepared --prepared_path "${TRAIN_DATA}" \
  --T "${T}" --K "${K}" --steps "${STEPS}" --batch "${BATCH}" --use_sdf "${USE_SDF}" \
  --cond_start_goal "${COND_START_GOAL}" --clamp_endpoints "${CLAMP_ENDPOINTS}" \
  --deterministic "${DETERMINISTIC}" \
  --ckpt_dir "${CKPT_KP}" --log_dir "${LOG_KP}"

echo "==> Train unified stage-2 (dist corruption)"
PYTHONPATH=. .venv/bin/python -m src.train.train_interp_levels \
  --dataset d4rl_prepared --prepared_path "${TRAIN_DATA}" \
  --T "${T}" --K_min "${K_MIN}" --levels "${LEVELS}" --steps "${STEPS}" --batch "${BATCH}" --use_sdf "${USE_SDF}" \
  --deterministic "${DETERMINISTIC}" \
  --clamp_endpoints "${CLAMP_ENDPOINTS}" --cond_start_goal "${COND_START_GOAL}" \
  --stage2_mode adj \
  --k_schedule "${K_SCHEDULE}" "${common_k_schedule[@]}" \
  --corrupt_mode dist \
  --corrupt_sigma_max 0.08 --corrupt_sigma_min 0.012 --corrupt_sigma_pow 0.75 \
  --corrupt_anchor_frac 0.25 \
  --bootstrap_stage1_ckpt "${CKPT_KP}/ckpt_final.pt" \
  --bootstrap_ddim_steps 20 --bootstrap_ddim_schedule quadratic \
  --bootstrap_prob_start 0.0 --bootstrap_prob_end 0.5 --bootstrap_warmup_steps 5000 \
  --ckpt_dir "${CKPT_S2}" --log_dir "${LOG_S2}"

echo "==> Sample unified eval (points-based plotting)"
EVAL_DATA=${EVAL_DATA:-outputs/d4rl_prepared_unified_T${T}_eval/dataset.npz}
EVAL_DATA="${EVAL_DATA}" \
CKPT_ROOT=${CKPT_ROOT:-checkpoints} \
CKPT_KEYPOINTS="${CKPT_KP}/ckpt_final.pt" \
SAMPLE_CASES=${SAMPLE_CASES:-interp_levels_unified_T${T}} \
OUT_ROOT=${OUT_ROOT:-runs/ablate_stage2_unified} \
STAGE1_CACHE_MODE=${STAGE1_CACHE_MODE:-none} \
COND_START_GOAL="${COND_START_GOAL}" \
CLAMP_ENDPOINTS="${CLAMP_ENDPOINTS}" \
USE_SDF="${USE_SDF}" \
bash runs/d4rl_ablate_stage2_sample_only.sh

echo "==> Done"
