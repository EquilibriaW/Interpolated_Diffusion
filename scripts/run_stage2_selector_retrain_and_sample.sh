#!/usr/bin/env bash
set -euo pipefail

export D4RL_SUPPRESS_IMPORT_ERROR=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

PYTHON_BIN=${PYTHON_BIN:-}
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

T=${T:-128}
K_MIN=${K_MIN:-8}
LEVELS=${LEVELS:-8}
STEPS=${STEPS:-20000}
BATCH=${BATCH:-256}
USE_SDF=${USE_SDF:-1}
COND_START_GOAL=${COND_START_GOAL:-1}
CLAMP_ENDPOINTS=${CLAMP_ENDPOINTS:-1}
POS_CLIP=${POS_CLIP:-1}
POS_CLIP_MIN=${POS_CLIP_MIN:-0.0}
POS_CLIP_MAX=${POS_CLIP_MAX:-1.0}

TRAIN_DATA=${TRAIN_DATA:-outputs/d4rl_prepared_unified_T${T}_train_dp/dataset.npz}
EVAL_DATA=${EVAL_DATA:-outputs/d4rl_prepared_unified_T${T}_eval_dp/dataset.npz}

SELECTOR_CKPT=${SELECTOR_CKPT:-checkpoints/keypoint_selector_unified_T${T}_dp/ckpt_final.pt}
KP_CKPT=${KP_CKPT:-checkpoints/keypoints_unified_T${T}_dp/ckpt_final.pt}
DPHI_CKPT=${DPHI_CKPT:-checkpoints/segment_cost_unified_T${T}_dp/ckpt_final.pt}

S2_D_MODEL=${S2_D_MODEL:-384}
S2_N_LAYERS=${S2_N_LAYERS:-12}
S2_N_HEADS=${S2_N_HEADS:-12}
S2_D_FF=${S2_D_FF:-1536}
S2_D_COND=${S2_D_COND:-128}
S2_MAZE_CHANNELS=${S2_MAZE_CHANNELS:-32,64,128,128}

CKPT_S2=${CKPT_S2:-checkpoints/interp_levels_unified_T${T}_dp}
LOG_S2=${LOG_S2:-runs/interp_levels_unified_T${T}_dp}

K_SCHEDULE=${K_SCHEDULE:-geom}
CORRUPT_MODE=${CORRUPT_MODE:-dist}
CORRUPT_SIGMA_MAX=${CORRUPT_SIGMA_MAX:-0.08}
CORRUPT_SIGMA_MIN=${CORRUPT_SIGMA_MIN:-0.012}
CORRUPT_SIGMA_POW=${CORRUPT_SIGMA_POW:-0.75}
CORRUPT_ANCHOR_FRAC=${CORRUPT_ANCHOR_FRAC:-0.25}

OUT_ROOT=${OUT_ROOT:-runs/ablate_stage2_unified_dp}
SAMPLE_CASES=${SAMPLE_CASES:-interp_levels_unified_T${T}_dp}
STAGE1_CACHE_MODE=${STAGE1_CACHE_MODE:-none}
KP_INDEX_MODE=${KP_INDEX_MODE:-selector}
POS_CLIP=${POS_CLIP:-1}

echo "==> Train stage-2 with selector-aligned indices"
PYTHONPATH=. "${PYTHON_BIN}" -m src.train.train_interp_levels \
  --dataset d4rl_prepared --prepared_path "${TRAIN_DATA}" \
  --T "${T}" --K_min "${K_MIN}" --levels "${LEVELS}" --steps "${STEPS}" --batch "${BATCH}" --use_sdf "${USE_SDF}" \
  --cond_start_goal "${COND_START_GOAL}" --clamp_endpoints "${CLAMP_ENDPOINTS}" \
  --pos_clip "${POS_CLIP}" --pos_clip_min "${POS_CLIP_MIN}" --pos_clip_max "${POS_CLIP_MAX}" \
  --stage2_mode adj \
  --kp_index_mode selector \
  --selector_ckpt "${SELECTOR_CKPT}" \
  --bootstrap_stage1_ckpt "${KP_CKPT}" \
  --dphi_ckpt "${DPHI_CKPT}" \
  --k_schedule "${K_SCHEDULE}" \
  --corrupt_mode "${CORRUPT_MODE}" --corrupt_sigma_max "${CORRUPT_SIGMA_MAX}" --corrupt_sigma_min "${CORRUPT_SIGMA_MIN}" --corrupt_sigma_pow "${CORRUPT_SIGMA_POW}" \
  --corrupt_anchor_frac "${CORRUPT_ANCHOR_FRAC}" \
  --s2_d_model "${S2_D_MODEL}" --s2_n_layers "${S2_N_LAYERS}" --s2_n_heads "${S2_N_HEADS}" --s2_d_ff "${S2_D_FF}" --s2_d_cond "${S2_D_COND}" \
  --s2_maze_channels "${S2_MAZE_CHANNELS}" \
  --ckpt_dir "${CKPT_S2}" --log_dir "${LOG_S2}"

echo "==> Sample with retrained stage-2"
EVAL_DATA="${EVAL_DATA}" \
CKPT_ROOT=checkpoints \
CKPT_KEYPOINTS="${KP_CKPT}" \
SAMPLE_CASES="${SAMPLE_CASES}" \
OUT_ROOT="${OUT_ROOT}" \
DPHI_CKPT="${DPHI_CKPT}" \
SELECTOR_CKPT="${SELECTOR_CKPT}" \
KP_INDEX_MODE="${KP_INDEX_MODE}" \
POS_CLIP="${POS_CLIP}" \
STAGE1_CACHE_MODE="${STAGE1_CACHE_MODE}" \
COND_START_GOAL="${COND_START_GOAL}" \
CLAMP_ENDPOINTS="${CLAMP_ENDPOINTS}" \
USE_SDF="${USE_SDF}" \
bash runs/d4rl_ablate_stage2_sample_only.sh

echo "==> Done"
