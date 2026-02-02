#!/usr/bin/env bash
set -euo pipefail

# Retrain stage-2 for 10k steps with small dist noise (Soft-Diffusion style),
# then sample with the new checkpoint.
# Usage:
#   bash scripts/run_stage2_smallnoise_10k_and_sample.sh
#
# Optional overrides (env vars):
#   TRAIN_DATA, EVAL_DATA, CKPT_DIR, LOG_DIR, OUT_ROOT
#   KP_INDEX_MODE, SELECTOR_CKPT, DPHI_CKPT, BOOTSTRAP_STAGE1
#   S2_D_MODEL, S2_N_LAYERS, S2_N_HEADS, S2_D_FF, S2_D_COND, S2_MAZE_CHANNELS
#   STEPS, BATCH, LEVELS, K_MIN, K_SCHEDULE, K_GEOM_GAMMA
#   CORRUPT_SIGMA_MAX, CORRUPT_SIGMA_MIN, CORRUPT_SIGMA_POW, CORRUPT_ANCHOR_FRAC
#   MASK_POLICY_MIX

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

TRAIN_DATA=${TRAIN_DATA:-outputs/d4rl_prepared_unified_T128_train_dp/dataset.npz}
EVAL_DATA=${EVAL_DATA:-outputs/d4rl_prepared_unified_T128_eval_dp/dataset.npz}

CKPT_DIR=${CKPT_DIR:-checkpoints/interp_levels_unified_T128_dp_smallnoise_10k}
LOG_DIR=${LOG_DIR:-runs/interp_levels_unified_T128_dp_smallnoise_10k}
OUT_ROOT=${OUT_ROOT:-runs/ablate_stage2_unified_dp_smallnoise_10k}

BOOTSTRAP_STAGE1=${BOOTSTRAP_STAGE1:-checkpoints/keypoints_unified_T128_dp/ckpt_final.pt}
DPHI_CKPT=${DPHI_CKPT:-checkpoints/segment_cost_unified_T128_dp/ckpt_final.pt}

KP_INDEX_MODE=${KP_INDEX_MODE:-uniform}
SELECTOR_CKPT=${SELECTOR_CKPT:-checkpoints/keypoint_selector_unified_T128_dp/ckpt_final.pt}

STAGE1_CACHE_MODE=${STAGE1_CACHE_MODE:-none}
COND_START_GOAL=${COND_START_GOAL:-1}
CLAMP_ENDPOINTS=${CLAMP_ENDPOINTS:-1}
USE_SDF=${USE_SDF:-1}
POS_CLIP=${POS_CLIP:-1}

STEPS=${STEPS:-10000}
BATCH=${BATCH:-256}
LEVELS=${LEVELS:-8}
K_MIN=${K_MIN:-8}
K_SCHEDULE=${K_SCHEDULE:-geom}
K_GEOM_GAMMA=${K_GEOM_GAMMA:-}
MASK_POLICY_MIX=${MASK_POLICY_MIX:-}

S2_D_MODEL=${S2_D_MODEL:-384}
S2_N_LAYERS=${S2_N_LAYERS:-12}
S2_N_HEADS=${S2_N_HEADS:-12}
S2_D_FF=${S2_D_FF:-1536}
S2_D_COND=${S2_D_COND:-128}
S2_MAZE_CHANNELS=${S2_MAZE_CHANNELS:-32,64,128,128}

CORRUPT_SIGMA_MAX=${CORRUPT_SIGMA_MAX:-0.02}
CORRUPT_SIGMA_MIN=${CORRUPT_SIGMA_MIN:-0.003}
CORRUPT_SIGMA_POW=${CORRUPT_SIGMA_POW:-0.75}
CORRUPT_ANCHOR_FRAC=${CORRUPT_ANCHOR_FRAC:-0.25}
S2_SAMPLE_NOISE_MODE=${S2_SAMPLE_NOISE_MODE:-level}
S2_SAMPLE_NOISE_SCALE=${S2_SAMPLE_NOISE_SCALE:-1.0}
S2_SAMPLE_NOISE_SIGMA=${S2_SAMPLE_NOISE_SIGMA:-${CORRUPT_SIGMA_MIN}}

mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "${OUT_ROOT}"

common_k_schedule=()
if [[ -n "${K_GEOM_GAMMA}" ]]; then
  common_k_schedule+=(--k_geom_gamma "${K_GEOM_GAMMA}")
fi
mix_args=()
if [[ -n "${MASK_POLICY_MIX}" ]]; then
  mix_args+=(--mask_policy_mix "${MASK_POLICY_MIX}")
fi

echo "==> Train stage-2 (small dist noise, 10k steps)"
"${PYTHON_BIN}" -m src.train.train_interp_levels \
  --dataset d4rl_prepared --prepared_path "${TRAIN_DATA}" \
  --T 128 --K_min "${K_MIN}" --levels "${LEVELS}" --steps "${STEPS}" --batch "${BATCH}" --use_sdf "${USE_SDF}" \
  --cond_start_goal "${COND_START_GOAL}" --clamp_endpoints "${CLAMP_ENDPOINTS}" \
  --stage2_mode adj \
  --kp_index_mode "${KP_INDEX_MODE}" \
  --selector_ckpt "${SELECTOR_CKPT}" \
  --bootstrap_stage1_ckpt "${BOOTSTRAP_STAGE1}" \
  --dphi_ckpt "${DPHI_CKPT}" \
  --k_schedule "${K_SCHEDULE}" "${common_k_schedule[@]}" \
  "${mix_args[@]}" \
  --corrupt_mode dist \
  --corrupt_sigma_max "${CORRUPT_SIGMA_MAX}" \
  --corrupt_sigma_min "${CORRUPT_SIGMA_MIN}" \
  --corrupt_sigma_pow "${CORRUPT_SIGMA_POW}" \
  --corrupt_anchor_frac "${CORRUPT_ANCHOR_FRAC}" \
  --pos_clip "${POS_CLIP}" --pos_clip_min 0.0 --pos_clip_max 1.0 \
  --s2_d_model "${S2_D_MODEL}" --s2_n_layers "${S2_N_LAYERS}" --s2_n_heads "${S2_N_HEADS}" \
  --s2_d_ff "${S2_D_FF}" --s2_d_cond "${S2_D_COND}" --s2_maze_channels "${S2_MAZE_CHANNELS}" \
  --ckpt_dir "${CKPT_DIR}" --log_dir "${LOG_DIR}"

echo "==> Sample stage-2 with new checkpoint"
EVAL_DATA="${EVAL_DATA}" \
CKPT_KEYPOINTS="${BOOTSTRAP_STAGE1}" \
CKPT_ROOT="$(dirname "${CKPT_DIR}")" \
OUT_ROOT="${OUT_ROOT}" \
DPHI_CKPT="${DPHI_CKPT}" \
KP_INDEX_MODE="${KP_INDEX_MODE}" \
SELECTOR_CKPT="${SELECTOR_CKPT}" \
COND_START_GOAL="${COND_START_GOAL}" \
CLAMP_ENDPOINTS="${CLAMP_ENDPOINTS}" \
USE_SDF="${USE_SDF}" \
POS_CLIP="${POS_CLIP}" \
S2_SAMPLE_NOISE_MODE="${S2_SAMPLE_NOISE_MODE}" \
S2_SAMPLE_NOISE_SCALE="${S2_SAMPLE_NOISE_SCALE}" \
S2_SAMPLE_NOISE_SIGMA="${S2_SAMPLE_NOISE_SIGMA}" \
S2_CORRUPT_SIGMA_MAX="${CORRUPT_SIGMA_MAX}" \
S2_CORRUPT_SIGMA_MIN="${CORRUPT_SIGMA_MIN}" \
S2_CORRUPT_SIGMA_POW="${CORRUPT_SIGMA_POW}" \
STAGE1_CACHE_MODE="${STAGE1_CACHE_MODE}" \
SAMPLE_CASES="$(basename "${CKPT_DIR}")" \
bash runs/d4rl_ablate_stage2_sample_only.sh

echo "Done. Samples in: ${OUT_ROOT}/$(basename "${CKPT_DIR}")_sample"
