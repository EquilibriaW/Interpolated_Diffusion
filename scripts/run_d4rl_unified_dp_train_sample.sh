#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_PY_MUJOCO_PATH=${MUJOCO_PY_MUJOCO_PATH:-/workspace/Interpolated_Diffusion/mujoco210}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/workspace/Interpolated_Diffusion/mujoco210/bin:${LD_LIBRARY_PATH:-}}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export D4RL_SUPPRESS_IMPORT_ERROR=${D4RL_SUPPRESS_IMPORT_ERROR:-1}
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}

PYTHON=${PYTHON:-.venv/bin/python}

T=${T:-128}
K=${K:-8}
K_MIN=${K_MIN:-8}
LEVELS=${LEVELS:-8}
STEPS=${STEPS:-20000}
BATCH=${BATCH:-256}
USE_SDF=${USE_SDF:-1}
DETERMINISTIC=${DETERMINISTIC:-0}
COND_START_GOAL=${COND_START_GOAL:-1}
CLAMP_ENDPOINTS=${CLAMP_ENDPOINTS:-1}
POS_CLIP=${POS_CLIP:-1}
POS_CLIP_MIN=${POS_CLIP_MIN:-0.0}
POS_CLIP_MAX=${POS_CLIP_MAX:-1.0}

K_SCHEDULE=${K_SCHEDULE:-geom}
K_GEOM_GAMMA=${K_GEOM_GAMMA:-}

TRAIN_DATA=${TRAIN_DATA:-outputs/d4rl_prepared_unified_T${T}_train/dataset.npz}
EVAL_DATA=${EVAL_DATA:-outputs/d4rl_prepared_unified_T${T}_eval/dataset.npz}

DP_TRAIN_DATA=${DP_TRAIN_DATA:-outputs/d4rl_prepared_unified_T${T}_train_dp/dataset.npz}
DP_EVAL_DATA=${DP_EVAL_DATA:-outputs/d4rl_prepared_unified_T${T}_eval_dp/dataset.npz}
SELECTOR_DP_TRAIN_DATA=${SELECTOR_DP_TRAIN_DATA:-outputs/d4rl_prepared_unified_T${T}_train_dp_dphi/dataset.npz}
SELECTOR_DP_EVAL_DATA=${SELECTOR_DP_EVAL_DATA:-outputs/d4rl_prepared_unified_T${T}_eval_dp_dphi/dataset.npz}

DP_SUBSET_FRAC=${DP_SUBSET_FRAC:-0.1}
DP_SUBSET_NUM=${DP_SUBSET_NUM:-0}
DP_BATCH=${DP_BATCH:-64}
DP_DEVICE=${DP_DEVICE:-cuda}
DP_T_STEPS=${DP_T_STEPS:-16}
DP_SEG_SAMPLES=${DP_SEG_SAMPLES:-16}
DP_SNR_MIN=${DP_SNR_MIN:-0.1}
DP_SNR_MAX=${DP_SNR_MAX:-10.0}
DP_SNR_GAMMA=${DP_SNR_GAMMA:-1.0}
DP_PER_SAMPLE=${DP_PER_SAMPLE:-1}

IDX_POLICY_MIX=${IDX_POLICY_MIX:-selector:0.7,uniform:0.2,random:0.1}
USE_KP_FEAT=${USE_KP_FEAT:-1}
KP_FEAT_DIM=${KP_FEAT_DIM:-5}
KP_D_MODEL=${KP_D_MODEL:-384}
KP_N_LAYERS=${KP_N_LAYERS:-12}
KP_N_HEADS=${KP_N_HEADS:-12}
KP_D_FF=${KP_D_FF:-1536}
KP_D_COND=${KP_D_COND:-128}
KP_MAZE_CHANNELS=${KP_MAZE_CHANNELS:-32,64,128,128}
S2_D_MODEL=${S2_D_MODEL:-384}
S2_N_LAYERS=${S2_N_LAYERS:-12}
S2_N_HEADS=${S2_N_HEADS:-12}
S2_D_FF=${S2_D_FF:-1536}
S2_D_COND=${S2_D_COND:-128}
S2_MAZE_CHANNELS=${S2_MAZE_CHANNELS:-${KP_MAZE_CHANNELS}}

DPHI_STEPS=${DPHI_STEPS:-10000}
DPHI_BATCH=${DPHI_BATCH:-64}
DPHI_CKPT=${DPHI_CKPT:-checkpoints/segment_cost_unified_T${T}_dp}
DPHI_LOG=${DPHI_LOG:-runs/segment_cost_unified_T${T}_dp}
DPHI_STATS_SUBSET=${DPHI_STATS_SUBSET:-512}
DPHI_MAZE_CHANNELS=${DPHI_MAZE_CHANNELS:-${KP_MAZE_CHANNELS}}

SELECTOR_STEPS=${SELECTOR_STEPS:-10000}
SELECTOR_BATCH=${SELECTOR_BATCH:-128}
SELECTOR_CKPT=${SELECTOR_CKPT:-checkpoints/keypoint_selector_unified_T${T}_dp}
SELECTOR_LOG=${SELECTOR_LOG:-runs/keypoint_selector_unified_T${T}_dp}
SELECTOR_D_MODEL=${SELECTOR_D_MODEL:-256}
SELECTOR_N_HEADS=${SELECTOR_N_HEADS:-8}
SELECTOR_D_FF=${SELECTOR_D_FF:-512}
SELECTOR_N_LAYERS=${SELECTOR_N_LAYERS:-2}
SELECTOR_POS_DIM=${SELECTOR_POS_DIM:-64}
SELECTOR_DROPOUT=${SELECTOR_DROPOUT:-0.0}
SELECTOR_USE_SG_MAP=${SELECTOR_USE_SG_MAP:-1}
SELECTOR_USE_SG_TOKEN=${SELECTOR_USE_SG_TOKEN:-1}
SELECTOR_USE_GOAL_DIST_TOKEN=${SELECTOR_USE_GOAL_DIST_TOKEN:-1}
SELECTOR_USE_COND_BIAS=${SELECTOR_USE_COND_BIAS:-1}
SELECTOR_COND_BIAS_MODE=${SELECTOR_COND_BIAS_MODE:-encoder}
SELECTOR_USE_LEVEL=${SELECTOR_USE_LEVEL:-1}
SELECTOR_LEVEL_MODE=${SELECTOR_LEVEL_MODE:-k_norm}
SELECTOR_SG_MAP_SIGMA=${SELECTOR_SG_MAP_SIGMA:-1.5}
SELECTOR_KL_WEIGHT=${SELECTOR_KL_WEIGHT:-0.02}
SELECTOR_TAU_START=${SELECTOR_TAU_START:-1.0}
SELECTOR_TAU_END=${SELECTOR_TAU_END:-0.3}
SELECTOR_TAU_ANNEAL=${SELECTOR_TAU_ANNEAL:-cosine}
SELECTOR_TAU_FRAC=${SELECTOR_TAU_FRAC:-0.8}
SELECTOR_MAZE_CHANNELS=${SELECTOR_MAZE_CHANNELS:-${KP_MAZE_CHANNELS}}

S2_KP_INDEX_MODE=${S2_KP_INDEX_MODE:-selector}

CKPT_KP=${CKPT_KP:-checkpoints/keypoints_unified_T${T}_dp}
CKPT_S2=${CKPT_S2:-checkpoints/interp_levels_unified_T${T}_dp}
LOG_KP=${LOG_KP:-runs/keypoints_unified_T${T}_dp}
LOG_S2=${LOG_S2:-runs/interp_levels_unified_T${T}_dp}
OUT_ROOT=${OUT_ROOT:-runs/ablate_stage2_unified_dp}

common_k_schedule=()
if [[ -n "${K_GEOM_GAMMA}" ]]; then
  common_k_schedule+=(--k_geom_gamma "${K_GEOM_GAMMA}")
fi

echo "==> Build DP keypoint indices from train (subset)"
PYTHONPATH=. ${PYTHON} src/data/prepare_dp_keypoints.py \
  --in_npz "${TRAIN_DATA}" \
  --out_npz "${DP_TRAIN_DATA}" \
  --K "${K}" --schedule cosine --N_train 1000 \
  --snr_min "${DP_SNR_MIN}" --snr_max "${DP_SNR_MAX}" --snr_gamma "${DP_SNR_GAMMA}" \
  --t_steps "${DP_T_STEPS}" --segment_cost_samples "${DP_SEG_SAMPLES}" \
  --subset_frac "${DP_SUBSET_FRAC}" --subset_num "${DP_SUBSET_NUM}" --per_sample "${DP_PER_SAMPLE}" \
  --batch "${DP_BATCH}" --device "${DP_DEVICE}"

echo "==> Apply DP keypoint indices to eval (reuse train indices)"
if [[ "${DP_PER_SAMPLE}" == "1" ]]; then
  PYTHONPATH=. ${PYTHON} src/data/prepare_dp_keypoints.py \
    --in_npz "${EVAL_DATA}" \
    --out_npz "${DP_EVAL_DATA}" \
    --K "${K}" --schedule cosine --N_train 1000 \
    --snr_min "${DP_SNR_MIN}" --snr_max "${DP_SNR_MAX}" --snr_gamma "${DP_SNR_GAMMA}" \
    --t_steps "${DP_T_STEPS}" --segment_cost_samples "${DP_SEG_SAMPLES}" \
    --per_sample 1 \
    --batch "${DP_BATCH}" --device "${DP_DEVICE}"
else
  PYTHONPATH=. ${PYTHON} src/data/prepare_dp_keypoints.py \
    --in_npz "${EVAL_DATA}" \
    --out_npz "${DP_EVAL_DATA}" \
    --K "${K}" --kp_idx_from "${DP_TRAIN_DATA}"
fi

echo "==> Train segment-cost predictor (D_phi)"
PYTHONPATH=. ${PYTHON} -m src.train.train_segment_cost \
  --prepared_path "${DP_TRAIN_DATA}" \
  --T "${T}" --batch "${DPHI_BATCH}" --steps "${DPHI_STEPS}" \
  --use_sdf "${USE_SDF}" --cond_start_goal "${COND_START_GOAL}" \
  --segment_cost_samples "${DP_SEG_SAMPLES}" \
  --maze_channels "${DPHI_MAZE_CHANNELS}" \
  --schedule cosine --N_train 1000 \
  --snr_min "${DP_SNR_MIN}" --snr_max "${DP_SNR_MAX}" --snr_gamma "${DP_SNR_GAMMA}" \
  --t_steps "${DP_T_STEPS}" \
  --stats_subset "${DPHI_STATS_SUBSET}" \
  --ckpt_dir "${DPHI_CKPT}" --log_dir "${DPHI_LOG}"

echo "==> Build selector DP labels from D_phi (cond-only)"
PYTHONPATH=. ${PYTHON} src/data/prepare_dp_keypoints.py \
  --in_npz "${TRAIN_DATA}" \
  --out_npz "${SELECTOR_DP_TRAIN_DATA}" \
  --K "${K}" --segment_cost_samples "${DP_SEG_SAMPLES}" \
  --store_kp_mask_levels 1 --levels "${LEVELS}" --k_schedule "${K_SCHEDULE}" ${K_GEOM_GAMMA:+--k_geom_gamma "${K_GEOM_GAMMA}"} \
  --cost_source dphi --dphi_ckpt "${DPHI_CKPT}/ckpt_final.pt" \
  --per_sample 1 \
  --batch "${DP_BATCH}" --device "${DP_DEVICE}"

PYTHONPATH=. ${PYTHON} src/data/prepare_dp_keypoints.py \
  --in_npz "${EVAL_DATA}" \
  --out_npz "${SELECTOR_DP_EVAL_DATA}" \
  --K "${K}" --segment_cost_samples "${DP_SEG_SAMPLES}" \
  --store_kp_mask_levels 1 --levels "${LEVELS}" --k_schedule "${K_SCHEDULE}" ${K_GEOM_GAMMA:+--k_geom_gamma "${K_GEOM_GAMMA}"} \
  --cost_source dphi --dphi_ckpt "${DPHI_CKPT}/ckpt_final.pt" \
  --per_sample 1 \
  --batch "${DP_BATCH}" --device "${DP_DEVICE}"

echo "==> Train keypoint selector (cond-only)"
PYTHONPATH=. ${PYTHON} -m src.train.train_keypoint_selector \
  --prepared_path "${SELECTOR_DP_TRAIN_DATA}" \
  --T "${T}" --K "${K}" --batch "${SELECTOR_BATCH}" --steps "${SELECTOR_STEPS}" \
  --use_sdf "${USE_SDF}" --cond_start_goal "${COND_START_GOAL}" \
  --d_model "${SELECTOR_D_MODEL}" --n_heads "${SELECTOR_N_HEADS}" --d_ff "${SELECTOR_D_FF}" \
  --n_layers "${SELECTOR_N_LAYERS}" --pos_dim "${SELECTOR_POS_DIM}" \
  --dropout "${SELECTOR_DROPOUT}" --maze_channels "${SELECTOR_MAZE_CHANNELS}" \
  --use_sg_map "${SELECTOR_USE_SG_MAP}" --use_sg_token "${SELECTOR_USE_SG_TOKEN}" \
  --use_goal_dist_token "${SELECTOR_USE_GOAL_DIST_TOKEN}" \
  --use_cond_bias "${SELECTOR_USE_COND_BIAS}" \
  --cond_bias_mode "${SELECTOR_COND_BIAS_MODE}" \
  --use_level "${SELECTOR_USE_LEVEL}" --level_mode "${SELECTOR_LEVEL_MODE}" \
  --levels "${LEVELS}" --k_schedule "${K_SCHEDULE}" ${K_GEOM_GAMMA:+--k_geom_gamma "${K_GEOM_GAMMA}"} \
  --sg_map_sigma "${SELECTOR_SG_MAP_SIGMA}" \
  --sel_kl_weight "${SELECTOR_KL_WEIGHT}" \
  --sel_tau_start "${SELECTOR_TAU_START}" \
  --sel_tau_end "${SELECTOR_TAU_END}" \
  --sel_tau_anneal "${SELECTOR_TAU_ANNEAL}" \
  --sel_tau_frac "${SELECTOR_TAU_FRAC}" \
  --ckpt_dir "${SELECTOR_CKPT}" --log_dir "${SELECTOR_LOG}"

echo "==> Train unified stage-1 keypoints (DP mix)"
PYTHONPATH=. ${PYTHON} -m src.train.train_keypoints \
  --dataset d4rl_prepared --prepared_path "${DP_TRAIN_DATA}" \
  --T "${T}" --K "${K}" --steps "${STEPS}" --batch "${BATCH}" --use_sdf "${USE_SDF}" \
  --cond_start_goal "${COND_START_GOAL}" --clamp_endpoints "${CLAMP_ENDPOINTS}" \
  --deterministic "${DETERMINISTIC}" \
  --logit_space 0 \
  --idx_policy_mix "${IDX_POLICY_MIX}" \
  --use_kp_feat "${USE_KP_FEAT}" --kp_feat_dim "${KP_FEAT_DIM}" \
  --kp_d_model "${KP_D_MODEL}" --kp_n_layers "${KP_N_LAYERS}" --kp_n_heads "${KP_N_HEADS}" \
  --kp_d_ff "${KP_D_FF}" --kp_d_cond "${KP_D_COND}" --kp_maze_channels "${KP_MAZE_CHANNELS}" \
  --dphi_ckpt "${DPHI_CKPT}/ckpt_final.pt" \
  --selector_ckpt "${SELECTOR_CKPT}/ckpt_final.pt" \
  --ckpt_dir "${CKPT_KP}" --log_dir "${LOG_KP}"

echo "==> Train unified stage-2 (dist corruption)"
PYTHONPATH=. ${PYTHON} -m src.train.train_interp_levels \
  --dataset d4rl_prepared --prepared_path "${DP_TRAIN_DATA}" \
  --T "${T}" --K_min "${K_MIN}" --levels "${LEVELS}" --steps "${STEPS}" --batch "${BATCH}" --use_sdf "${USE_SDF}" \
  --deterministic "${DETERMINISTIC}" \
  --clamp_endpoints "${CLAMP_ENDPOINTS}" --cond_start_goal "${COND_START_GOAL}" \
  --pos_clip "${POS_CLIP}" --pos_clip_min "${POS_CLIP_MIN}" --pos_clip_max "${POS_CLIP_MAX}" \
  --stage2_mode adj \
  --kp_index_mode "${S2_KP_INDEX_MODE}" \
  --selector_ckpt "${SELECTOR_CKPT}/ckpt_final.pt" \
  --k_schedule "${K_SCHEDULE}" "${common_k_schedule[@]}" \
  --corrupt_mode dist \
  --corrupt_sigma_max 0.08 --corrupt_sigma_min 0.012 --corrupt_sigma_pow 0.75 \
  --corrupt_anchor_frac 0.25 \
  --s2_d_model "${S2_D_MODEL}" --s2_n_layers "${S2_N_LAYERS}" --s2_n_heads "${S2_N_HEADS}" \
  --s2_d_ff "${S2_D_FF}" --s2_d_cond "${S2_D_COND}" --s2_maze_channels "${S2_MAZE_CHANNELS}" \
  --bootstrap_stage1_ckpt "${CKPT_KP}/ckpt_final.pt" \
  --dphi_ckpt "${DPHI_CKPT}/ckpt_final.pt" \
  --bootstrap_ddim_steps 20 --bootstrap_ddim_schedule quadratic \
  --bootstrap_prob_start 0.0 --bootstrap_prob_end 0.5 --bootstrap_warmup_steps 5000 \
  --ckpt_dir "${CKPT_S2}" --log_dir "${LOG_S2}"

echo "==> Sample unified eval (points-based plotting)"
EVAL_DATA="${DP_EVAL_DATA}" \
CKPT_ROOT=${CKPT_ROOT:-checkpoints} \
CKPT_KEYPOINTS="${CKPT_KP}/ckpt_final.pt" \
SAMPLE_CASES=${SAMPLE_CASES:-interp_levels_unified_T${T}_dp} \
OUT_ROOT=${OUT_ROOT:-runs/ablate_stage2_unified_dp} \
DPHI_CKPT="${DPHI_CKPT}/ckpt_final.pt" \
SELECTOR_CKPT="${SELECTOR_CKPT}/ckpt_final.pt" \
KP_INDEX_MODE=${KP_INDEX_MODE:-selector} \
STAGE1_CACHE_MODE=${STAGE1_CACHE_MODE:-none} \
COND_START_GOAL="${COND_START_GOAL}" \
CLAMP_ENDPOINTS="${CLAMP_ENDPOINTS}" \
USE_SDF="${USE_SDF}" \
POS_CLIP="${POS_CLIP}" \
bash runs/d4rl_ablate_stage2_sample_only.sh

echo "==> Done"

if [[ "${LOG_EXPERIMENT:-0}" == "1" ]]; then
  EXP_NAME=${EXP_NAME:-"d4rl_unified_dp"}
  EXP_COMMAND=${EXP_COMMAND:-"bash scripts/run_d4rl_unified_dp_train_sample.sh"}
  KEY_SETTINGS=${KEY_SETTINGS:-"T=${T},K=${K},levels=${LEVELS},steps=${STEPS},batch=${BATCH},idx_mix=${IDX_POLICY_MIX},kp_feat_dim=${KP_FEAT_DIM},kp_model=${KP_D_MODEL}x${KP_N_LAYERS},s2_model=${S2_D_MODEL}x${S2_N_LAYERS},maze_channels=${KP_MAZE_CHANNELS},dp_per_sample=${DP_PER_SAMPLE},snr=[${DP_SNR_MIN},${DP_SNR_MAX},${DP_SNR_GAMMA}],pos_clip=${POS_CLIP}"}
  RESULTS_DIR=${RESULTS_DIR:-"${OUT_ROOT}"}
  LOG_PATH=${LOG_PATH:-notes/PROJECT_LOG.md} \
    EXP_NAME="${EXP_NAME}" \
    EXP_COMMAND="${EXP_COMMAND}" \
    TRAIN_DATA="${DP_TRAIN_DATA}" \
    EVAL_DATA="${DP_EVAL_DATA}" \
    CKPT_KP="${CKPT_KP}" \
    CKPT_S2="${CKPT_S2}" \
    DPHI_CKPT="${DPHI_CKPT}" \
    LOG_KP="${LOG_KP}" \
    LOG_S2="${LOG_S2}" \
    DPHI_LOG="${DPHI_LOG}" \
    RESULTS_DIR="${RESULTS_DIR}" \
    KEY_SETTINGS="${KEY_SETTINGS}" \
    scripts/log_experiment.sh
fi
