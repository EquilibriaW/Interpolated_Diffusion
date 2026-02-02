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
LEVELS=8
STEPS=20000
BATCH=256
USE_SDF=1
DETERMINISTIC=0

K_SCHEDULE=geom
K_GEOM_GAMMA=

DDIM_STEPS=100
DDIM_SCHEDULE=linear
N_SAMPLES=50
SAMPLE_BATCH=50

CKPT_KP=checkpoints/keypoints_unified_T${T}
CKPT_S2=checkpoints/interp_levels_unified_T${T}
LOG_KP=runs/keypoints_unified_T${T}
LOG_S2=runs/interp_levels_unified_T${T}

TRAIN_OUT=outputs/d4rl_prepared_unified_T${T}_train
EVAL_OUT=outputs/d4rl_prepared_unified_T${T}_eval

inputs_train=()
inputs_eval=()
for env in "${ENVS[@]}"; do
  inputs_train+=( "outputs/d4rl_prepared_${env}_T${T}_train/dataset.npz" )
  inputs_eval+=( "outputs/d4rl_prepared_${env}_T${T}_eval/dataset.npz" )
done

echo "==> Build unified train dataset"
PYTHONPATH=. .venv/bin/python scripts/datasets/d4rl/build_unified_prepared.py \
  --inputs "${inputs_train[@]}" \
  --out_dir "${TRAIN_OUT}" \
  --use_sdf "${USE_SDF}" \
  --resize_mode pad \
  --pad_scale_mode none

echo "==> Build unified eval dataset"
PYTHONPATH=. .venv/bin/python scripts/datasets/d4rl/build_unified_prepared.py \
  --inputs "${inputs_eval[@]}" \
  --out_dir "${EVAL_OUT}" \
  --use_sdf "${USE_SDF}" \
  --resize_mode pad \
  --pad_scale_mode none

common_k_schedule=()
if [[ -n "${K_GEOM_GAMMA}" ]]; then
  common_k_schedule+=(--k_geom_gamma "${K_GEOM_GAMMA}")
fi

echo "==> Train unified stage-1 keypoints"
PYTHONPATH=. .venv/bin/python -m src.train.train_keypoints \
  --dataset d4rl_prepared --prepared_path "${TRAIN_OUT}/dataset.npz" \
  --T "${T}" --K "${K}" --steps "${STEPS}" --batch "${BATCH}" --use_sdf "${USE_SDF}" \
  --deterministic "${DETERMINISTIC}" \
  --ckpt_dir "${CKPT_KP}" --log_dir "${LOG_KP}"

echo "==> Train unified stage-2 (dist corruption)"
PYTHONPATH=. .venv/bin/python -m src.train.train_interp_levels \
  --dataset d4rl_prepared --prepared_path "${TRAIN_OUT}/dataset.npz" \
  --T "${T}" --K_min "${K_MIN}" --levels "${LEVELS}" --steps "${STEPS}" --batch "${BATCH}" --use_sdf "${USE_SDF}" \
  --deterministic "${DETERMINISTIC}" \
  --clamp_endpoints 1 --cond_start_goal 0 \
  --stage2_mode adj \
  --k_schedule "${K_SCHEDULE}" "${common_k_schedule[@]}" \
  --corrupt_mode dist \
  --corrupt_sigma_max 0.08 --corrupt_sigma_min 0.012 --corrupt_sigma_pow 0.75 \
  --corrupt_anchor_frac 0.25 \
  --bootstrap_stage1_ckpt "${CKPT_KP}/ckpt_final.pt" \
  --bootstrap_ddim_steps 20 --bootstrap_ddim_schedule quadratic \
  --bootstrap_prob_start 0.0 --bootstrap_prob_end 0.5 --bootstrap_warmup_steps 5000 \
  --ckpt_dir "${CKPT_S2}" --log_dir "${LOG_S2}"

echo "==> Sample unified train"
PYTHONPATH=. .venv/bin/python -m src.sample.sample_generate \
  --dataset d4rl_prepared --prepared_path "${TRAIN_OUT}/dataset.npz" \
  --ckpt_keypoints "${CKPT_KP}/ckpt_final.pt" \
  --ckpt_interp "${CKPT_S2}/ckpt_final.pt" \
  --clamp_endpoints 1 --cond_start_goal 0 \
  --stage2_mode adj \
  --anchor_conf 1 \
  --anchor_conf_teacher 0.95 \
  --anchor_conf_student 0.5 \
  --anchor_conf_endpoints 1.0 \
  --anchor_conf_missing 0.0 \
  --anchor_conf_anneal 1 \
  --anchor_conf_anneal_mode linear \
  --soft_anchor_clamp 1 \
  --soft_clamp_schedule linear \
  --soft_clamp_max 0.1 \
  --compare_oracle 1 --plot_keypoints 1 --plot_points 1 \
  --kp_index_mode uniform \
  --ddim_steps "${DDIM_STEPS}" --ddim_schedule "${DDIM_SCHEDULE}" \
  --T "${T}" --K_min "${K_MIN}" --levels "${LEVELS}" --k_schedule "${K_SCHEDULE}" "${common_k_schedule[@]}" \
  --n_samples "${N_SAMPLES}" --batch "${SAMPLE_BATCH}" \
  --save_diffusion_frames 1 --frames_stride 1 --frames_include_stage2 1 \
  --export_video mp4 --video_fps 8 \
  --out_dir "runs/gen_unified_train"

echo "==> Sample unified eval"
PYTHONPATH=. .venv/bin/python -m src.sample.sample_generate \
  --dataset d4rl_prepared --prepared_path "${EVAL_OUT}/dataset.npz" \
  --ckpt_keypoints "${CKPT_KP}/ckpt_final.pt" \
  --ckpt_interp "${CKPT_S2}/ckpt_final.pt" \
  --clamp_endpoints 1 --cond_start_goal 0 \
  --stage2_mode adj \
  --anchor_conf 1 \
  --anchor_conf_teacher 0.95 \
  --anchor_conf_student 0.5 \
  --anchor_conf_endpoints 1.0 \
  --anchor_conf_missing 0.0 \
  --anchor_conf_anneal 1 \
  --anchor_conf_anneal_mode linear \
  --soft_anchor_clamp 1 \
  --soft_clamp_schedule linear \
  --soft_clamp_max 0.1 \
  --compare_oracle 1 --plot_keypoints 1 --plot_points 1 \
  --kp_index_mode uniform \
  --ddim_steps "${DDIM_STEPS}" --ddim_schedule "${DDIM_SCHEDULE}" \
  --T "${T}" --K_min "${K_MIN}" --levels "${LEVELS}" --k_schedule "${K_SCHEDULE}" "${common_k_schedule[@]}" \
  --n_samples "${N_SAMPLES}" --batch "${SAMPLE_BATCH}" \
  --save_diffusion_frames 1 --frames_stride 1 --frames_include_stage2 1 \
  --export_video mp4 --video_fps 8 \
  --out_dir "runs/gen_unified_eval"

echo "==> Done"
