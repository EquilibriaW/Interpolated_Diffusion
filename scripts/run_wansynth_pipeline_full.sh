#!/usr/bin/env bash
set -euo pipefail

DATA_PATTERN="${DATA_PATTERN:?Set DATA_PATTERN to Wan synth shard pattern}"
WAN_REPO="${WAN_REPO:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
WAN_SUBFOLDER="${WAN_SUBFOLDER:-transformer}"
WAN_DTYPE="${WAN_DTYPE:-bf16}"
FLOW_DTYPE="${FLOW_DTYPE:-${WAN_DTYPE}}"
WAN_ATTN="${WAN_ATTN:-sagesla}"
SLA_TOPK="${SLA_TOPK:-0.1}"

T="${T:-21}"
PATCH_SIZE="${PATCH_SIZE:-4}"
K="${K:-4}"
LEVELS="${LEVELS:-4}"

FLOW_STEPS="${FLOW_STEPS:-20000}"
P1_STEPS="${P1_STEPS:-20000}"
P2_STEPS="${P2_STEPS:-20000}"
BATCH="${BATCH:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SHUFFLE_BUFFER="${SHUFFLE_BUFFER:-1000}"
ANCHOR_NUM_WORKERS="${ANCHOR_NUM_WORKERS:-0}"

FLOW_SAVE_EVERY="${FLOW_SAVE_EVERY:-2000}"
P1_SAVE_EVERY="${P1_SAVE_EVERY:-5000}"
P2_SAVE_EVERY="${P2_SAVE_EVERY:-5000}"

FLOW_CKPT_DIR="${FLOW_CKPT_DIR:-checkpoints/latent_flow_interp_wansynth}"
FLOW_LOG_DIR="${FLOW_LOG_DIR:-runs/latent_flow_interp_wansynth}"
P1_CKPT_DIR="${P1_CKPT_DIR:-checkpoints/keypoints_wansynth}"
P1_LOG_DIR="${P1_LOG_DIR:-runs/keypoints_wansynth}"
ANCHOR_OUT="${ANCHOR_OUT:-data/wan_synth_anchors}"
ANCHOR_JOIN="${ANCHOR_JOIN:-1}"
ANCHOR_KEY_BUFFER="${ANCHOR_KEY_BUFFER:-2000}"
P2_CKPT_DIR="${P2_CKPT_DIR:-checkpoints/interp_levels_wansynth}"
P2_LOG_DIR="${P2_LOG_DIR:-runs/interp_levels_wansynth}"

LORA_RANK="${LORA_RANK:-8}"

echo "==> Train flow interpolator"
python -m src.train.train_flow_interpolator_wansynth \
  --data_pattern "${DATA_PATTERN}" \
  --T "${T}" \
  --steps "${FLOW_STEPS}" \
  --batch "${BATCH}" \
  --num_workers "${NUM_WORKERS}" \
  --shuffle_buffer "${SHUFFLE_BUFFER}" \
  --model_dtype "${FLOW_DTYPE}" \
  --save_every "${FLOW_SAVE_EVERY}" \
  --ckpt_dir "${FLOW_CKPT_DIR}" \
  --log_dir "${FLOW_LOG_DIR}"

FLOW_CKPT="${FLOW_CKPT:-${FLOW_CKPT_DIR}/ckpt_final.pt}"

echo "==> Train phase-1 keypoints (Wan2.1 + LoRA)"
python -m src.train.train_keypoints_wansynth \
  --data_pattern "${DATA_PATTERN}" \
  --T "${T}" \
  --K "${K}" \
  --patch_size "${PATCH_SIZE}" \
  --steps "${P1_STEPS}" \
  --batch "${BATCH}" \
  --num_workers "${NUM_WORKERS}" \
  --shuffle_buffer "${SHUFFLE_BUFFER}" \
  --save_every "${P1_SAVE_EVERY}" \
  --use_wan 1 \
  --wan_repo "${WAN_REPO}" \
  --wan_subfolder "${WAN_SUBFOLDER}" \
  --wan_dtype "${WAN_DTYPE}" \
  --wan_attn "${WAN_ATTN}" \
  --sla_topk "${SLA_TOPK}" \
  --video_interp_mode flow \
  --flow_interp_ckpt "${FLOW_CKPT}" \
  --lora_rank "${LORA_RANK}" \
  --ckpt_dir "${P1_CKPT_DIR}" \
  --log_dir "${P1_LOG_DIR}"

P1_CKPT="${P1_CKPT:-${P1_CKPT_DIR}/ckpt_final.pt}"

echo "==> Precompute phase-1 anchors"
PYTHONPATH=. python scripts/datasets/wan_synth/precompute_phase1_anchors.py \
  --data_pattern "${DATA_PATTERN}" \
  --out_dir "${ANCHOR_OUT}" \
  --phase1_ckpt "${P1_CKPT}" \
  --T "${T}" \
  --K "${K}" \
  --patch_size "${PATCH_SIZE}" \
  --num_workers "${ANCHOR_NUM_WORKERS}" \
  --shuffle_buffer "${SHUFFLE_BUFFER}" \
  --use_wan 1 \
  --wan_repo "${WAN_REPO}" \
  --wan_subfolder "${WAN_SUBFOLDER}" \
  --wan_dtype "${WAN_DTYPE}" \
  --wan_attn "${WAN_ATTN}" \
  --sla_topk "${SLA_TOPK}" \
  --video_interp_mode flow \
  --flow_interp_ckpt "${FLOW_CKPT}"

ANCHOR_PATTERN="${ANCHOR_PATTERN:-${ANCHOR_OUT}/anchor-*.tar}"

echo "==> Train phase-2 interp-levels (Wan2.1 + LoRA)"
python -m src.train.train_interp_levels_wansynth \
  --data_pattern "${DATA_PATTERN}" \
  --anchor_pattern "${ANCHOR_PATTERN}" \
  --T "${T}" \
  --patch_size "${PATCH_SIZE}" \
  --K_min "${K}" \
  --levels "${LEVELS}" \
  --steps "${P2_STEPS}" \
  --batch "${BATCH}" \
  --num_workers "${NUM_WORKERS}" \
  --shuffle_buffer "${SHUFFLE_BUFFER}" \
  --anchor_join "${ANCHOR_JOIN}" \
  --anchor_key_buffer "${ANCHOR_KEY_BUFFER}" \
  --save_every "${P2_SAVE_EVERY}" \
  --use_wan 1 \
  --wan_repo "${WAN_REPO}" \
  --wan_subfolder "${WAN_SUBFOLDER}" \
  --wan_dtype "${WAN_DTYPE}" \
  --wan_attn "${WAN_ATTN}" \
  --sla_topk "${SLA_TOPK}" \
  --video_interp_mode flow \
  --flow_interp_ckpt "${FLOW_CKPT}" \
  --flow_uncertainty_mode replace \
  --student_replace_prob 0.5 \
  --lora_rank "${LORA_RANK}" \
  --ckpt_dir "${P2_CKPT_DIR}" \
  --log_dir "${P2_LOG_DIR}"

echo "==> Full pipeline complete"
