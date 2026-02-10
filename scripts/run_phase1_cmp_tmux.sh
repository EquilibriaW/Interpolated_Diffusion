#!/usr/bin/env bash
set -euo pipefail

# Phase-1 comparison: short_midpoints vs short_meanpool (absolute-time RoPE enabled in code).
#
# Usage:
#   bash scripts/run_phase1_cmp_tmux.sh
#
# Optional environment overrides:
#   SESSION=phase1_cmp
#   DATA_PATTERN='data/wan_synth/.../shard-0000[0-7].tar'
#   BATCH=8 STEPS=10000 K=4 T=21
#   WAN_ATTN=sagesla SLA_TOPK=0.07 WAN_DTYPE=bfloat16 GRAD_CKPT=1
#   NUM_WORKERS=4 SHUFFLE_BUFFER=32
#   SAVE_EVERY=2500 SAVE_OPTIMIZER=0 MAX_CPU_MEM_PERCENT=98
#   VAL_PATTERN='data/wan_synth/.../shard-0000[8-9].tar' VAL_EVERY=500 VAL_BATCHES=10 VAL_NUM_WORKERS=0
#   TB_PORT=6006

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SESSION="${SESSION:-phase1_cmp}"

DATA_PATTERN="${DATA_PATTERN:-data/wan_synth/Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K/shard-0000[0-7].tar}"

WAN_REPO="${WAN_REPO:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
WAN_SUBFOLDER="${WAN_SUBFOLDER:-transformer}"
WAN_ATTN="${WAN_ATTN:-sagesla}"
SLA_TOPK="${SLA_TOPK:-0.07}"
WAN_DTYPE="${WAN_DTYPE:-bfloat16}"
GRAD_CKPT="${GRAD_CKPT:-1}"
WAN_FRAME_COND="${WAN_FRAME_COND:-1}"

SEED="${SEED:-0}"
K="${K:-4}"
T="${T:-21}"
BATCH="${BATCH:-8}"
STEPS="${STEPS:-10000}"
LOG_EVERY="${LOG_EVERY:-20}"

NUM_WORKERS="${NUM_WORKERS:-4}"
SHUFFLE_BUFFER="${SHUFFLE_BUFFER:-32}"

SAVE_EVERY="${SAVE_EVERY:-2500}"
SAVE_OPTIMIZER="${SAVE_OPTIMIZER:-0}"
MAX_CPU_MEM_PERCENT="${MAX_CPU_MEM_PERCENT:-98}"

VAL_PATTERN="${VAL_PATTERN:-}"
VAL_EVERY="${VAL_EVERY:-0}"
VAL_BATCHES="${VAL_BATCHES:-10}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-0}"

TB_PORT="${TB_PORT:-6006}"

TAG="${TAG:-$(date +%Y%m%d_%H%M%S)_b${BATCH}_s${STEPS}_k${K}_L$((2 * K - 1))}"
CKPT_BASE="${CKPT_BASE:-checkpoints/phase1_cmp/${TAG}}"
RUN_BASE="${RUN_BASE:-runs/phase1_cmp/${TAG}}"
LOG_BASE="${LOG_BASE:-logs/phase1_cmp}"

mkdir -p "${CKPT_BASE}" "${RUN_BASE}" "${LOG_BASE}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session '${SESSION}' already exists."
  echo "Attach with: tmux attach -t ${SESSION}"
  exit 1
fi

MID_LOG="${LOG_BASE}/${TAG}_midpoints.log"
MP_LOG="${LOG_BASE}/${TAG}_meanpool.log"

MID_CKPT="${CKPT_BASE}/midpoints"
MP_CKPT="${CKPT_BASE}/meanpool"
MID_RUN="${RUN_BASE}/midpoints"
MP_RUN="${RUN_BASE}/meanpool"

mkdir -p "${MID_CKPT}" "${MP_CKPT}" "${MID_RUN}" "${MP_RUN}"

COMMON_ARGS=(
  --data_pattern "${DATA_PATTERN}"
  ${VAL_PATTERN:+--val_pattern} ${VAL_PATTERN:+"${VAL_PATTERN}"}
  --val_every "${VAL_EVERY}"
  --val_batches "${VAL_BATCHES}"
  --val_num_workers "${VAL_NUM_WORKERS}"
  --use_wan 1
  --wan_repo "${WAN_REPO}"
  --wan_subfolder "${WAN_SUBFOLDER}"
  --wan_dtype "${WAN_DTYPE}"
  --wan_attn "${WAN_ATTN}"
  --sla_topk "${SLA_TOPK}"
  --grad_ckpt "${GRAD_CKPT}"
  --wan_frame_cond "${WAN_FRAME_COND}"
  --seed "${SEED}"
  --K "${K}"
  --T "${T}"
  --batch "${BATCH}"
  --steps "${STEPS}"
  --log_every "${LOG_EVERY}"
  --num_workers "${NUM_WORKERS}"
  --shuffle_buffer "${SHUFFLE_BUFFER}"
  --save_every "${SAVE_EVERY}"
  --save_final 1
  --save_optimizer "${SAVE_OPTIMIZER}"
  --max_cpu_mem_percent "${MAX_CPU_MEM_PERCENT}"
)

MID_CMD=(python -u -m src.train.train_keypoints_wansynth "${COMMON_ARGS[@]}" --phase1_input_mode short_midpoints --ckpt_dir "${MID_CKPT}" --log_dir "${MID_RUN}")
MP_CMD=(python -u -m src.train.train_keypoints_wansynth "${COMMON_ARGS[@]}" --phase1_input_mode short_meanpool --ckpt_dir "${MP_CKPT}" --log_dir "${MP_RUN}")

quote_cmd() {
  # Print a shell-escaped command line so patterns like shard-0000[0-7].tar are not glob-expanded.
  printf '%q ' "$@"
}

MID_CMD_STR="$(quote_cmd "${MID_CMD[@]}")"
MP_CMD_STR="$(quote_cmd "${MP_CMD[@]}")"

tmux new-session -d -s "${SESSION}" -c "${ROOT_DIR}" "bash -lc '
set -euo pipefail
echo \"[phase1_cmp] tag=${TAG}\"
echo \"[phase1_cmp] midpoints -> ${MID_RUN}\"
echo \"[phase1_cmp] meanpool  -> ${MP_RUN}\"

echo \"[phase1_cmp] starting midpoints...\"
${MID_CMD_STR} |& tee \"${MID_LOG}\"

echo \"[phase1_cmp] starting meanpool...\"
${MP_CMD_STR} |& tee \"${MP_LOG}\"

echo \"[phase1_cmp] DONE\"
exec bash
'"

tmux rename-window -t "${SESSION}:0" train

# Best-effort TensorBoard window (can be killed/restarted if port is busy).
tmux new-window -t "${SESSION}" -n tb -c "${ROOT_DIR}" "bash -lc '
set -euo pipefail
echo \"[tensorboard] logdir=runs/phase1_cmp port=${TB_PORT}\"
tensorboard --logdir runs/phase1_cmp --bind_all --port ${TB_PORT}
exec bash
'"

echo "Started tmux session: ${SESSION}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Midpoints log: ${MID_LOG}"
echo "Meanpool  log: ${MP_LOG}"
