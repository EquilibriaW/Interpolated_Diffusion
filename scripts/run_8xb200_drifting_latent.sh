#!/usr/bin/env bash
set -euo pipefail

# 8x B200 drifting training in latent space with latent-MAE feature loss.
#
# Target paper-style effective batch geometry:
#   N_c=64 classes / optimizer step
#   N_neg=64 generated samples / class
#   => B = N_c * N_neg = 4096
#
# Realization on 8 GPUs (default):
#   local: N_CLASSES_STEP=2, N_GEN_PER_CLASS=64  => 128 / GPU
#   grad_accum=4                                 => 64 classes / step globally
#   effective global batch = 128 * 8 * 4 = 4096
#
# Steps for 100 epochs on ImageNet-1k train:
#   ceil(1281167 / 4096) * 100 = 313 * 100 = 31300
#
# Usage:
#   DATA_ROOT=/path/to/imagenet/train \
#   LATENT_FEATURE_CKPT=/path/to/latent_mae/ckpt_final.pt \
#   bash scripts/run_8xb200_drifting_latent.sh

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
DATA_ROOT=${DATA_ROOT:-}
LATENT_FEATURE_CKPT=${LATENT_FEATURE_CKPT:-}

IMAGE_SIZE=${IMAGE_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-8}
SEED=${SEED:-0}
STEPS=${STEPS:-31300}

N_CLASSES_STEP=${N_CLASSES_STEP:-2}
N_GEN_PER_CLASS=${N_GEN_PER_CLASS:-64}
GRAD_ACCUM=${GRAD_ACCUM:-4}

OPTIMIZER=${OPTIMIZER:-muon}         # adamw | muon
LR=${LR:-2e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
GRAD_CLIP=${GRAD_CLIP:-2.0}
MUON_LR=${MUON_LR:-0.02}
MUON_MOMENTUM=${MUON_MOMENTUM:-0.95}
MUON_WEIGHT_DECAY=${MUON_WEIGHT_DECAY:-0.01}
MUON_ADAM_BETA1=${MUON_ADAM_BETA1:-0.9}
MUON_ADAM_BETA2=${MUON_ADAM_BETA2:-0.95}
MUON_ADAM_EPS=${MUON_ADAM_EPS:-1e-10}

PATCH_SIZE=${PATCH_SIZE:-2}
K_MIN=${K_MIN:-32}
LEVELS=${LEVELS:-4}
K_SCHEDULE=${K_SCHEDULE:-geom}
INTERP_MODE=${INTERP_MODE:-laplacian}
INTERP_POWER=${INTERP_POWER:-2.0}
INTERP_LAP_SIGMA_FEATURE=${INTERP_LAP_SIGMA_FEATURE:-0.20}
INTERP_LAP_SIGMA_SPACE=${INTERP_LAP_SIGMA_SPACE:-1.0}
INTERP_LAP_LAMBDA_REG=${INTERP_LAP_LAMBDA_REG:-1e-4}
INTERP_LAP_DIAG_NEIGHBORS=${INTERP_LAP_DIAG_NEIGHBORS:-0}
INTERP_LAP_MIN_WEIGHT=${INTERP_LAP_MIN_WEIGHT:-1e-6}
KEEP_ANCHORS_FIXED=${KEEP_ANCHORS_FIXED:-0}

D_MODEL=${D_MODEL:-512}
D_FF=${D_FF:-2048}
N_LAYERS=${N_LAYERS:-8}
N_HEADS=${N_HEADS:-8}
DROPOUT=${DROPOUT:-0.0}

# Feature extraction in latent space (no pixel decode path).
FEATURE_SPACE=${FEATURE_SPACE:-latent_mae}
LATENT_FEATURE_BASE_WIDTH=${LATENT_FEATURE_BASE_WIDTH:-256}
FEATURE_INCLUDE_GLOBAL=${FEATURE_INCLUDE_GLOBAL:-1}
FEATURE_INCLUDE_INPUT_SQ=${FEATURE_INCLUDE_INPUT_SQ:-1}
FEATURE_INCLUDE_LOCAL=${FEATURE_INCLUDE_LOCAL:-0}
FEATURE_INCLUDE_PATCH_STATS=${FEATURE_INCLUDE_PATCH_STATS:-1}
FEATURE_MAX_LOCATIONS=${FEATURE_MAX_LOCATIONS:-32}

DRIFT_N_POS=${DRIFT_N_POS:-64}
DRIFT_N_NEG=${DRIFT_N_NEG:-64}
DRIFT_N_UNC=${DRIFT_N_UNC:-16}
DRIFT_TAUS=${DRIFT_TAUS:-0.02,0.05,0.2}
DRIFT_NORM_MODE=${DRIFT_NORM_MODE:-both}
SHARE_SPATIAL_NORM=${SHARE_SPATIAL_NORM:-1}
CFG_ALPHA_VALUES=${CFG_ALPHA_VALUES:-1.0,2.0,4.0}
CFG_ALPHA_PROBS=${CFG_ALPHA_PROBS:-}
VANILLA_LOSS_WEIGHT=${VANILLA_LOSS_WEIGHT:-1.0}
QUEUE_POS=${QUEUE_POS:-128}
QUEUE_NEG=${QUEUE_NEG:-128}
QUEUE_UNC=${QUEUE_UNC:-1000}

VAE_MODEL=${VAE_MODEL:-stabilityai/sd-vae-ft-mse}
VAE_SCALE=${VAE_SCALE:-0.18215}
VAE_USE_MEAN=${VAE_USE_MEAN:-1}
AMP_BF16=${AMP_BF16:-1}

LOG_EVERY=${LOG_EVERY:-20}
SAVE_EVERY=${SAVE_EVERY:-1000}
CKPT_DIR=${CKPT_DIR:-checkpoints/drifting_latent_8xb200}
LOG_DIR=${LOG_DIR:-runs/drifting_latent_8xb200}
RESUME=${RESUME:-}

if [[ -z "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT is required (ImageFolder root for ImageNet train split)." >&2
  exit 1
fi
if [[ -z "${LATENT_FEATURE_CKPT}" ]]; then
  echo "LATENT_FEATURE_CKPT is required (latent-MAE checkpoint path)." >&2
  exit 1
fi

LOCAL_BATCH=$((N_CLASSES_STEP * N_GEN_PER_CLASS))
EFF_BATCH=$((LOCAL_BATCH * NPROC_PER_NODE * GRAD_ACCUM))
echo "Launching latent drifting on ${NPROC_PER_NODE} GPUs"
echo "  data_root=${DATA_ROOT}"
echo "  local_batch=${LOCAL_BATCH} grad_accum=${GRAD_ACCUM} effective_global_batch=${EFF_BATCH}"
echo "  N_c global / step=$((N_CLASSES_STEP * NPROC_PER_NODE * GRAD_ACCUM)) N_neg=${N_GEN_PER_CLASS}"
echo "  optimizer=${OPTIMIZER} lr=${LR} wd=${WEIGHT_DECAY} muon_lr=${MUON_LR}"
echo "  latent_feature_ckpt=${LATENT_FEATURE_CKPT}"

CMD=(
  torchrun
  --nproc_per_node "${NPROC_PER_NODE}"
  -m src.train.train_drifting_image_keypatch
  --dataset imagefolder
  --data_root "${DATA_ROOT}"
  --image_size "${IMAGE_SIZE}"
  --num_classes 1000
  --input_space latent
  --feature_space "${FEATURE_SPACE}"
  --latent_feature_ckpt "${LATENT_FEATURE_CKPT}"
  --latent_feature_base_width "${LATENT_FEATURE_BASE_WIDTH}"
  --vae_model "${VAE_MODEL}"
  --vae_scale "${VAE_SCALE}"
  --vae_use_mean "${VAE_USE_MEAN}"
  --steps "${STEPS}"
  --seed "${SEED}"
  --num_workers "${NUM_WORKERS}"
  --optimizer "${OPTIMIZER}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --grad_clip "${GRAD_CLIP}"
  --grad_accum "${GRAD_ACCUM}"
  --muon_lr "${MUON_LR}"
  --muon_momentum "${MUON_MOMENTUM}"
  --muon_weight_decay "${MUON_WEIGHT_DECAY}"
  --muon_adam_beta1 "${MUON_ADAM_BETA1}"
  --muon_adam_beta2 "${MUON_ADAM_BETA2}"
  --muon_adam_eps "${MUON_ADAM_EPS}"
  --n_classes_step "${N_CLASSES_STEP}"
  --n_gen_per_class "${N_GEN_PER_CLASS}"
  --patch_size "${PATCH_SIZE}"
  --K_min "${K_MIN}"
  --levels "${LEVELS}"
  --k_schedule "${K_SCHEDULE}"
  --interp_power "${INTERP_POWER}"
  --interp_mode "${INTERP_MODE}"
  --interp_lap_sigma_feature "${INTERP_LAP_SIGMA_FEATURE}"
  --interp_lap_sigma_space "${INTERP_LAP_SIGMA_SPACE}"
  --interp_lap_lambda_reg "${INTERP_LAP_LAMBDA_REG}"
  --interp_lap_diag_neighbors "${INTERP_LAP_DIAG_NEIGHBORS}"
  --interp_lap_min_weight "${INTERP_LAP_MIN_WEIGHT}"
  --keep_anchors_fixed "${KEEP_ANCHORS_FIXED}"
  --d_model "${D_MODEL}"
  --d_ff "${D_FF}"
  --n_layers "${N_LAYERS}"
  --n_heads "${N_HEADS}"
  --dropout "${DROPOUT}"
  --feature_include_global "${FEATURE_INCLUDE_GLOBAL}"
  --feature_include_input_sq "${FEATURE_INCLUDE_INPUT_SQ}"
  --feature_include_local "${FEATURE_INCLUDE_LOCAL}"
  --feature_include_patch_stats "${FEATURE_INCLUDE_PATCH_STATS}"
  --feature_max_locations "${FEATURE_MAX_LOCATIONS}"
  --drift_n_pos "${DRIFT_N_POS}"
  --drift_n_neg "${DRIFT_N_NEG}"
  --drift_n_unc "${DRIFT_N_UNC}"
  --drift_taus "${DRIFT_TAUS}"
  --drift_norm_mode "${DRIFT_NORM_MODE}"
  --share_spatial_norm "${SHARE_SPATIAL_NORM}"
  --cfg_alpha_values "${CFG_ALPHA_VALUES}"
  --vanilla_loss_weight "${VANILLA_LOSS_WEIGHT}"
  --queue_pos "${QUEUE_POS}"
  --queue_neg "${QUEUE_NEG}"
  --queue_unc "${QUEUE_UNC}"
  --amp_bf16 "${AMP_BF16}"
  --log_every "${LOG_EVERY}"
  --save_every "${SAVE_EVERY}"
  --ckpt_dir "${CKPT_DIR}"
  --log_dir "${LOG_DIR}"
)

if [[ -n "${CFG_ALPHA_PROBS}" ]]; then
  CMD+=(--cfg_alpha_probs "${CFG_ALPHA_PROBS}")
fi
if [[ -n "${RESUME}" ]]; then
  CMD+=(--resume "${RESUME}")
fi

PYTHONPATH=. "${CMD[@]}"

