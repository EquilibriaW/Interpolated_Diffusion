#!/usr/bin/env bash
set -euo pipefail

# 8x B200 latent-MAE pretrain on ImageNet (SD-VAE latents).
#
# Effective global batch:
#   LOCAL_BATCH * NPROC * GRAD_ACCUM
# Default below: 64 * 8 * 8 = 4096
#
# Steps for 192 epochs on ImageNet-1k train:
#   ceil(1281167 / 4096) * 192 = 313 * 192 = 60096
#
# Usage:
#   DATA_ROOT=/path/to/imagenet/train bash scripts/run_8xb200_latent_mae.sh

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
DATA_ROOT=${DATA_ROOT:-}

IMAGE_SIZE=${IMAGE_SIZE:-256}
LOCAL_BATCH=${LOCAL_BATCH:-64}
GRAD_ACCUM=${GRAD_ACCUM:-8}
STEPS=${STEPS:-60096}
NUM_WORKERS=${NUM_WORKERS:-8}
SEED=${SEED:-0}

OPTIMIZER=${OPTIMIZER:-adamw}      # adamw | muon
LR=${LR:-4e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
GRAD_CLIP=${GRAD_CLIP:-1.0}
MUON_LR=${MUON_LR:-0.02}
MUON_MOMENTUM=${MUON_MOMENTUM:-0.95}
MUON_WEIGHT_DECAY=${MUON_WEIGHT_DECAY:-0.01}
MUON_ADAM_BETA1=${MUON_ADAM_BETA1:-0.9}
MUON_ADAM_BETA2=${MUON_ADAM_BETA2:-0.95}
MUON_ADAM_EPS=${MUON_ADAM_EPS:-1e-10}

BASE_WIDTH=${BASE_WIDTH:-256}
MASK_RATIO=${MASK_RATIO:-0.5}
MASK_PATCH=${MASK_PATCH:-2}

VAE_MODEL=${VAE_MODEL:-stabilityai/sd-vae-ft-mse}
VAE_SCALE=${VAE_SCALE:-0.18215}
VAE_USE_MEAN=${VAE_USE_MEAN:-1}
AMP_BF16=${AMP_BF16:-1}

LOG_EVERY=${LOG_EVERY:-20}
SAVE_EVERY=${SAVE_EVERY:-1000}
CKPT_DIR=${CKPT_DIR:-checkpoints/latent_mae_imagenet_8xb200}
LOG_DIR=${LOG_DIR:-runs/latent_mae_imagenet_8xb200}
RESUME=${RESUME:-}

if [[ -z "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT is required (ImageFolder root for ImageNet train split)." >&2
  exit 1
fi

EFF_BATCH=$((LOCAL_BATCH * NPROC_PER_NODE * GRAD_ACCUM))
echo "Launching latent-MAE pretrain on ${NPROC_PER_NODE} GPUs"
echo "  data_root=${DATA_ROOT}"
echo "  local_batch=${LOCAL_BATCH} grad_accum=${GRAD_ACCUM} effective_global_batch=${EFF_BATCH}"
echo "  steps=${STEPS} optimizer=${OPTIMIZER} lr=${LR} wd=${WEIGHT_DECAY}"
echo "  base_width=${BASE_WIDTH} mask_ratio=${MASK_RATIO} mask_patch=${MASK_PATCH}"

CMD=(
  torchrun
  --nproc_per_node "${NPROC_PER_NODE}"
  -m src.train.train_latent_mae_imagenet
  --dataset imagefolder
  --data_root "${DATA_ROOT}"
  --image_size "${IMAGE_SIZE}"
  --batch_size "${LOCAL_BATCH}"
  --grad_accum "${GRAD_ACCUM}"
  --steps "${STEPS}"
  --num_workers "${NUM_WORKERS}"
  --seed "${SEED}"
  --optimizer "${OPTIMIZER}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --grad_clip "${GRAD_CLIP}"
  --muon_lr "${MUON_LR}"
  --muon_momentum "${MUON_MOMENTUM}"
  --muon_weight_decay "${MUON_WEIGHT_DECAY}"
  --muon_adam_beta1 "${MUON_ADAM_BETA1}"
  --muon_adam_beta2 "${MUON_ADAM_BETA2}"
  --muon_adam_eps "${MUON_ADAM_EPS}"
  --base_width "${BASE_WIDTH}"
  --mask_ratio "${MASK_RATIO}"
  --mask_patch "${MASK_PATCH}"
  --vae_model "${VAE_MODEL}"
  --vae_scale "${VAE_SCALE}"
  --vae_use_mean "${VAE_USE_MEAN}"
  --amp_bf16 "${AMP_BF16}"
  --log_every "${LOG_EVERY}"
  --save_every "${SAVE_EVERY}"
  --ckpt_dir "${CKPT_DIR}"
  --log_dir "${LOG_DIR}"
)

if [[ -n "${RESUME}" ]]; then
  CMD+=(--resume "${RESUME}")
fi

PYTHONPATH=. "${CMD[@]}"

