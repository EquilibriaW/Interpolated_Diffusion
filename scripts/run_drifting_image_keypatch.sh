#!/usr/bin/env bash
set -euo pipefail

# Example:
#   DATA_ROOT=/path/to/imagenet/train \
#   STEPS=30000 \
#   bash scripts/run_drifting_image_keypatch.sh

DATASET=${DATASET:-imagefolder}            # imagefolder | fake
DATA_ROOT=${DATA_ROOT:-}
IMAGE_SIZE=${IMAGE_SIZE:-64}
NUM_CLASSES=${NUM_CLASSES:-1000}
INPUT_SPACE=${INPUT_SPACE:-pixel}         # pixel | latent
FEATURE_SPACE=${FEATURE_SPACE:-resnet}    # resnet | latent_mae
LATENT_FEATURE_CKPT=${LATENT_FEATURE_CKPT:-}
LATENT_FEATURE_BASE_WIDTH=${LATENT_FEATURE_BASE_WIDTH:-256}
VAE_MODEL=${VAE_MODEL:-stabilityai/sd-vae-ft-mse}
VAE_SCALE=${VAE_SCALE:-0.18215}
VAE_USE_MEAN=${VAE_USE_MEAN:-1}

STEPS=${STEPS:-30000}
SEED=${SEED:-0}
NUM_WORKERS=${NUM_WORKERS:-8}
LR=${LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
GRAD_CLIP=${GRAD_CLIP:-1.0}
GRAD_ACCUM=${GRAD_ACCUM:-1}
OPTIMIZER=${OPTIMIZER:-adamw}           # adamw | muon
MUON_LR=${MUON_LR:-0.02}
MUON_MOMENTUM=${MUON_MOMENTUM:-0.95}
MUON_WEIGHT_DECAY=${MUON_WEIGHT_DECAY:-0.01}
MUON_ADAM_BETA1=${MUON_ADAM_BETA1:-0.9}
MUON_ADAM_BETA2=${MUON_ADAM_BETA2:-0.95}
MUON_ADAM_EPS=${MUON_ADAM_EPS:-1e-10}

N_CLASSES_STEP=${N_CLASSES_STEP:-8}
N_GEN_PER_CLASS=${N_GEN_PER_CLASS:-4}     # batch = N_CLASSES_STEP * N_GEN_PER_CLASS

PATCH_SIZE=${PATCH_SIZE:-4}
K_MIN=${K_MIN:-32}
LEVELS=${LEVELS:-4}
K_SCHEDULE=${K_SCHEDULE:-geom}
K_GEOM_GAMMA=${K_GEOM_GAMMA:-}
INTERP_POWER=${INTERP_POWER:-2.0}
INTERP_MODE=${INTERP_MODE:-laplacian}      # idw | laplacian
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

FEATURE_ARCH=${FEATURE_ARCH:-resnet18}
FEATURE_PRETRAINED=${FEATURE_PRETRAINED:-1}
FEATURE_INCLUDE_GLOBAL=${FEATURE_INCLUDE_GLOBAL:-1}
FEATURE_INCLUDE_INPUT_SQ=${FEATURE_INCLUDE_INPUT_SQ:-1}
FEATURE_INCLUDE_LOCAL=${FEATURE_INCLUDE_LOCAL:-0}
FEATURE_INCLUDE_PATCH_STATS=${FEATURE_INCLUDE_PATCH_STATS:-1}
FEATURE_MAX_LOCATIONS=${FEATURE_MAX_LOCATIONS:-16}

DRIFT_N_POS=${DRIFT_N_POS:-64}
DRIFT_N_NEG=${DRIFT_N_NEG:-64}
DRIFT_N_UNC=${DRIFT_N_UNC:-32}
DRIFT_TAUS=${DRIFT_TAUS:-0.02,0.05,0.2}
DRIFT_NORM_MODE=${DRIFT_NORM_MODE:-both}
SHARE_SPATIAL_NORM=${SHARE_SPATIAL_NORM:-1}
CFG_ALPHA_VALUES=${CFG_ALPHA_VALUES:-1.0,2.0,4.0}
CFG_ALPHA_PROBS=${CFG_ALPHA_PROBS:-}
VANILLA_LOSS_WEIGHT=${VANILLA_LOSS_WEIGHT:-1.0}
QUEUE_POS=${QUEUE_POS:-256}
QUEUE_NEG=${QUEUE_NEG:-256}
QUEUE_UNC=${QUEUE_UNC:-1000}
AMP_BF16=${AMP_BF16:-1}
COMPILE_DRIFT=${COMPILE_DRIFT:-0}
COMPILE_MODE=${COMPILE_MODE:-max-autotune-no-cudagraphs}

LOG_EVERY=${LOG_EVERY:-20}
SAVE_EVERY=${SAVE_EVERY:-1000}
CKPT_DIR=${CKPT_DIR:-checkpoints/drifting_image_keypatch}
LOG_DIR=${LOG_DIR:-runs/drifting_image_keypatch}
RESUME=${RESUME:-}

PY_ARGS=(
  --dataset "${DATASET}"
  --image_size "${IMAGE_SIZE}"
  --num_classes "${NUM_CLASSES}"
  --input_space "${INPUT_SPACE}"
  --feature_space "${FEATURE_SPACE}"
  --latent_feature_base_width "${LATENT_FEATURE_BASE_WIDTH}"
  --vae_model "${VAE_MODEL}"
  --vae_scale "${VAE_SCALE}"
  --vae_use_mean "${VAE_USE_MEAN}"
  --steps "${STEPS}"
  --seed "${SEED}"
  --num_workers "${NUM_WORKERS}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --grad_clip "${GRAD_CLIP}"
  --grad_accum "${GRAD_ACCUM}"
  --optimizer "${OPTIMIZER}"
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
  --feature_arch "${FEATURE_ARCH}"
  --feature_pretrained "${FEATURE_PRETRAINED}"
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
  --compile_drift "${COMPILE_DRIFT}"
  --compile_mode "${COMPILE_MODE}"
  --log_every "${LOG_EVERY}"
  --save_every "${SAVE_EVERY}"
  --ckpt_dir "${CKPT_DIR}"
  --log_dir "${LOG_DIR}"
)

if [[ -n "${DATA_ROOT}" ]]; then
  PY_ARGS+=(--data_root "${DATA_ROOT}")
fi
if [[ -n "${K_GEOM_GAMMA}" ]]; then
  PY_ARGS+=(--k_geom_gamma "${K_GEOM_GAMMA}")
fi
if [[ -n "${RESUME}" ]]; then
  PY_ARGS+=(--resume "${RESUME}")
fi
if [[ -n "${CFG_ALPHA_PROBS}" ]]; then
  PY_ARGS+=(--cfg_alpha_probs "${CFG_ALPHA_PROBS}")
fi
if [[ -n "${LATENT_FEATURE_CKPT}" ]]; then
  PY_ARGS+=(--latent_feature_ckpt "${LATENT_FEATURE_CKPT}")
fi

echo "Running drifting image keypatch training"
echo "  dataset=${DATASET} data_root=${DATA_ROOT:-<none>}"
echo "  local_batch=$((N_CLASSES_STEP * N_GEN_PER_CLASS)) grad_accum=${GRAD_ACCUM} steps=${STEPS}"
echo "  input_space=${INPUT_SPACE} feature_space=${FEATURE_SPACE}"
echo "  latent_feature_ckpt=${LATENT_FEATURE_CKPT:-<none>} latent_feature_base_width=${LATENT_FEATURE_BASE_WIDTH}"
echo "  optimizer=${OPTIMIZER} lr=${LR} wd=${WEIGHT_DECAY}"
echo "  muon_lr=${MUON_LR} muon_momentum=${MUON_MOMENTUM} muon_wd=${MUON_WEIGHT_DECAY}"
echo "  feature_arch=${FEATURE_ARCH} patch_stats=${FEATURE_INCLUDE_PATCH_STATS}"
echo "  interp_mode=${INTERP_MODE} keep_anchors_fixed=${KEEP_ANCHORS_FIXED}"
echo "  drift_n_pos=${DRIFT_N_POS} drift_n_neg=${DRIFT_N_NEG} drift_n_unc=${DRIFT_N_UNC} taus=${DRIFT_TAUS}"
echo "  cfg_alpha_values=${CFG_ALPHA_VALUES} cfg_alpha_probs=${CFG_ALPHA_PROBS:-<uniform>} vanilla_loss_weight=${VANILLA_LOSS_WEIGHT}"
echo "  share_spatial_norm=${SHARE_SPATIAL_NORM} queue_unc=${QUEUE_UNC}"
echo "  amp_bf16=${AMP_BF16} compile_drift=${COMPILE_DRIFT} compile_mode=${COMPILE_MODE}"

PYTHONPATH=. python -m src.train.train_drifting_image_keypatch "${PY_ARGS[@]}"
