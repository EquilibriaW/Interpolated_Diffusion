#!/usr/bin/env bash
set -euo pipefail

# Resample stage-2 with selector and save fresh samples.npz + PNGs.
# Usage:
#   OUT_ROOT=... N_SAMPLES=50 BATCH=50 bash scripts/resample_unified_dp.sh

SAMPLE_CASES=${SAMPLE_CASES:-interp_levels_unified_T128_dp}
OUT_ROOT=${OUT_ROOT:-runs/ablate_stage2_unified_dp_resample_$(date +%Y%m%d_%H%M%S)}

EVAL_DATA=${EVAL_DATA:-outputs/d4rl_prepared_unified_T128_eval_dp/dataset.npz}
CKPT_KEYPOINTS=${CKPT_KEYPOINTS:-checkpoints/keypoints_unified_T128_dp/ckpt_final.pt}
CKPT_ROOT=${CKPT_ROOT:-checkpoints}
DPHI_CKPT=${DPHI_CKPT:-checkpoints/segment_cost_unified_T128_dp/ckpt_final.pt}
SELECTOR_CKPT=${SELECTOR_CKPT:-checkpoints/keypoint_selector_unified_T128_dp/ckpt_final.pt}

KP_INDEX_MODE=${KP_INDEX_MODE:-selector}
COND_START_GOAL=${COND_START_GOAL:-1}
CLAMP_ENDPOINTS=${CLAMP_ENDPOINTS:-1}
USE_SDF=${USE_SDF:-1}
POS_CLIP=${POS_CLIP:-1}
POS_CLIP_MIN=${POS_CLIP_MIN:-0.0}
POS_CLIP_MAX=${POS_CLIP_MAX:-1.0}
STAGE1_CACHE_MODE=${STAGE1_CACHE_MODE:-none}

# Optional: N_SAMPLES, BATCH, DDIM_STEPS, DDIM_SCHEDULE, LEVELS, K_SCHEDULE, K_GEOM_GAMMA

EVAL_DATA="$EVAL_DATA" \
CKPT_KEYPOINTS="$CKPT_KEYPOINTS" \
CKPT_ROOT="$CKPT_ROOT" \
OUT_ROOT="$OUT_ROOT" \
DPHI_CKPT="$DPHI_CKPT" \
SELECTOR_CKPT="$SELECTOR_CKPT" \
KP_INDEX_MODE="$KP_INDEX_MODE" \
COND_START_GOAL="$COND_START_GOAL" \
CLAMP_ENDPOINTS="$CLAMP_ENDPOINTS" \
USE_SDF="$USE_SDF" \
POS_CLIP="$POS_CLIP" \
POS_CLIP_MIN="$POS_CLIP_MIN" \
POS_CLIP_MAX="$POS_CLIP_MAX" \
STAGE1_CACHE_MODE="$STAGE1_CACHE_MODE" \
SAMPLE_CASES="$SAMPLE_CASES" \
N_SAMPLES="${N_SAMPLES:-50}" \
BATCH="${BATCH:-50}" \
DDIM_STEPS="${DDIM_STEPS:-100}" \
DDIM_SCHEDULE="${DDIM_SCHEDULE:-linear}" \
LEVELS="${LEVELS:-8}" \
K_SCHEDULE="${K_SCHEDULE:-geom}" \
K_GEOM_GAMMA="${K_GEOM_GAMMA:-}" \
bash runs/d4rl_ablate_stage2_sample_only.sh

NPZ_PATH="${OUT_ROOT}/${SAMPLE_CASES}_sample/samples.npz"
if [[ -f "${NPZ_PATH}" ]]; then
  PYTHONPATH=. .venv/bin/python - <<'PY'
import os
import numpy as np
import subprocess

npz_path = os.environ["NPZ_PATH"]
out_dir = os.path.dirname(npz_path)
data = np.load(npz_path)
n = data["interp"].shape[0]
indices = [0, 1, 2, 11, 29]
indices = [i for i in indices if i < n]
for idx in indices:
    out = os.path.join(out_dir, f"rerender_{idx:04d}.png")
    subprocess.check_call([
        ".venv/bin/python",
        "scripts/rerender_sample_from_npz.py",
        "--npz", npz_path,
        "--index", str(idx),
        "--out", out,
        "--show_start_goal", "1",
    ])
print(f"Rerendered indices: {indices}")
PY
else
  echo "[warn] samples.npz not found at ${NPZ_PATH}"
fi

echo "Done. Outputs in: ${OUT_ROOT}/${SAMPLE_CASES}_sample"
