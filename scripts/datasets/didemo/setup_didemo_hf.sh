#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/didemo}"
VIDEO_DIR="${VIDEO_DIR:-data/didemo/videos}"
HF_CACHE_DIR="${HF_CACHE_DIR:-data/didemo/hf_cache}"
REPO_ID="${REPO_ID:-friedrichor/DiDeMo}"

python scripts/datasets/didemo/fetch_didemo_metadata.py --out_dir "${DATA_DIR}"

python scripts/datasets/didemo/download_videos_hf.py \
  --repo_id "${REPO_ID}" \
  --cache_dir "${HF_CACHE_DIR}" \
  --video_dir "${VIDEO_DIR}" \
  --flatten 1
