#!/usr/bin/env bash
set -euo pipefail

# Move nested "videos/video" folder to "didemo/video" and optionally remove "videos".
#
# Usage:
#   bash scripts/datasets/didemo/fix_video_folder_layout.sh
# Optional:
#   DIDEMO_DIR=data/didemo REMOVE_VIDEOS_DIR=1 bash ...

DIDEMO_DIR="${DIDEMO_DIR:-data/didemo}"
SRC_DIR="${DIDEMO_DIR}/videos/video"
DST_DIR="${DIDEMO_DIR}/video"
REMOVE_VIDEOS_DIR="${REMOVE_VIDEOS_DIR:-1}"

if [ ! -d "${SRC_DIR}" ]; then
  echo "Expected nested folder not found: ${SRC_DIR}"
  exit 1
fi

mkdir -p "${DST_DIR}"

if [ "${DST_DIR}" != "${SRC_DIR}" ]; then
  # If DST is empty, a direct move is fastest.
  if [ -z "$(ls -A "${DST_DIR}" 2>/dev/null)" ]; then
    mv "${SRC_DIR}" "${DST_DIR}"
  else
    find "${SRC_DIR}" -type f -print0 | xargs -0 -I{} mv -n {} "${DST_DIR}/"
    find "${SRC_DIR}" -type d -empty -delete
  fi
fi

if [ "${REMOVE_VIDEOS_DIR}" = "1" ]; then
  rm -rf "${DIDEMO_DIR}/videos"
else
  echo "Leaving ${DIDEMO_DIR}/videos in place (REMOVE_VIDEOS_DIR=0)."
fi

echo "Done. Videos now under: ${DST_DIR}"
