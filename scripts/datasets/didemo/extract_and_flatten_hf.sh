#!/usr/bin/env bash
set -euo pipefail

ASSEMBLED_DIR="${ASSEMBLED_DIR:-data/didemo/hf_cache/assembled}"
VIDEO_DIR="${VIDEO_DIR:-data/didemo/videos}"

mkdir -p "${VIDEO_DIR}"

for f in "${ASSEMBLED_DIR}"/*.tar; do
  echo "Extracting $f"
  tar -xf "$f" -C "${VIDEO_DIR}"
done

# Flatten any nested directories into VIDEO_DIR (keep first occurrence)
find "${VIDEO_DIR}" -type f \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mkv" \) -not -path "${VIDEO_DIR}/*" -print0 | \
  xargs -0 -I{} mv -n {} "${VIDEO_DIR}/"

# Remove empty folders
find "${VIDEO_DIR}" -type d -empty -delete

echo "Videos in root: $(ls "${VIDEO_DIR}" | wc -l)"
