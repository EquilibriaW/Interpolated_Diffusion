#!/usr/bin/env bash
set -euo pipefail

cd /workspace/Interpolated_Diffusion

echo "[runpod-entry] python=$(python3 --version)"
echo "[runpod-entry] torch=$(python3 -c 'import torch; print(torch.__version__, torch.version.cuda)')"
echo "[runpod-entry] nvidia-smi:"
nvidia-smi || true

if [[ -n "${LAUNCH_SCRIPT:-}" ]]; then
  echo "[runpod-entry] launching script: ${LAUNCH_SCRIPT}"
  exec bash "${LAUNCH_SCRIPT}"
fi

echo "[runpod-entry] no LAUNCH_SCRIPT specified; dropping into shell."
exec bash
