#!/usr/bin/env bash
set -euo pipefail

# Bootstrap script for a fresh B200 node.
# - Clones/pulls repo
# - Recreates .venv from notes/pip_freeze.txt (fallback to requirements.txt)
# - Optionally restores .codex state and checkpoints/runs from Google Drive via rclone

REPO_URL="${REPO_URL:-git@github.com:EquilibriaW/Interpolated_Diffusion.git}"
REPO_DIR="${REPO_DIR:-/workspace/Interpolated_Diffusion}"
BRANCH="${BRANCH:-main}"

RESTORE_CODEX="${RESTORE_CODEX:-0}"
CODEX_TARBALL="${CODEX_TARBALL:-}"      # e.g. codex_state_YYYYMMDD_HHMMSS.tgz
RESTORE_RUNS="${RESTORE_RUNS:-0}"
RUNS_TARBALL="${RUNS_TARBALL:-}"        # e.g. checkpoints_runs_YYYYMMDD_HHMMSS.tgz

RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive:Interpolated_diffusion}"
RCLONE_BIN="${RCLONE_BIN:-/workspace/bin/rclone}"

echo "==> Repo: ${REPO_URL} (${BRANCH})"
if [[ ! -d "${REPO_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${REPO_DIR}")"
  git clone "${REPO_URL}" "${REPO_DIR}"
fi
cd "${REPO_DIR}"
git fetch origin "${BRANCH}"
git checkout "${BRANCH}"
git pull --ff-only

echo "==> Python env"
if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
if [[ -f "notes/pip_freeze.txt" ]]; then
  echo "==> Installing from notes/pip_freeze.txt"
  pip install -r notes/pip_freeze.txt || true
else
  echo "==> Installing from requirements.txt"
  pip install -r requirements.txt
fi

if [[ "${RESTORE_CODEX}" == "1" || "${RESTORE_RUNS}" == "1" ]]; then
  echo "==> Ensuring rclone"
  if [[ ! -x "${RCLONE_BIN}" ]]; then
    mkdir -p /workspace/bin
    curl -L -o /workspace/bin/rclone.zip https://downloads.rclone.org/rclone-current-linux-amd64.zip
    python - <<'PY'
import zipfile, pathlib
zip_path = pathlib.Path("/workspace/bin/rclone.zip")
with zipfile.ZipFile(zip_path) as z:
    z.extractall("/workspace/bin")
PY
    cp /workspace/bin/rclone-*-linux-amd64/rclone /workspace/bin/rclone
    chmod +x /workspace/bin/rclone
  fi
  export PATH="/workspace/bin:${PATH}"
fi

if [[ "${RESTORE_CODEX}" == "1" ]]; then
  if [[ -z "${CODEX_TARBALL}" ]]; then
    echo "ERROR: RESTORE_CODEX=1 but CODEX_TARBALL is empty."
    exit 1
  fi
  echo "==> Restoring codex state: ${CODEX_TARBALL}"
  "${RCLONE_BIN}" copy "${RCLONE_REMOTE}/${CODEX_TARBALL}" "${REPO_DIR}/notes/" --progress
  tar -xzf "${REPO_DIR}/notes/${CODEX_TARBALL}" -C /
  ln -sfn /workspace/.codex /root/.codex
fi

if [[ "${RESTORE_RUNS}" == "1" ]]; then
  if [[ -z "${RUNS_TARBALL}" ]]; then
    echo "ERROR: RESTORE_RUNS=1 but RUNS_TARBALL is empty."
    exit 1
  fi
  echo "==> Restoring checkpoints/runs: ${RUNS_TARBALL}"
  "${RCLONE_BIN}" copy "${RCLONE_REMOTE}/${RUNS_TARBALL}" "${REPO_DIR}/notes/" --progress
  tar -xzf "${REPO_DIR}/notes/${RUNS_TARBALL}" -C /
fi

echo "==> Done."
