#!/usr/bin/env bash
set -euo pipefail

LOG_PATH=${LOG_PATH:-notes/PROJECT_LOG.md}
EXP_NAME=${EXP_NAME:-experiment}
EXP_COMMAND=${EXP_COMMAND:-}
TRAIN_DATA=${TRAIN_DATA:-}
EVAL_DATA=${EVAL_DATA:-}
CKPT_KP=${CKPT_KP:-}
CKPT_S2=${CKPT_S2:-}
DPHI_CKPT=${DPHI_CKPT:-}
LOG_KP=${LOG_KP:-}
LOG_S2=${LOG_S2:-}
DPHI_LOG=${DPHI_LOG:-}
RESULTS_PATH=${RESULTS_PATH:-}
RESULTS_DIR=${RESULTS_DIR:-}
KEY_SETTINGS=${KEY_SETTINGS:-}
NOTES=${NOTES:-}

DATE_UTC=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
GIT_DIRTY=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
if [[ "${GIT_DIRTY}" != "0" ]]; then
  GIT_STATE="dirty"
else
  GIT_STATE="clean"
fi

RESULTS_SUMMARY=""
if [[ -n "${RESULTS_PATH}" || -n "${RESULTS_DIR}" ]]; then
  RESULTS_SUMMARY=$(python - <<'PY'
import csv, glob, os, sys

results_path = os.environ.get("RESULTS_PATH","").strip()
results_dir = os.environ.get("RESULTS_DIR","").strip()

paths = []
if results_path:
    paths.append(results_path)
if results_dir:
    paths.extend(glob.glob(os.path.join(results_dir, "**", "metrics.csv"), recursive=True))
paths = [p for p in paths if os.path.isfile(p)]
if not paths:
    print("")
    sys.exit(0)

order = [
    "collision_interp",
    "collision_refined",
    "goal_dist_interp",
    "goal_dist_refined",
    "success_interp",
    "success_refined",
]

def summarize(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None
    cols = {}
    for k in reader.fieldnames:
        if k == "sample_id":
            continue
        cols[k] = []
    for r in rows:
        for k, v in r.items():
            if k == "sample_id":
                continue
            try:
                cols[k].append(float(v))
            except Exception:
                pass
    summary = {}
    for k, vals in cols.items():
        if vals:
            summary[k] = sum(vals) / len(vals)
    return summary

lines = []
for path in sorted(set(paths)):
    summary = summarize(path)
    if not summary:
        continue
    parts = []
    for k in order:
        if k in summary:
            parts.append(f"{k}={summary[k]:.4f}")
    if not parts:
        # fallback: show up to 4 arbitrary keys
        for k in sorted(summary.keys())[:4]:
            parts.append(f"{k}={summary[k]:.4f}")
    rel = os.path.relpath(path)
    lines.append(f"{rel}: " + ", ".join(parts))
if lines:
    print("\\n".join(lines))
PY
)
fi

{
  echo ""
  echo "## ${DATE_UTC} – ${EXP_NAME}"
  if [[ -n "${EXP_COMMAND}" ]]; then
    echo "Command: ${EXP_COMMAND}"
  fi
  echo "Git: ${GIT_HASH} (${GIT_STATE})"
  if [[ -n "${TRAIN_DATA}" ]]; then
    echo "Train data: ${TRAIN_DATA}"
  fi
  if [[ -n "${EVAL_DATA}" ]]; then
    echo "Eval data: ${EVAL_DATA}"
  fi
  if [[ -n "${CKPT_KP}" ]]; then
    echo "CKPT keypoints: ${CKPT_KP}"
  fi
  if [[ -n "${CKPT_S2}" ]]; then
    echo "CKPT stage2: ${CKPT_S2}"
  fi
  if [[ -n "${DPHI_CKPT}" ]]; then
    echo "Dphi ckpt: ${DPHI_CKPT}"
  fi
  if [[ -n "${LOG_KP}" ]]; then
    echo "Log keypoints: ${LOG_KP}"
  fi
  if [[ -n "${LOG_S2}" ]]; then
    echo "Log stage2: ${LOG_S2}"
  fi
  if [[ -n "${DPHI_LOG}" ]]; then
    echo "Log Dphi: ${DPHI_LOG}"
  fi
  if [[ -n "${KEY_SETTINGS}" ]]; then
    echo "Key settings: ${KEY_SETTINGS}"
  fi
  if [[ -n "${RESULTS_SUMMARY}" ]]; then
    echo "Results:"
    while IFS= read -r line; do
      [[ -n "${line}" ]] && echo "  - ${line}"
    done <<< "${RESULTS_SUMMARY}"
  fi
  if [[ -n "${NOTES}" ]]; then
    echo "Notes: ${NOTES}"
  fi
} >> "${LOG_PATH}"
