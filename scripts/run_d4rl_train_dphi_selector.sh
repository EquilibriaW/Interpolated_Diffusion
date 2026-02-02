#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_PY_MUJOCO_PATH=${MUJOCO_PY_MUJOCO_PATH:-/workspace/Interpolated_Diffusion/mujoco210}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/workspace/Interpolated_Diffusion/mujoco210/bin:${LD_LIBRARY_PATH:-}}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export D4RL_SUPPRESS_IMPORT_ERROR=${D4RL_SUPPRESS_IMPORT_ERROR:-1}
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}

PYTHON=${PYTHON:-.venv/bin/python}

T=${T:-128}
K=${K:-8}
LEVELS=${LEVELS:-8}
K_SCHEDULE=${K_SCHEDULE:-geom}
K_GEOM_GAMMA=${K_GEOM_GAMMA:-}
USE_SDF=${USE_SDF:-1}
COND_START_GOAL=${COND_START_GOAL:-1}

TRAIN_DATA=${TRAIN_DATA:-outputs/d4rl_prepared_unified_T${T}_train/dataset.npz}
EVAL_DATA=${EVAL_DATA:-outputs/d4rl_prepared_unified_T${T}_eval/dataset.npz}

DPHI_STEPS=${DPHI_STEPS:-10000}
DPHI_BATCH=${DPHI_BATCH:-64}
DPHI_CKPT=${DPHI_CKPT:-checkpoints/segment_cost_unified_T${T}_dp}
DPHI_LOG=${DPHI_LOG:-runs/segment_cost_unified_T${T}_dp}
DPHI_STATS_SUBSET=${DPHI_STATS_SUBSET:-512}
DPHI_MAZE_CHANNELS=${DPHI_MAZE_CHANNELS:-32,64,128,128}

DP_BATCH=${DP_BATCH:-64}
DP_DEVICE=${DP_DEVICE:-cuda}
DP_SEG_SAMPLES=${DP_SEG_SAMPLES:-16}

SELECTOR_DP_TRAIN_DATA=${SELECTOR_DP_TRAIN_DATA:-outputs/d4rl_prepared_unified_T${T}_train_dp_dphi/dataset.npz}
SELECTOR_DP_EVAL_DATA=${SELECTOR_DP_EVAL_DATA:-outputs/d4rl_prepared_unified_T${T}_eval_dp_dphi/dataset.npz}

SELECTOR_STEPS=${SELECTOR_STEPS:-10000}
SELECTOR_BATCH=${SELECTOR_BATCH:-128}
SELECTOR_CKPT=${SELECTOR_CKPT:-checkpoints/keypoint_selector_unified_T${T}_dp}
SELECTOR_LOG=${SELECTOR_LOG:-runs/keypoint_selector_unified_T${T}_dp}
SELECTOR_D_MODEL=${SELECTOR_D_MODEL:-256}
SELECTOR_N_HEADS=${SELECTOR_N_HEADS:-8}
SELECTOR_D_FF=${SELECTOR_D_FF:-512}
SELECTOR_N_LAYERS=${SELECTOR_N_LAYERS:-2}
SELECTOR_POS_DIM=${SELECTOR_POS_DIM:-64}
SELECTOR_DROPOUT=${SELECTOR_DROPOUT:-0.0}
SELECTOR_USE_SG_MAP=${SELECTOR_USE_SG_MAP:-1}
SELECTOR_USE_SG_TOKEN=${SELECTOR_USE_SG_TOKEN:-1}
SELECTOR_USE_GOAL_DIST_TOKEN=${SELECTOR_USE_GOAL_DIST_TOKEN:-1}
SELECTOR_USE_COND_BIAS=${SELECTOR_USE_COND_BIAS:-1}
SELECTOR_COND_BIAS_MODE=${SELECTOR_COND_BIAS_MODE:-encoder}
SELECTOR_USE_LEVEL=${SELECTOR_USE_LEVEL:-1}
SELECTOR_LEVEL_MODE=${SELECTOR_LEVEL_MODE:-k_norm}
SELECTOR_SG_MAP_SIGMA=${SELECTOR_SG_MAP_SIGMA:-1.5}
SELECTOR_KL_WEIGHT=${SELECTOR_KL_WEIGHT:-0.02}
SELECTOR_TAU_START=${SELECTOR_TAU_START:-1.0}
SELECTOR_TAU_END=${SELECTOR_TAU_END:-0.3}
SELECTOR_TAU_ANNEAL=${SELECTOR_TAU_ANNEAL:-cosine}
SELECTOR_TAU_FRAC=${SELECTOR_TAU_FRAC:-0.8}
SELECTOR_MAZE_CHANNELS=${SELECTOR_MAZE_CHANNELS:-${DPHI_MAZE_CHANNELS}}

echo "==> Train segment-cost predictor (D_phi)"
PYTHONPATH=. ${PYTHON} -m src.train.train_segment_cost \
  --prepared_path "${TRAIN_DATA}" \
  --T "${T}" --batch "${DPHI_BATCH}" --steps "${DPHI_STEPS}" \
  --use_sdf "${USE_SDF}" --cond_start_goal "${COND_START_GOAL}" \
  --segment_cost_samples "${DP_SEG_SAMPLES}" \
  --maze_channels "${DPHI_MAZE_CHANNELS}" \
  --schedule cosine --N_train 1000 \
  --snr_min 0.1 --snr_max 10.0 --snr_gamma 1.0 \
  --t_steps 16 \
  --stats_subset "${DPHI_STATS_SUBSET}" \
  --ckpt_dir "${DPHI_CKPT}" --log_dir "${DPHI_LOG}"

echo "==> Build selector DP labels from D_phi (cond-only)"
PYTHONPATH=. ${PYTHON} src/data/prepare_dp_keypoints.py \
  --in_npz "${TRAIN_DATA}" \
  --out_npz "${SELECTOR_DP_TRAIN_DATA}" \
  --K "${K}" --segment_cost_samples "${DP_SEG_SAMPLES}" \
  --store_kp_mask_levels 1 --levels "${LEVELS}" --k_schedule "${K_SCHEDULE}" ${K_GEOM_GAMMA:+--k_geom_gamma "${K_GEOM_GAMMA}"} \
  --cost_source dphi --dphi_ckpt "${DPHI_CKPT}/ckpt_final.pt" \
  --per_sample 1 \
  --batch "${DP_BATCH}" --device "${DP_DEVICE}"

PYTHONPATH=. ${PYTHON} src/data/prepare_dp_keypoints.py \
  --in_npz "${EVAL_DATA}" \
  --out_npz "${SELECTOR_DP_EVAL_DATA}" \
  --K "${K}" --segment_cost_samples "${DP_SEG_SAMPLES}" \
  --store_kp_mask_levels 1 --levels "${LEVELS}" --k_schedule "${K_SCHEDULE}" ${K_GEOM_GAMMA:+--k_geom_gamma "${K_GEOM_GAMMA}"} \
  --cost_source dphi --dphi_ckpt "${DPHI_CKPT}/ckpt_final.pt" \
  --per_sample 1 \
  --batch "${DP_BATCH}" --device "${DP_DEVICE}"

echo "==> Train keypoint selector (cond-only, DP(D_phi) labels)"
PYTHONPATH=. ${PYTHON} -m src.train.train_keypoint_selector \
  --prepared_path "${SELECTOR_DP_TRAIN_DATA}" \
  --T "${T}" --K "${K}" --batch "${SELECTOR_BATCH}" --steps "${SELECTOR_STEPS}" \
  --use_sdf "${USE_SDF}" --cond_start_goal "${COND_START_GOAL}" \
  --d_model "${SELECTOR_D_MODEL}" --n_heads "${SELECTOR_N_HEADS}" --d_ff "${SELECTOR_D_FF}" \
  --n_layers "${SELECTOR_N_LAYERS}" --pos_dim "${SELECTOR_POS_DIM}" \
  --dropout "${SELECTOR_DROPOUT}" --maze_channels "${SELECTOR_MAZE_CHANNELS}" \
  --use_sg_map "${SELECTOR_USE_SG_MAP}" --use_sg_token "${SELECTOR_USE_SG_TOKEN}" \
  --use_goal_dist_token "${SELECTOR_USE_GOAL_DIST_TOKEN}" \
  --use_cond_bias "${SELECTOR_USE_COND_BIAS}" \
  --cond_bias_mode "${SELECTOR_COND_BIAS_MODE}" \
  --use_level "${SELECTOR_USE_LEVEL}" --level_mode "${SELECTOR_LEVEL_MODE}" \
  --levels "${LEVELS}" --k_schedule "${K_SCHEDULE}" ${K_GEOM_GAMMA:+--k_geom_gamma "${K_GEOM_GAMMA}"} \
  --sg_map_sigma "${SELECTOR_SG_MAP_SIGMA}" \
  --sel_kl_weight "${SELECTOR_KL_WEIGHT}" \
  --sel_tau_start "${SELECTOR_TAU_START}" \
  --sel_tau_end "${SELECTOR_TAU_END}" \
  --sel_tau_anneal "${SELECTOR_TAU_ANNEAL}" \
  --sel_tau_frac "${SELECTOR_TAU_FRAC}" \
  --ckpt_dir "${SELECTOR_CKPT}" --log_dir "${SELECTOR_LOG}"

if [[ "${SELECTOR_DIAG:-1}" == "1" ]]; then
  echo "==> Selector diagnostics (overlap + distribution)"
  SELECTOR_CKPT="${SELECTOR_CKPT}" \
  SELECTOR_DP_EVAL_DATA="${SELECTOR_DP_EVAL_DATA}" \
  PYTHONPATH=. ${PYTHON} - <<'PY'
import numpy as np
import os
import torch
from src.data.dataset import PreparedTrajectoryDataset
from src.models.keypoint_selector import KeypointSelector, select_topk_indices
from src.corruptions.keyframes import _compute_k_schedule

ckpt = os.path.join(os.environ["SELECTOR_CKPT"], "ckpt_final.pt")
eval_path = os.environ["SELECTOR_DP_EVAL_DATA"]

payload = torch.load(ckpt, map_location="cpu")
meta = payload["meta"]
state = payload.get("model", payload)
use_cond_bias = bool(meta.get("use_cond_bias", False))
if not use_cond_bias:
    for key in state.keys():
        if key.startswith("cond_bias."):
            use_cond_bias = True
            break
cond_bias_mode = meta.get("cond_bias_mode", "memory")
if any(k.startswith("cond_enc.") for k in state.keys()):
    cond_bias_mode = "encoder"
use_level = bool(meta.get("use_level", False))
level_mode = str(meta.get("level_mode", "k_norm"))
model = KeypointSelector(
    T=meta["T"],
    d_model=meta["d_model"],
    n_heads=meta["n_heads"],
    d_ff=meta["d_ff"],
    n_layers=meta["n_layers"],
    pos_dim=meta["pos_dim"],
    dropout=meta["dropout"],
    use_sdf=meta["use_sdf"],
    use_start_goal=meta["cond_start_goal"],
    use_sg_map=meta.get("use_sg_map", True),
    use_sg_token=meta.get("use_sg_token", True),
    use_goal_dist_token=meta.get("use_goal_dist_token", False),
    use_cond_bias=use_cond_bias,
    cond_bias_mode=str(cond_bias_mode),
    use_level=use_level,
    level_mode=level_mode,
    sg_map_sigma=meta.get("sg_map_sigma", 1.5),
    maze_channels=tuple(int(x) for x in str(meta.get("maze_channels", "32,64")).split(",")),
).eval()
model.load_state_dict(state)

ds = PreparedTrajectoryDataset(eval_path, use_sdf=bool(meta["use_sdf"]))
B = min(512, len(ds))
idxs = np.random.choice(len(ds), size=B, replace=False)
cond = {}
for k in ds[0]["cond"]:
    cond[k] = torch.stack([ds[i]["cond"][k] for i in idxs], dim=0)
if "kp_mask_levels" in ds[0]["cond"]:
    true_levels = torch.stack([ds[i]["cond"]["kp_mask_levels"] for i in idxs], dim=0)
    levels = int(meta.get("levels", true_levels.shape[1] - 1))
    k_list = _compute_k_schedule(int(meta["T"]), int(meta["K"]), levels,
                                 schedule=str(meta.get("k_schedule", "geom")),
                                 geom_gamma=meta.get("k_geom_gamma", None))
    s_idx = torch.full((B,), levels, dtype=torch.long)
    true_mask = true_levels[torch.arange(B), s_idx]
    if use_level:
        if level_mode == "s_norm":
            level_val = s_idx.float() / float(max(1, levels))
        else:
            k_list_t = torch.tensor(k_list, dtype=torch.float32)
            level_val = k_list_t[s_idx] / float(max(1, meta["T"] - 1))
        cond = dict(cond)
        cond["level"] = level_val.unsqueeze(1)
    true = []
    for b in range(B):
        idx = torch.nonzero(true_mask[b], as_tuple=False).squeeze(-1).cpu().numpy()
        true.append(idx)
    true = np.stack(true, axis=0)
else:
    true = np.stack([ds[i]["cond"]["kp_idx"].numpy() for i in idxs], axis=0)

with torch.no_grad():
    logits = model({k: v for k, v in cond.items()})
    pred = select_topk_indices(logits, meta["K"]).cpu().numpy()

mae = np.abs(pred - true).mean()
overlap = np.mean([len(set(pred[i]) & set(true[i])) / len(true[i]) for i in range(B)])
counts = np.zeros(meta["T"], dtype=np.int64)
for row in pred:
    for t in row:
        counts[t] += 1
top = np.argsort(-counts[1:-1])[:10] + 1

print(f"selector mae={mae:.2f} overlap={overlap:.3f}")
print("top interior predicted idx:", top.tolist())
PY
fi

echo "==> Done"
