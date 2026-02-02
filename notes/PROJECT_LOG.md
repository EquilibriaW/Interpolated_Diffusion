# Project Log (Decisions, Implementation, Experiments)

This is a running log of user decisions, implementation status, and experiment results.
Process: update this file after each experiment run and after major design discussions.
Include: date, command or script used, dataset/checkpoint paths, key settings, and observed results.

## 2026-01-30

### Decisions / Requests
- Use GT trajectories as teacher for maze; use teacher diffusion only for video later.
- Keypoint selection: global DP using subset of training data (10%). (Superseded by per-sample DP for selector training.)
- Diffusion-per-step KL weighting: SNR weighting with `s_min=0.1`, `s_max=10`, `gamma=1`, `t_steps=16`.
- Segment cost approximation: `segment_cost_samples=16`.
- kp_feat layout: `[left_gap, right_gap, t_norm, left_diff, right_diff]` (dim=5).
- Phase-1 index policy mix: `dp:0.7, uniform:0.2, random:0.1`.
- Train a segment-difficulty predictor `D_phi` offline and use predicted difficulties as features during phase-1 training/sampling (no maze-specific hand features).
- Do **not** add auxiliary KL loss yet (hold for video).
- Logit-space disabled for Maze2D; rely on [0,1] normalization + optional position clipping during sampling.
- Train a **keypoint selector** that uses **cond only** (occ/start_goal/sdf) and **per-sample DP indices** as labels; use selector indices in training/sampling (mix with uniform/random).
- Selector + phase-1/2 conditioning should include start/goal (`COND_START_GOAL=1`) so selector can predict path-relevant indices.
- Asked about how `occ` and `sdf` are constructed; response: occ from D4RL maze_map via `_maze_map_to_occ` (handles 10/11/12 + transpose), optional flip_y, pad-only unified; sdf from `sdf_from_occupancy` (grid L1 distance, signed) computed from occ.
- **Level-conditioned selector requested**: selector should rank indices per noise/level; stage-2 should add most-informative points as noise decreases (nested masks built by selector ranking). For maze, treat “level” as **information budget (K_s)** rather than teacher diffusion noise. D_phi remains **noise-agnostic** in maze; per-level DP masks are computed from the same D_phi cost matrix for each K_s. For video, revisit with diffusion-noise-conditioned teacher.
- **Noise mismatch acknowledgement**: Without a diffusion teacher, maze-level conditioning cannot reflect diffusion noise; we use K_s (information budget) as proxy. For video, will need D_phi conditioned on teacher noise (or per-step KL) and align selector level with teacher noise schedule; also resolve mismatch between teacher-noise type and phase-2 corruption noise.

### Implementation Status
- **DP keypoint preprocessing**: Implemented (`src/data/prepare_dp_keypoints.py`) with subset + SNR weighting. Status: done.
- **Per-level DP masks for selector**: `prepare_dp_keypoints.py` can now store `kp_mask_levels` (per-sample) using DP for each K_s; `PreparedTrajectoryDataset` loads `kp_mask_levels`. Stage-2 uses selector ranking **per level** (no random fill) via `build_nested_masks_from_level_logits`. Status: done.
- **Stage-2 position clipping**: added `--pos_clip` (min/max) to stage-2 training to clamp corrupted inputs to [0,1], and sampling now applies `pos_clip` after each stage‑2 update to prevent off‑screen drift. Scripts updated to pass `POS_CLIP` to stage‑2 training.
- **Stage-2 mask diagnostics**: added `scripts/diagnose_stage2_masks.py` to compare selector vs baseline masks by level (gap stats, corruption magnitude, OOB fractions). Use this to quantify why selector‑conditioned masks may yield larger corrections.
- **Stage-2 model error diagnostics**: added `scripts/diagnose_stage2_model_error.py` to measure stage‑2 prediction loss per level for selector vs baseline masks using a trained stage‑2 checkpoint. This tests whether selector masks are harder **for the model itself**, not just in corruption stats.
- **KP features (gaps + t_norm)**: Implemented. Status: done.
- **Segment cost predictor D_phi**:
  - Model: `src/models/segment_cost.py`. Status: done.
  - Trainer: `src/train/train_segment_cost.py`. Status: done.
  - Integration into phase-1 training + sampling + stage-2 bootstrap: done (requires `--dphi_ckpt`).
- **Phase-1 capacity upgrade**:
  - Maze encoder now supports configurable depth via `maze_channels`.
  - KeypointDenoiser size configurable via CLI (d_model/layers/heads/ff).
  - Defaults in run script set to a larger model (384x12).
- **Stage-2 capacity upgrade**:
  - InterpLevelDenoiser size configurable via CLI and stored in ckpt meta.
  - Defaults in run script set to 384x12 with deeper maze encoder channels.
- **Unified DP pipeline script**:
  - `scripts/run_d4rl_unified_dp_train_sample.sh` updated to run DP prep → train D_phi → stage-1 → stage-2 → sample. Status: done.
- **Experiment logging helper**:
  - `scripts/log_experiment.sh` appends structured summaries to this file when `LOG_EXPERIMENT=1`. Status: done.
- **Efficiency tweaks**:
  - Avoid full-segment D_phi inference in phase-1 training and sampling: now compute segment features only for adjacent keypoint pairs (K-1) per batch and run D_phi on those. This reduces per-batch D_phi cost from O(B*T^2) to O(B*K). (`src/train/train_keypoints.py`, `src/sample/sample_generate.py`)
  - `interpolate_from_indices` now calls `idx.contiguous()` to avoid `torch.searchsorted` non-contiguous warning. (`src/corruptions/keyframes.py`)
- **Per-sample DP preprocessing**:
  - `prepare_dp_keypoints.py` now supports `--per_sample 1` and batch DP via `dp_select_indices_batch`, producing per-sample `kp_idx` (and kp_feat). (`src/selection/epiplexity_dp.py`, `src/data/prepare_dp_keypoints.py`)
- **Keypoint selector**:
  - New model `KeypointSelector` with **spatial cross-attention** over occ/sdf tokens, plus start/goal **as both spatial heatmaps and a token**.
  - Added optional **goal-distance token** (default on in unified script).
  - **Level-conditioned selector support**: selector can take a scalar `level` input (K_s/T or s/levels) and use it as a bias on time queries. Training supports per-level masks (`kp_mask_levels`) and samples a level per batch. (Implemented 2026-01-31.)
  - Selector training script: `src/train/train_keypoint_selector.py`.
  - Stage-1 training supports selector indices (`idx_source=selector` or mix includes `selector`), with `--selector_ckpt`. (`src/train/train_keypoints.py`)
  - Sampling supports `kp_index_mode=selector` with `--selector_ckpt`. (`src/sample/sample_generate.py`, `runs/d4rl_ablate_stage2_sample_only.sh`)
  - Stage-2 training/sampling now builds nested masks **from selector logits per level** (no random fill), consistent with “add most-informative points as noise decreases.” (`build_nested_masks_from_level_logits`, `train_interp_levels.py`, `sample_generate.py`)
- **Sampling position clipping**:
  - `sample_generate.py` supports `--pos_clip` (clamps pos dims to [min,max]) to prevent boundary blowup when logit-space is disabled.

### Experiments / Results
- 2026-01-30: d4rl_unified_dp run failed during D_phi training due to missing `CUBLAS_WORKSPACE_CONFIG` when deterministic algos are enabled. Fixed by exporting in `scripts/run_d4rl_unified_dp_train_sample.sh` and setting default in `src/train/train_segment_cost.py`. Rerun needed.
- 2026-01-30: d4rl_unified_dp run failed at stage-1 training with `UnboundLocalError: EMA` due to local import shadowing in `src/train/train_keypoints.py`. Fixed by removing inner import; rerun needed.
- 2026-01-30: d4rl_unified_dp run reached sampling, then crashed during logging with `OUT_ROOT: unbound variable` in `scripts/run_d4rl_unified_dp_train_sample.sh`. Fixed by defining OUT_ROOT default near top.
- 2026-01-30: manual sampling failed because `SAMPLE_CASES` defaulted to `none`, so `ckpt_interp` resolved to `checkpoints/none/ckpt_final.pt`. Fix: set `SAMPLE_CASES=interp_levels_unified_T128_dp` and use dp eval dataset.
- 2026-01-30: stage-2 training failed when loading stage-1 ckpt due to size mismatch (new larger kp model). Fixed by reading kp_* sizes from stage-1 meta when constructing bootstrap model in `train_interp_levels*.py`.
- 2026-01-30: stage-2 training failed with deterministic CuBLAS error (missing `CUBLAS_WORKSPACE_CONFIG`) when running outside the unified script. Fixed by setting default in `src/utils/seed.py` so all deterministic runs export the env var.
- 2026-01-31: sampling failed with `Checkpoint not found or invalid: checkpoints/none/ckpt_final.pt` because `SAMPLE_CASES` was unset, causing the sample script to use the default `"none"` case. Fix: set `SAMPLE_CASES=interp_levels_unified_T128_dp` (or desired stage2 folder) when running `runs/d4rl_ablate_stage2_sample_only.sh`.

## 2026-01-31

### Experiments / Results
- 2026-01-31: **D_phi + selector-only distillation (DP(D_phi) labels)**  
  - Script: `scripts/run_d4rl_train_dphi_selector.sh`  
  - Outputs:  
    - D_phi ckpt: `checkpoints/segment_cost_unified_T128_dp/ckpt_final.pt`  
    - Selector train data: `outputs/d4rl_prepared_unified_T128_train_dp_dphi/dataset.npz`  
    - Selector eval data: `outputs/d4rl_prepared_unified_T128_eval_dp_dphi/dataset.npz`  
    - Selector ckpt: `checkpoints/keypoint_selector_unified_T128_dp/ckpt_final.pt`  
  - Training results:  
    - D_phi loss: `0.1497` (final step)  
    - Selector loss: `0.9569` (final step)  
  - Selector diagnostic:  
    - `mae=24.10`, `overlap=0.489`  
    - Top interior predicted indices: `[1, 126, 21, 20, 22, 19, 18, 24, 23, 37]`  
  - Notes: selector now learns (loss below constant baseline), but still biases near ends; next step is to inspect DP(D_phi) label distribution to verify if labels themselves are clustered.

- 2026-01-31: **Selector retrain with cond-bias injection (use_cond_bias=1)**  
  - Script: `scripts/run_d4rl_train_dphi_selector.sh`  
  - Training results:  
    - D_phi loss: `0.1497` (final step)  
    - Selector loss: `0.3449` (final step, improved)  
  - Selector diagnostic (after fix):  
    - `mae=12.73`, `overlap=0.646`  
    - Top interior predicted indices: `[20, 21, 18, 19, 17, 39, 22, 23, 24, 25]`
  - Notes: cond-bias improves alignment substantially; still biased toward early indices. Next step: compare selector output distribution vs DP(D_phi) labels and verify per-maze variability.
- 2026-01-31: **Selector diagnostic vs DP(D_phi) label distribution**  
  - Script: `scripts/diagnose_selector.py`  
  - Results: `mae=13.00`, `overlap=0.641`  
  - Top interior label idx: `[110, 108, 111, 22, 93, 19, 20, 113, 90, 17]`  
  - Top interior pred idx: `[20, 21, 18, 17, 19, 22, 39, 23, 24, 25]`  
  - Notes: labels include late indices (~108–113) but selector predictions are still early-heavy → cond signal still too weak.
  - 2026-01-31 (cond_bias_mode=encoder): `mae=9.99`, `overlap=0.505`  
    - Top interior pred idx: `[110, 111, 109, 112, 113, 20, 18, 22, 36, 21]`  
    - Notes: predictions now include late indices (108–113) matching labels; overlap metric drops due to spread, but distribution alignment improved.

- 2026-01-31: **Per-maze selector diagnostics (cond_bias_mode=encoder)**  
  - Script: `scripts/diagnose_selector_per_maze.py`  
  - maze[0] mae=10.47 overlap=0.556  
    - label: `[113, 95, 78, 110, 76, 93, 22, 42, 61, 111]`  
    - pred : `[22, 113, 45, 24, 112, 115, 111, 25, 26, 114]`  
  - maze[1] mae=9.37 overlap=0.458  
    - label: `[111, 20, 22, 110, 114, 94, 23, 112, 79, 113]`  
    - pred : `[112, 113, 114, 111, 21, 20, 22, 41, 40, 23]`  
  - maze[2] mae=9.93 overlap=0.512  
    - label: `[108, 17, 19, 89, 107, 72, 18, 90, 54, 36]`  
    - pred : `[109, 18, 110, 36, 35, 19, 17, 108, 37, 34]`  
  - Notes: per-maze distributions match reasonably; selector captures late indices and some early indices per maze.
  - Follow-up: cond_bias_mode=encoder introduced; diagnostic script updated to detect `cond_enc.*` weights in ckpt.

- 2026-01-31: **Stage-2 mask diagnostics (selector vs uniform)**  
  - Script: `scripts/diagnose_stage2_masks.py`  
  - Key finding: **max_gap is dramatically larger with selector masks**, while mean_gap is the same.  
    - Example at K=8 (level=8):  
      - selector max_gap ≈ 62  
      - uniform max_gap ≈ 19  
  - Mean correction magnitudes slightly higher for selector across levels (`mean_adj`, `mean_xs`).  
  - OOB during corruption is low for both (≈0.1–0.2%), so off-screen drift likely comes from model updates on large-gap cases rather than corruption itself.  
  - Interpretation: selector clusters anchors (highly informative points) and leaves **very long gaps**, shifting the corruption distribution and making stage‑2 corrections much harder.
- 2026-01-31: **Stage-2 mask diagnostics (selector vs random_nested)**  
  - Script: `scripts/diagnose_stage2_masks.py --baseline random_nested`  
  - Result: random_nested also has large max_gap, but **smaller tail** than selector.  
    - Example at K=8 (level=8): selector max_gap ≈ 62 vs random_nested ≈ 46.  
  - Mean correction magnitudes (`mean_adj`, `mean_xs`) are slightly lower for random_nested, consistent with fewer extreme gaps.  
  - Interpretation: selector‑based masks create **heavier‑tailed gap distribution**, which likely destabilizes stage‑2 more than random_nested.

## 2026-02-01

### Experiments / Results
- **Resample comparison (selector vs prior uniform)**  
  - New run: `runs/ablate_stage2_unified_dp_resample_20260201_010819/interp_levels_unified_T128_dp_sample/metrics.csv`  
    - mean `collision_interp` ≈ **0.08**  
    - mean `collision_refined` ≈ **0.454**  
    - mean `collision_oracle_interp` ≈ **0.077**  
    - mean `collision_oracle_refined` ≈ **0.478**  
  - Old run (uniform masks): `runs/ablate_stage2_small_jitter/dist_jitter_sample/metrics.csv`  
    - mean `collision_interp` ≈ **0.288**  
    - mean `collision_refined` ≈ **0.291**  
    - mean `collision_oracle_interp` ≈ **0.276**  
    - mean `collision_oracle_refined` ≈ **0.339**  
  - Interpretation: selector‑conditioned stage‑2 is dramatically worse, even in oracle mode, indicating stage‑2 model/conditioning failure rather than visualization error.

- **Uniform-mask inference test on the same stage‑2 ckpt**  
  - Run: `runs/ablate_stage2_unified_dp_uniformmask_test/interp_levels_unified_T128_dp_sample/metrics.csv`  
    - mean `collision_interp` ≈ **0.010**  
    - mean `collision_refined` ≈ **0.461**  
    - mean `collision_oracle_interp` ≈ **0.0003**  
    - mean `collision_oracle_refined` ≈ **0.465**  
  - Conclusion: stage‑2 is broken **independent of selector masks** (oracle also degrades), so the issue is with stage‑2 training/inference mismatch, not keypoint selection.

### Pending / Next
- For maze: validate level‑conditioned selector + stage‑2 retrain; compare phase‑2 correction vs prior run.
- For video: add **teacher‑noise‑conditioned** D_phi (or per‑step KL distillation) and align selector level with teacher noise schedule; resolve mismatch between teacher noise and stage‑2 corruption noise.

## 2026-02-01

### Debug / Logging Fixes
- **Sampling NPZ mismatch**: `samples.npz` previously stored only the first `occ`/`sdf`, so rerenders could plot the wrong maze for multi‑maze runs. Fixed `src/sample/sample_generate.py` to save per‑sample `occ` and `sdf` arrays.  
- **Rerender script**: `scripts/rerender_sample_from_npz.py` now:
  - validates index range,
  - uses per‑sample `occ` if present,
  - overlays explicit start/goal markers (`--show_start_goal 1`).
- **Endpoint clamp sanity**: For sample 29 in `samples.npz`, `refined[0]` and `refined[-1]` exactly match `start_goal` (distance 0), and no OOB points. This suggests stage‑2 drift is not due to endpoint unclamping but due to model behavior on the interior.
- **Sampling mismatch fix**: `src/sample/sample_generate.py` now builds **anchor confidence per level** using the current `mask_s` (instead of the stage‑1 K_min mask) and clamps `all_anchors` using `mask_s`. This matches training (`conf_s` derived from mask_s/mask_prev) and prevents newly added anchors from being treated as missing during sampling. Also fixed a bug where **stage‑2 oracle updates were outside the loop**, so oracle was only updated once. (This affected oracle panels.)
- **Resample crash fix**: `sample_generate.py` raised `_build_anchor_conf() missing clamp_endpoints` during resample. Added `clamp_endpoints` argument to the new per‑level calls so sampling runs.
- **Prepared‑meta propagation**: `prepare_dp_keypoints.py` now copies `meta.json` from the source prepared dataset (when present) into the DP output directory, and forwards `d4rl_flip_y` into the `_meta.json` file. `sample_generate.py` now falls back to `{dataset}_meta.json` if `meta.json` is missing. This prevents flip‑y visualization mismatches in DP datasets.
- **Stage‑2 sampling noise knob**: added `--s2_sample_noise_sigma` (default 0) to optionally inject small Gaussian noise during stage‑2 reverse updates; `runs/d4rl_ablate_stage2_sample_only.sh` now forwards `S2_SAMPLE_NOISE_SIGMA`.
  - Added `--s2_sample_noise_mode {none,constant,level}` and `--s2_sample_noise_scale`.  
  - `level` mode ties sampling noise to the **training schedule** (uses stage‑2 ckpt meta `corrupt_sigma_*`) and applies noise **only to missing points**.  
  - `scripts/run_stage2_smallnoise_10k_and_sample.sh` now defaults to `S2_SAMPLE_NOISE_MODE=level` and uses the same `corrupt_sigma_*` for sampling.

### Experiments / Results
- **Stage‑2 (no corruption) 10k**
  - Script: `scripts/run_stage2_nocorrupt_10k_and_sample.sh`
  - Outputs:
    - ckpt: `checkpoints/interp_levels_unified_T128_dp_nocorrupt_10k/ckpt_final.pt`
    - logs: `runs/interp_levels_unified_T128_dp_nocorrupt_10k`
    - samples: `runs/ablate_stage2_unified_dp_nocorrupt_10k/interp_levels_unified_T128_dp_nocorrupt_10k_sample/`
  - Observation (user): stage‑2 is **not worsening** as before, but **still not correcting** interpolation errors; refined trajectories often remain off‑track. Hypothesis: deterministic corruption (no Gaussian noise) may be too brittle; add small stochastic noise (Soft Diffusion style).
  - Next: run `scripts/run_stage2_smallnoise_10k_and_sample.sh` (small dist noise) for comparison.

- **Stage‑2 (small dist noise) 10k + level‑tied sampling noise**
  - Samples: `runs/ablate_stage2_unified_dp_smallnoise_10k/interp_levels_unified_T128_dp_smallnoise_10k_sample/`
  - Computed from `samples.npz` (metrics.csv was empty at time of inspection):
    - interp collision ≈ **0.0795**
    - refined collision ≈ **0.0769** (Δ ≈ **-0.0027**)
    - interp MSE‑to‑GT ≈ **0.00164**
    - refined MSE‑to‑GT ≈ **0.00143**
    - collision improved in **32%** of samples; worse in **30%** (rest unchanged)
    - MSE improved in **56%**; worse in **44%**
  - Interpretation: stage‑2 is **slightly corrective on average**, but improvements are weak and inconsistent; many cases still degrade (especially for collision).
  - Training loss curve (TensorBoard `train/loss`):
    - step 0: **0.2922**
    - step 5000: **0.0003207**
    - step 9900: **0.0001427**
    - Last 5 steps: 0.0001947, 0.0004334, 0.0001578, 0.0001770, 0.0001427 (small jitter around ~1e‑4).

- **Stage‑2 model error diagnostic (selector vs uniform masks)**  
  - Script: `scripts/diagnose_stage2_model_error.py`  
  - CKPT: `checkpoints/interp_levels_unified_T128_dp_smallnoise_10k/ckpt_final.pt`  
  - Eval: `outputs/d4rl_prepared_unified_T128_eval_dp/dataset.npz`  
  - Results (loss by level):  
    - selector @ level 8 (K=8): **0.000859** vs uniform_base **0.000165**  
    - selector losses are higher than uniform across all levels, with a large spike at the coarsest level.  
  - Gaps: selector max_gap up to **62** vs uniform_base **19**.  
  - Interpretation: stage‑2 struggles most at coarsest levels when selector masks create very large gaps; error grows with gap tail.

### Implementation Updates
- **Stage‑2 mask mixing (training)**: `train_interp_levels.py` now supports `--mask_policy_mix` (e.g., `selector:0.5,uniform:0.5`) to sample a mask policy per batch. Allowed: `selector`, `uniform`, `random_nested`, `dp_precomputed`. This helps keep the correction distribution non‑degenerate when selector masks yield large gaps.
  - `scripts/run_stage2_smallnoise_10k_and_sample.sh` accepts `MASK_POLICY_MIX` and forwards it to training.
  - Fix: selector is now loaded if **either** `kp_index_mode=selector` or `mask_policy_mix` includes `selector` (previously required kp_index_mode=selector).

## 2026-02-01 (Video Transition)

### Decisions / Requests
- Move to **LSMDC** for text‑conditioned video generation (prompt‑only at inference).
- Use **padded_start/end** timestamps from `LSMDC16_annos_*_someone.csv` (task1) and **crop to 5s** if longer.
- Store downloaded videos under `data/LSMDC/videos/`.

### Implementation Updates
- Added `src/data/lsmdc.py` (LSMDC dataset) with CSV parsing, padded windows, optional 5s crop.
- Added `src/data/lsmdc_cache.py` and `scripts/datasets/lsmdc/precompute_cache.py` for latent/text caching.
- Updated `src/data/video_io.resolve_video_path` to support `.avi`/`.mkv`.
- Updated `data/LSMDC/download_videos.sh` and `para_phrases_download.sh` to download annotations into `data/LSMDC/task1` and videos into `data/LSMDC/videos/`.
  - Download now preserves per‑movie subfolders (`wget -x -nH --cut-dirs=3`) instead of a flat directory.

## 2026-02-01 (Switch to DiDeMo / HF)

### Decisions / Requests
- Switch from LSMDC to **DiDeMo** (HF dataset) for text‑conditioned video generation.
- Store videos under `data/didemo/videos/` and use existing `train_data.json`/`val_data.json`/`test_data.json`.
- Prefer **fixed 5s clips**; if segment length differs, we will clip (center/random) to 5s and sample a fixed number of frames (time‑normalize).

### Implementation Updates
- Added `scripts/datasets/didemo/download_videos_hf.py` to download DiDeMo mp4 tar parts from HuggingFace and extract into `data/didemo/videos/`.
- Added `scripts/datasets/didemo/setup_didemo_hf.sh` to run metadata fetch + HF download in one step.
- Updated `scripts/datasets/didemo/README.md` with HF instructions.
- `src/data/video_io.resolve_video_path` now also tries swapping extensions (e.g., metadata `.avi` → `.mp4`) to match HF files.

### Wan2.1 Training References (Self-Forcing / TurboDiffusion)
- **Self‑Forcing (guandeh17)**  
  - Reuses Wan2.1 backbone, **freezes text encoder + VAE** (`requires_grad_(False)`); trains only diffusion backbone(s).  
  - `model/base.py`: `generator` and `fake_score` trainable; `real_score`, text encoder, VAE frozen.  
  - Uses **custom training pipeline** (self‑forcing, DMD/ODE init, AR rollout), not vanilla diffusion finetune.  
  - Code inspected in `/workspace/Interpolated_Diffusion/tmp_repos/Self-Forcing`.  

- **TurboDiffusion (thu‑ml)**  
  - Reuses Wan2.1 backbone, **freezes VAE + conditioner** (asserts conditioner has no trainable params).  
  - Teacher / EMA are frozen; student net trainable.  
  - **SLA integration**: swaps only attention operator inside WanSelfAttention via `replace_attention_with_sla`, keeping all other Wan weights intact.  
  - Code inspected in `/workspace/Interpolated_Diffusion/tmp_repos/TurboDiffusion`.  

### Synthetic Wan2.1 dataset (TurboDiffusion)
- TurboDiffusion uses **Wan2.1‑synthesized webdataset shards** (latents + text embeddings + prompt text), not the original Wan training set.  
- Added loader + download scripts:
  - `src/data/wan_synth.py` (`create_wan_synth_dataloader`)
  - `scripts/datasets/wan_synth/download_wan_synth.py`
  - `scripts/datasets/wan_synth/README.md`

## 2026-02-02 (DiDeMo layout + Wan synth size)

### Implementation Updates
- Added `scripts/datasets/didemo/extract_and_flatten_hf.sh` and `scripts/datasets/didemo/fix_video_folder_layout.sh` to flatten HF tar extraction and move `data/didemo/videos/video` → `data/didemo/video`.
- DiDeMo counts (after flatten): `data/didemo/video/train` = 8395, `data/didemo/video/test` = 1004.
- Added ignore rules for dataset/large artifacts: `data/LSMDC/`, `data/wan_synth/`, `tmp_repos/`, `notes/*.tgz`.

### Notes / Issues
- Wan2.1 synthetic dataset on HF is very large. The 250K dataset has **~970 shards** at ~**2.15GB** each → **~2.1TB total**. Downloads hit disk quota. Use subset by pattern (e.g. `shard-0000*.tar` for ~10 shards ≈ 21GB).
- HF progress bar “incomplete total” is misleading; compute total via shard count × average shard size.

### Next Direction (Video)
- Use Wan2.1 synthetic webdataset subset for initial experiments (latents + text embeds + prompts).
- Implement phase‑1/phase‑2 finetuning on Wan2.1 backbone (LoRA likely), keep VAE + text encoder frozen.
- D_phi for video should be trained with a **teacher diffusion** (Wan2.1 full‑sequence) and a **student** defined as keyframes+interpolator. Selector then distills per‑segment costs; confirm per‑step KL weighting schedule.

## 2026-02-02 (State Summary / Theory)

### Project Goal
Two‑phase diffusion model for sequence/video generation:
- **Phase‑1 (keyframes/keypoints)**: generate a sparse set of keyframes/points from noise.
- **Interpolator**: fill missing frames/points in latent space (or position space in maze).
- **Phase‑2 (refiner)**: denoise the full sequence given keyframes+interp as corrupted input.

This is inspired by **cold/soft diffusion**: corruption is not necessarily Gaussian; still treat it as a diffusion‑like denoising process. Phase‑2 corruption combines **interpolation replacement** + optional **Gaussian noise on missing tokens only**.

### Why two‑phase
- Full‑sequence diffusion is expensive and overkills for structured sequences.
- Keyframes capture “structure”; interpolator captures low‑entropy content.
- Phase‑2 corrects interpolator errors (ideally) and learns a clean full‑sequence model.

### Key technical pieces (Maze)
- Keypoints are **nested masks**: more keypoints = lower noise level.
- DP selection: choose K indices to minimize a KL proxy between **teacher** and **student** distributions.
  - Maze teacher ≈ GT trajectories (acceptable proxy).
  - Student = keypoints + interpolation; cost computed per‑segment.
- D_phi predicts per‑segment difficulty/cost for DP; selector predicts keyframe indices from cond.
- Selector should provide **informative** keyframes; phase‑2 corruption removes least‑informative points first in foward process, so that in backwards process, the most informative will be added back.

### Phase‑1 details
- Input: K keypoints + index mask, plus optional features (gaps, t_norm, difficulty).
- Conditioned on maze map + start/goal (cond_start_goal=1).
- Index policy mix: selector/uniform/random for robustness.

### Phase‑2 details
- Corruption: replace non‑anchors with interpolated points (and optional Gaussian noise on missing points).
- Nested masks per level (K_s) – as noise decreases, **more keypoints are revealed**.
- Forward process: remove least‑informative points first; backward: add most‑informative points.

### Observations (Maze)
- Selector helps **phase‑1** keypoints substantially.
- Phase‑2 sometimes fails to correct when interpolations are already good; tends to “do nothing.”
- Uniform masks previously gave stronger corrections but worse keyframes.
- Selector masks create larger max gaps (heavy tails), making phase‑2 harder.
- Adding small Gaussian noise + matching inference noise to training schedule (noise on missing points only) improves stability; still needs more correction strength.

### Video Transition
- **Teacher for D_phi must be a full video diffusion model** (Wan 2.1 or similar). Maze GT is not sufficient.
- D_phi should be trained using **diffusion‑per‑step KL** between teacher and student:
  - Teacher: full‑sequence denoiser.
  - Student: keyframes + interpolator.
  - Weight per step using SNR schedule (s_min=0.1, s_max=10, gamma=1).
- Selector then distills D_phi outputs → per‑segment costs → DP labels → predicts indices from cond only.
- Phase‑1/Phase‑2 should both be **Wan2.1 finetunes** (likely LoRA), keeping VAE + text encoder frozen.

### Current Video Plan
1) Use **Wan2.1 synthetic webdataset** subset (latents + text embeds + prompts) for initial testing.
2) Implement phase‑1 + phase‑2 finetunes on Wan2.1 backbone.
3) D_phi training uses teacher Wan2.1; selector trained from D_phi DP labels.
4) Interpolation in **latent space** (VAE). Keyframes are selected latent tokens.
5) Conditioning: text prompt + clip length (and later segment difficulty tokens).

### Key Datasets
- **DiDeMo**: HF video tar parts; flattened to `data/didemo/video/{train,test}` (8395/1004). Used for text‑conditioned baseline experiments.
- **Wan2.1 synthetic**: webdataset shards with `*.latent.pt`, `*.embed.pt`, `*.prompt.txt`. Large (≈2.1TB for 250K). Use shard subset.

### Open Questions (Video)
- Teacher/Student noise mismatch: D_phi should align with teacher diffusion noise; phase‑2 corruption is interpolation+Gaussian.
- How to best condition selector without GT: text prompt + clip length, maybe coarse motion cues.
- Phase‑2 correction strength when phase‑1+interp is already good; may need noise schedule, negative mining, or mask mixing.

### Next Steps
- Upload checkpoints/runs and codex state to Drive.
- Move to B200 instance; re‑setup env from `notes/pip_freeze.txt`.
- Download small subset of Wan2.1 synthetic shards (e.g. `shard-0000*.tar`) and verify loader.
- Implement Wan2.1 LoRA finetune for phase‑1 and phase‑2.

## 2026-02-02 (Milestone / RCM distillation)

### Milestone Definition
- Target: after distillation, achieve **quality comparable to 4‑step RCM** with:
  - few steps for Phase‑1 keyframes,
  - interpolation,
  - **1–2 step** Phase‑2 correction (ideally 1 step).
- Goal: demonstrate **compute efficiency vs full‑sequence diffusion** while maintaining comparable quality.

### Distillation Direction
- Add **RCM / step distillation** to reduce sampling steps for Phase‑1 and Phase‑2.
- Phase‑1 is already cheaper; Phase‑2 should converge to 1–2 steps post‑distillation.
- SLA / AR diffusion are orthogonal and postponed.

### Next Implementation Plan (Video)
- Build teacher full‑sequence model (Wan2.1) and distill:
  1) Phase‑1 keyframe diffusion to few steps,
  2) Phase‑2 refiner to 1–2 steps.
- Evaluate vs 4‑step RCM baselines.
