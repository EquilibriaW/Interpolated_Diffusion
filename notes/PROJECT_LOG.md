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

## 2026-02-06

### Video (Wan2.1-Synth) Notes
- Dataset shard subset used for rapid iteration:
  - Train: `data/wan_synth/.../shard-0000[0-7].tar`
  - Val: `data/wan_synth/.../shard-0000[8-9].tar`
- Wan2.1-synth shapes:
  - Latents: `[B, T=21, C=16, H=60, W=104]` (bf16)
  - Text embeddings: `[B, L=512, D=4096]` (bf16)
- `TextConditionEncoder` updated to:
  - pool over sequence dims (`[B,L,D] -> [B,D]`) for cond-vector models (D_phi/selector),
  - cast input dtype to match module params (fixes bf16 input vs fp32 weights crash).

### Straightener Status
- Best straightener so far is the token-grid transformer (~14.35M params):
  - ckpt: `checkpoints/latent_straightener_toktr_d256_L9_ff1024_b64_10k_l2_v2/ckpt_final.pt`

### Video D_phi (Segment Cost) + Selector (DP Labels)
- Added video-specific implementations:
  - D_phi trainer: `src/train/train_segment_cost_wansynth.py` (teacher-student eps MSE on interior frames)
  - Selector model: `src/models/video_selector.py`
  - Selector trainer: `src/train/train_video_selector_wansynth.py` (DP labels from D_phi)

- D_phi v1 experiment (cost averaged over interior frames, later identified as non-additive for DP):
  - ckpt: `checkpoints/segment_cost_wansynth_v1_b8_200/ckpt_final.pt`
  - val (step=100): `val/loss ~= 0.0743`
  - Key issue: DP indices were prompt-independent and degenerate; for `K=4`, DP consistently returned `[0, 18, 19, 20]` across shards.

- Selector v1 experiment (trained against DP(D_phi) labels above):
  - ckpt: `checkpoints/video_selector_wansynth_v1_dphi200/ckpt_final.pt`
  - Observations:
    - training loss collapsed to ~0, indicating labels were nearly deterministic/easy,
    - eval overlap for `K=4` stayed at `0.5` (endpoints only), i.e. no interior match.

- Diagnosis and follow-up:
  - Segment cost used for DP must be additive across time; averaging over interior frames breaks additivity and encourages boundary-adjacent degeneracies.
  - Updated cost definition in D_phi training to scale with interior length (sum over interior frames).
  - Early re-trains with additive cost still produced degenerate DP labels (e.g. `[0, 1, 2, 20]` for `K=4`) and remained essentially prompt-independent.
  - Conclusion: the current eps-diff target is dominated by gap/time terms (and/or too weakly prompt-dependent) to produce meaningful prompt-adaptive keyframe DP labels; needs a rethink (target definition, features, or training objective) before investing in long D_phi/selector runs.
  - Added `--target_mode {teacher_eps,latent_mse}` to allow a cheap latent-space target; initial latent-MSE run still produced near-degenerate DP paths (mostly `[0, 13/14, 19, 20]` for `K=4`), indicating weak prompt dependence even for direct interpolation error.

## 2026-02-08

### Oracle DP Investigation (Wan2.1-synth)
- **Motivation**: verify whether DP-optimal keyframes vary across samples/prompts when using an *oracle* cost (computed from real latents), before blaming D_phi/selector training.
- **Code**:
  - `src/selection/oracle_segment_cost.py`: exact interior-frame oracle segment cost for all (i,j), additive for DP.
  - `scripts/diagnose_oracle_dp_wansynth.py`: compute oracle cost-matrix, run DP, and report sequence diversity + repeat-prompt stats.
  - `src/train/train_video_selector_wansynth.py`: added `--label_mode {dphi,oracle_z_mse,oracle_s_mse}` to train selector from oracle DP labels for debugging identifiability.

### Findings (val shards `shard-0000[8-9].tar`, K=4)
- Oracle DP schedules are **not constant** across samples, but are **deterministic within repeated prompts** in sampled batches.
  - Oracle `z_mse` (raw latent LERP error): ~`57` unique sequences out of `800` samples (top mode `[0,3,12,20]`).
  - Oracle `s_mse` (straightener.encode space): ~`81` unique sequences out of `800` samples (top modes around `[0,6/7,13/14,20]`).
- A simple **fixed schedule** is already a strong baseline under oracle `z_mse`:
  - Always selecting `[0,3,12,20]` gives `overlap_int ≈ 0.348` vs oracle DP labels on the val subset (significantly above random for 2 interior picks).

### Selector From Oracle Labels (Generalization Test)
- Trained `VideoKeyframeSelector` using oracle DP labels (no D_phi), and evaluated on val:
  - Multi-K (levels=6) run reached `val/overlap_int ≈ 0.041` at step 400 (worse than the fixed-schedule baseline).
  - Fixed-K=4 (levels=1, `--use_level 0`) reached `val/overlap_int ≈ 0.112` at step 200 and `≈ 0.096` at step 400 (near random; still far below the fixed `[0,3,12,20]` baseline).
- Interpretation: prompt-only selector training appears to **overfit** and **does not generalize** well; the mapping from text embeddings to DP-optimal indices is not reliably learnable under this oracle target on held-out shards.
  - This supports the earlier suspicion that prompt-conditioned D_phi/selector is fundamentally limited unless we provide additional inference-time signals beyond the prompt (e.g., anchors/preview/content features), or accept a fixed schedule.

## 2026-02-04

### Decisions / Requests
- **Performance**: GPU underutilized during Wan synth interpolator training → prioritize data pipeline parallelism and overlap H2D with compute.
- **Latent straightening**: investigate making Wan2.1 VAE latents more interpolation‑friendly; prefer a cheap method (OK to add an adapter if raw frames are unavailable).

### Implementation Status
- **Dataloader parallelism** (Wan synth): added optional `prefetch_factor`, `persistent_workers`, `pin_memory` to WebDataset loaders (`src/data/wan_synth.py`).
- **Training loop perf** (Wan synth flow/lerp interpolator): added vectorized triplet sampling, optional GPU prefetcher, and optional `torch.compile` (`src/train/train_flow_interpolator_wansynth.py`).
- **Latent straightener (Option B)**: added `LatentStraightener` model (`src/models/latent_straightener.py`) and training script (`src/train/train_latent_straightener_wansynth.py`) with linearity + recon losses and throughput knobs.
- **Straightener integration**: flow/lerp interpolator training can optionally run in straightened space via `--straightener_ckpt` (`src/train/train_flow_interpolator_wansynth.py`).

### Notes / Plan
- **Dataset check**: Wan synth shards currently contain only `latent.pt`, `embed.pt`, `prompt.txt` (no raw frames). VAE fine‑tune for straightening would require raw videos; otherwise use a **latent‑space straightener**:
  - Train small adapters `S` and `R` on latent triplets so `R(S(z))≈z` and `S(z_t)≈(1‑α)S(z_0)+αS(z_1)` (linearity loss).
  - Use `S` before interpolation and `R` after, keeping decoder/denoiser in original latent space.

## 2026-02-03

### Implementation Fixes (Wan synth pipeline)
- `src/data/wan_synth.py`: replaced unsupported `wds.zip` with `_ZippedIterableDataset` for pairing anchors + base; added `prefetch_factor=None` when `num_workers=0`.
- `scripts/run_wansynth_pipeline_debug.sh` and `scripts/run_wansynth_pipeline_full.sh`: added `ANCHOR_NUM_WORKERS` (default `0`) for deterministic anchor ordering.
- `scripts/datasets/wan_synth/precompute_phase1_anchors.py`: warning when `num_workers>0` (can reorder samples vs anchors).
- `src/corruptions/video_keyframes.py`:
  - `build_video_token_interp_adjacent_batch` now accepts `anchor_idx` and handles anchor replacement for levels where `K_s > K_base` via lookup (only replaces indices present in `anchor_idx`).
  - `build_video_token_interp_level_batch` signature now includes `anchor_values`/`anchor_idx` and uses the same lookup logic.
  - Flow interpolator inputs cast to flow dtype; flow outputs cast back to token dtype to avoid bf16/fp32 mismatches.
- `src/train/train_interp_levels_wansynth.py`: `_anneal_conf` now broadcasts correctly for `[B,T,N]` tensors.

### Debug Run (Wan synth)
- Script: `scripts/run_wansynth_pipeline_debug.sh`
- Command:
  - `DATA_PATTERN="data/wan_synth/Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K/shard-0000*.tar"`
  - `WAN_DTYPE=bf16 BATCH=2 NUM_WORKERS=8 SHUFFLE_BUFFER=1000`
  - `ANCHOR_NUM_WORKERS=0 ANCHOR_OUT=data/wan_synth_anchors_debug_ordered8`
  - `FLOW_STEPS=2 P1_STEPS=2 P2_STEPS=5`
  - `FLOW_SAVE_EVERY=1 P1_SAVE_EVERY=1 P2_SAVE_EVERY=1`
- Result: **debug pipeline completed successfully**.
  - Flow loss ≈ `0.2372`
  - Phase‑1 loss ≈ `1.4062`
  - Phase‑2 loss ≈ `0.7035`
  - Phase‑1/2 step time ≈ `7–10s` per step (bf16, B=2, grad‑ckpt on).
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


### Decisions / Requests
- Prefer **fixed 5s clips**; if segment length differs, we will clip (center/random) to 5s and sample a fixed number of frames (time‑normalize).

### Implementation Updates
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


### Implementation Updates
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

## 2026-02-03

### Decisions / Requests
- Reaffirmed video direction: **Wan2.1 synthetic (TurboDiffusion) dataset** is the initial training data source; download a shard subset for early experiments.
- Phase‑1 and Phase‑2 will be **Wan2.1 finetunes**. Start with **LoRA** for iteration speed; after stability, **unfreeze attention + MLP** blocks to increase capacity (keep VAE + text encoder frozen).
- Keep selector **entropy‑controlled** (Gumbel‑Top‑K + KL‑to‑uniform + temperature anneal) for video experiments.
- Training plan for Wan2.1: debug with a **small shard subset**, then train on the **full synthetic dataset**. Run **three training jobs**:
  1) LoRA rank **r=8**,
  2) LoRA rank **r=32**,
  3) Full finetune on top of Wan2.1 weights (no LoRA).
- D_phi/DP for video: **student uses teacher model with non‑keyframes replaced by interpolator output**; DP selects masks to minimize teacher‑vs‑student KL aggregated over diffusion steps.
- Use **D_phi‑derived difficulty features immediately** for pipeline debugging (not deferred).
- Potential objective mismatch between selector (teacher‑KL) and refiner loss is **flagged only**, not assumed.
- Phase‑2 corruption should **replace anchors with actual Phase‑1 outputs** (not Gaussian noise). This should match the true anchor error distribution.
- Phase‑2 training will use **Phase‑1 outputs with replacement probability p** (anchor replacement is stochastic). Phase‑1 is **frozen** during Phase‑2 training. Phase‑1 outputs for Phase‑2 are **precomputed** (cached).
- Wan2.1 backbone should be instantiated at **full size** (≈1.3B/“1.4B”), **no layer reduction** or slim configs.
- Interpolator design direction: **one‑pass, deterministic latent‑space motion‑compensated warp + blend** (no iterative VFI).  
  - Predict forward/backward flow + occlusion/blend mask (and optional tiny residual) from endpoint latents.  
  - Warp endpoints in latent space, blend with mask; optional residual refinement.  
  - Train with latent L1 (optional decode‑space L1), endpoint consistency, and gap‑length distribution matching.  
  - Optionally output uncertainty mask to guide phase‑2 corruption/noise injection (focus denoiser on unreliable regions).  
  - Keep interpolator **cheap** (small convnet, 2–3 scales max, no heavy attention).  

### Implementation Updates
- Added **LoRA injection utilities** (`src/models/lora.py`) targeting attention + MLP (configurable), with trainable‑params logging.
- Added Wan2.1 synthetic **Phase‑1/Phase‑2 token‑space training scripts**:
  - `src/train/train_keypoints_wansynth.py`
  - `src/train/train_interp_levels_wansynth.py`
- Added **latent‑space interpolator training** for Wan synth: `src/train/train_video_interpolator_wansynth.py`.
- Added **Phase‑1 anchor precompute** for Wan synth: `scripts/datasets/wan_synth/precompute_phase1_anchors.py`.
- Video corruption now supports **anchor replacement with external Phase‑1 anchors** (`anchor_values`, `anchor_idx`) in `src/corruptions/video_keyframes.py`.
- Default video interpolation mode in Wan synth stage‑2 training set to **smooth** (not learned).
- Added **latent flow‑warp interpolator** with **mask + uncertainty** outputs (`src/models/latent_flow_interpolator.py`) and training script `src/train/train_flow_interpolator_wansynth.py`.
- Integrated **flow interpolation + uncertainty‑gated corruption** into Wan synth Phase‑2 (token) corruption and added flags in `train_interp_levels_wansynth.py`.
- Added optional **flow interpolation** path for Wan backbone Phase‑1 training and Phase‑1 anchor precompute (`train_keypoints_wansynth.py`, `precompute_phase1_anchors.py`).
- Added pipeline scripts: `scripts/run_wansynth_pipeline_debug.sh` and `scripts/run_wansynth_pipeline_full.sh`.
- Wan synth dataset is present at `data/wan_synth/Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K/` (tar shards).
- Added throughput logging (samples/sec, frames/sec, peak memory) to Wan synth training scripts.
- Debug run note: Wan synth latents are stored as **[C,T,H,W]** (e.g. [16,21,60,104]); loader now auto‑transposes to **[T,C,H,W]** and warns once. Flow interpolator training now disables deterministic algorithms to allow `grid_sample` backward.
- Pipeline scripts now call precompute with `PYTHONPATH=.` so `src` imports resolve.
- Precompute anchors now casts Wan inputs to model dtype (bf16) and initializes keypoint noise in model dtype to avoid dtype mismatch errors.

### 2026-02-03 (PM) — Pipeline Optimizations + Calibration

#### Code Changes
- `src/data/wan_synth.py`:
  - Added `_KeyJoinIterableDataset` for key‑based anchor joins with optional `allow_missing`.
  - `create_wan_synth_anchor_dataloader` now accepts `join_by_key`, `max_key_buffer`, and `allow_missing`.
- `src/train/train_interp_levels_wansynth.py`:
  - Added CLI flags: `--anchor_join`, `--anchor_key_buffer`, `--anchor_allow_missing`.
- `src/models/latent_flow_interpolator.py`:
  - Vectorized `interpolate()` segment fill (removes per‑t inner loop).
  - `load_latent_flow_interpolator` accepts `dtype` and honors `meta["model_dtype"]`.
- `src/train/train_flow_interpolator_wansynth.py`:
  - Added `--model_dtype` and `--log_every`.
  - Model/latents cast to `model_dtype`; `meta` stores `model_dtype`.
- `src/train/train_keypoints_wansynth.py` and `src/train/train_interp_levels_wansynth.py`:
  - Explicitly cast `latents`/`text_embed`/`anchor_values` to Wan model dtype (bf16).
  - Flow interpolator loaded with bf16; flow inputs cast to flow dtype.
- `src/diffusion/ddpm.py`: fixed broadcasting for 4‑D tensors in `q_sample`, `predict_x0_from_eps`, and `ddim_step` (unsqueeze to match dims).
- `scripts/run_wansynth_pipeline_*`: added `FLOW_DTYPE` default to `WAN_DTYPE`, and anchor join flags in debug/full scripts.

#### Throughput Calibration — Phase‑1 (Wan2.1 + LoRA r=8, flow interp)
Dataset: `shard-0000*.tar`, T=21, bf16, grad‑ckpt on, log_every=1.
- **B=2**: step ≈ **6.92s** → ~**0.29 samples/s**, **~6.1 frames/s**
- **B=4**: step ≈ **13.75s** → ~**0.29 samples/s**, **~6.1 frames/s**
- **B=8**: step ≈ **27.3s** → ~**0.29 samples/s**, **~6.1 frames/s**
**Observation:** throughput per sample is flat; larger batch only increases latency.

#### Throughput Calibration — Phase‑2 (Wan2.1 + LoRA r=8, anchors, flow interp)
Anchors: `data/wan_synth_anchors_calib3/` (ddim_steps=4, B=4, 20 batches); joiner with `allow_missing=1`.
- **B=2**: step ≈ **7.44s** → ~**0.27 samples/s**, **~5.6 frames/s**
- **B=4**: step ≈ **14.17s** → ~**0.28 samples/s**, **~5.9 frames/s**
- **B=8**: step ≈ **28.6s** → ~**0.28 samples/s**, **~5.9 frames/s**
**Observation:** similar flat throughput; batch size doesn’t improve samples/sec.

#### Anchor Precompute Throughput (Phase‑1 outputs)
- **DDIM 20 steps, B=4**: ~**61s/batch** (~0.066 samples/s, ~1.4 frames/s). Major bottleneck.
- **DDIM 4 steps, B=4**: ~**9.65s/batch** (~0.41 samples/s, ~8.7 frames/s).
- **DDIM 4 steps, B=2**: ~**5–6s/batch** (~0.35–0.40 samples/s).

#### Long‑Run Calibration Attempt
- Phase‑1 long run (B=4, steps=100, log_every=10) hit tool timeout at step ~86.
  - Step time was stable around **13.75s**, consistent with short‑run throughput.

### 2026-02-03 (PM) — TurboDiffusion Repo + Dataset Timing Notes
- TurboDiffusion synthetic dataset latents are **[C=16, T=21, H=60, W=104]**; the Wan2.1 tokenizer uses **temporal compression factor 4**, so **T=21 latent frames corresponds to 81 pixel frames**.  
  - Sources (local clone): `tmp_repos/TurboDiffusion/turbodiffusion/rcm/tokenizers/wan2pt1.py` (temporal factor + latent frame formula), and `tmp_repos/TurboDiffusion/turbodiffusion/rcm/datasets/build_synthetic_dataset.py` (default `--num_frames 81`).  
  - If using the repo’s default visualization fps=16, this is ~**5.1s** per clip (assumed; fps not explicitly stored in dataset shards).  
- **SLA training (Wan2.1 1.3B)** config: `tmp_repos/TurboDiffusion/turbodiffusion/rcm/configs/experiments/sla/wan2pt1_t2v.py`  
  - `max_iter=100_000`, `batch_size=4`, `lr=1e-5`, `sla_topk=0.1`, `state_t=21`, `teacher_guidance=5.0`, `precision=bfloat16`.  
  - Uses teacher ckpt; **teacher frozen** and **conditioner frozen**; **student net trained**.  
  - See model code: `tmp_repos/TurboDiffusion/turbodiffusion/rcm/models/t2v_model_sla.py` (teacher `requires_grad_(False)`, conditioner no trainable params).  
- **rCM distillation (Wan2.1 1.3B)** config: `tmp_repos/TurboDiffusion/turbodiffusion/rcm/configs/experiments/rcm/wan2pt1_t2v.py`  
  - `max_iter=100_000`, `batch_size=1`, `lr=2e-6`, `student_update_freq=5`, `tangent_warmup=1000`, `state_t=20`, `max_simulation_steps_fake=4`.  
  - Teacher/conditioner frozen; student net + optional fake‑score net trained.  
  - See model code: `tmp_repos/TurboDiffusion/turbodiffusion/rcm/models/t2v_model_distill_rcm.py`.

### 2026-02-03 (PM) — Wan2.1 Attention vs MLP Profiling (Diffusers)
- Profiled **Wan2.1 1.3B Diffusers** forward pass on B200 using a real Wan synth sample (T=21 latent frames, bf16).  
  - Hooked `WanAttention` and `FeedForward` modules; measured with CUDA events for one forward.  
  - **B=1:** total ≈ **765 ms**, attention ≈ **81%**, MLP ≈ **5%**, other ≈ **14%**.  
  - **B=2:** total ≈ **1527 ms**, attention ≈ **80%**, MLP ≈ **5.5%**, other ≈ **14%**.  
- Conclusion: **attention dominates runtime (~80%)**, so SLA‑style sparsity is likely to yield large speedups if kernels are available.

### 2026-02-03 (PM) — SageSLA Integration (Diffusers Wan2.1)
- Added **SageSLA/SLA attention processor** for Diffusers Wan2.1: `src/models/wan_sla.py`.  
  - Implements a `WanSLAProcessor` (per‑layer) and `apply_wan_sla()` to swap `WanAttention` processors.  
  - Supports `attention_type={sla,sagesla}` and `sla_topk`.  
- Wired flags into Wan synth training + precompute:  
  - `src/train/train_keypoints_wansynth.py`  
  - `src/train/train_interp_levels_wansynth.py`  
  - `scripts/datasets/wan_synth/precompute_phase1_anchors.py`  
  - Pipeline scripts: `scripts/run_wansynth_pipeline_debug.sh`, `scripts/run_wansynth_pipeline_full.sh`  
- New CLI flags: `--wan_attn {default,sla,sagesla}` and `--sla_topk`.  
- Decision leaning: **skip LoRA when using SLA** (so SLA’s `proj_l` parameters can train; LoRA-only would freeze them).  
- Next: **top‑k sweep** to find quality/throughput knee (paper warns quality drops at extreme sparsity).  

### 2026-02-03 (PM) — SageSLA Build Fix
- **SageSLA build failed** with NVCC redefinition errors from `/usr/include/.../c++config.h` when using upstream `SpargeAttn`.  
- Installed `ninja-build` and **GCC/G++ 12**, but errors persisted.  
- **Patched local SpargeAttn** at `tmp_repos/SpargeAttn/setup.py` to **remove** `-Xcompiler -include,cassert` from `NVCC_FLAGS` (comment said it was for SM90+, but it triggered redefinition errors in this environment).  
- Built + installed from local path with GCC 12: `pip install --no-build-isolation ./tmp_repos/SpargeAttn`  
- Installed missing dependency `einops`; now `spas_sage_attn` loads and **`SAGESLA_ENABLED=True`** in `SLA.core`.  

### 2026-02-03 (PM) — SageSLA NaN Fix + Anchor Load Robustness
- **Root cause of NaNs**: `WanSLAProcessor._run_sla` permuted Q/K/V to `(B,H,L,D)`, but **SLA expects `(B,L,H,D)`** and internally transposes. This swapped L↔H, producing invalid block maps and NaNs for any sparsity (<1.0).  
- **Fix**: pass Q/K/V to SLA **without permuting**. (`src/models/wan_sla.py`)  
- **Verification**: forward pass with `sagesla` and `sla` at `topk=0.07` produces **no NaNs** on Wan2.1 diffusers model.  
- **Anchor precompute fix**: apply SLA **before** loading checkpoints and use `strict=False` with validation so SLA keys are handled correctly; error if non‑SLA keys mismatch. (`scripts/datasets/wan_synth/precompute_phase1_anchors.py`)  
- **Quick pipeline smoke test**: `scripts/run_wansynth_pipeline_debug.sh` with `WAN_ATTN=sagesla`, `SLA_TOPK=0.07`, 2 steps each stage, `BATCH=1` completed end‑to‑end (no NaNs).  
- **Minor perf cleanup**: made `t_grid` contiguous in `_distance_alpha` to avoid `torch.searchsorted` non‑contiguous warning and extra copies. (`src/corruptions/video_keyframes.py`)  

### 2026-02-03 (PM) — SLA Throughput Calibration (topk=0.07, no LoRA)
- **Phase‑1 (Wan2.1 + SageSLA, B=2, steps=10)**  
  - Step time ≈ **2.6–2.8s** (first step ~3.1s)  
  - Throughput ≈ **0.7–0.8 samples/s**, **~15–16 frames/s** (T=21)  
- **Phase‑2 (Wan2.1 + SageSLA, B=2, steps=10)**  
  - Step time ≈ **2.6–2.7s** (first step ~3.2s)  
  - Throughput ≈ **0.7–0.8 samples/s**, **~15–16 frames/s**  
- **Anchor precompute (DDIM‑4, B=2, 2 batches)**  
  - Warmup batch ~9.7s; steady batch ~**5–6s**  
  - Rough throughput ≈ **0.35–0.4 samples/s** (≈7–8 fps) after warm‑up  

### 2026-02-03 (PM) — Longer SLA Calibration (stable curves)
- **Phase‑1 (Wan2.1 + SageSLA, B=2, steps=200, log_every=10)**  
  - Avg `step_time_sec` ≈ **2.67s**, avg `samples_per_sec` ≈ **0.75**  
  - TensorBoard: `runs/keypoints_wansynth_calib_sla_long`  
- **Phase‑2 (Wan2.1 + SageSLA, B=2, steps=200, log_every=10)**  
  - Avg `step_time_sec` ≈ **2.64s**, avg `samples_per_sec` ≈ **0.76**  
  - TensorBoard: `runs/interp_levels_wansynth_calib_sla_long`  
- **Anchor precompute (DDIM‑4, B=2, 20 batches)**  
  - Warmup ~24s for first batch; steady state ≈ **2.38s/batch**  
  - Throughput ≈ **0.84 samples/s**  

### 2026-02-03 (PM) — Video D_phi/Selector Pipeline Order
- **Decision**: D_phi targets will be computed on a **subsample** of the dataset (aggressive subsampling ok).  
- **Order**: train **interpolator** → compute **teacher/student KL** (student = keyframes + interpolation) → train **D_phi** → train **selector**.  

### 2026-02-03 (PM) — Interpolator Long Run (Wan synth)
- **Run**: `src/train/train_flow_interpolator_wansynth.py`  
  - `steps=10000`, `batch=8`, `model_dtype=bf16`, `T=21`  
  - Data: `data/wan_synth/.../shard-*.tar`  
  - Logs: `runs/latent_flow_interp_wansynth_long`  
  - Ckpt: `checkpoints/latent_flow_interp_wansynth_long/ckpt_final.pt`  
  - Wall‑time: **~5.4 min** (10k steps)  
  - Approx throughput: **~30 steps/s** → **~240 samples/s** (B=8)  
- Final loss: **~0.235**  

### 2026-02-03 (PM) — Interpolator Eval (latent L1 vs LERP)
- Script: `scripts/eval_flow_interpolator_wansynth.py`  
- Data: `data/wan_synth/.../shard-0000*.tar`, `num_batches=200`, `B=8` (1600 samples)  
- Checkpoint: `checkpoints/latent_flow_interp_wansynth_long/ckpt_final.pt`  
- Results (latent L1):  
  - Flow interp: **0.2432**  
  - LERP baseline: **0.2412**  
  - Relative improvement: **-0.8%** (LERP slightly better on pure L1)  
  - Uncertainty correlation (Pearson): **0.696**  
  - Low‑unc error: **0.1387** vs High‑unc error: **0.4102**  
  - Gap buckets (flow vs lerp):  
    - 2–3: **0.1738 vs 0.1719**  
    - 4–6: **0.2119 vs 0.2080**  
    - 7–10: **0.2539 vs 0.2500**  
    - 11–20: **0.3125 vs 0.3125**  

### 2026-02-03 (PM) — Interpolator Train/Val Split + Residual Refiner
- **Change**: added residual refiner + time conditioning in `LatentFlowInterpolator`; training script now supports `--train_pattern`, `--val_pattern`, and periodic eval.  
- **Train**: `src/train/train_flow_interpolator_wansynth.py`  
  - Train shards: `shard-0000[0-7].tar`  
  - Val shards: `shard-0000[8-9].tar`  
  - `steps=10000`, `batch=8`, `model_dtype=bf16`  
  - `residual_blocks=2`, `residual_channels=32`  
  - Logs: `runs/latent_flow_interp_wansynth_residual`  
  - Ckpt: `checkpoints/latent_flow_interp_wansynth_residual/ckpt_final.pt`  
- **Eval (val)**: `scripts/eval_flow_interpolator_wansynth.py`  
  - `num_batches=200`, `B=8` (1600 samples)  
  - Flow L1: **0.243164**  
  - LERP L1: **0.242188**  
  - Relative improvement: **-0.4%** (LERP still slightly better on L1)  
  - Uncertainty correlation (Pearson): **0.780**  
  - Low‑unc error: **0.1206** vs High‑unc error: **0.4063**  
  - Gap buckets (flow vs lerp):  
    - 2–3: **0.1738 vs 0.1748**  
    - 4–6: **0.2188 vs 0.2188**  
    - 7–10: **0.2500 vs 0.2490**  
    - 11–20: **0.3086 vs 0.3066**  

### 2026-02-03 (PM) — Interpolator Loss/Capacity Ablations (Wan synth)
- **Code changes**:  
  - Added `predict_flow` / `blend_from_flow` to reuse flow in training.  
  - Added loss knobs: endpoint consistency, flow consistency, gap‑weighted L1.  
  - New CLI: `--endpoint_weight`, `--flow_consistency_weight`, `--gap_weight`, `--gap_gamma`.  

- **Run A: capacity only (bigger net)**  
  - `base_channels=64`, `residual_blocks=4`, `residual_channels=64`  
  - Train shards: `shard-0000[0-7].tar`  
  - Val shards: `shard-0000[8-9].tar`  
  - Steps: 10k, B=8, bf16  
  - Ckpt: `checkpoints/latent_flow_interp_wansynth_residual_cap64/ckpt_final.pt`  
  - Eval (val, 1600 samples):  
    - Flow L1: **0.2891**  
    - LERP L1: **0.2432**  
    - Relative improve: **-18.9%** (worse than LERP)  
    - Unc corr: **0.822**  
    - Gap buckets (flow vs lerp): 2–3 **0.2461 vs 0.1787**, 4–6 **0.2734 vs 0.2100**, 7–10 **0.2988 vs 0.2617**, 11–20 **0.3281 vs 0.3008**  

- **Run B: capacity + extra losses**  
  - Same capacity as Run A  
  - Added `endpoint_weight=0.1`, `flow_consistency_weight=0.05`, `gap_weight=1.0`, `gap_gamma=1.0`  
  - Ckpt: `checkpoints/latent_flow_interp_wansynth_residual_v2/ckpt_final.pt`  
  - Eval (val, 1600 samples):  
    - Flow L1: **0.2871**  
    - LERP L1: **0.2471**  
    - Relative improve: **-16.2%** (still worse than LERP)  
    - Unc corr: **0.811**  
    - Gap buckets (flow vs lerp): 2–3 **0.2461 vs 0.1787**, 4–6 **0.2734 vs 0.2246**, 7–10 **0.2910 vs 0.2559**, 11–20 **0.3262 vs 0.3066**  

- **Takeaway**: Larger capacity and the new loss terms **degraded** latent L1 vs LERP. The smaller model (base_channels=32) remains closest to LERP.  

### 2026-02-03 (PM) — Interpolator Time‑Mask + Gap Conditioning Ablations
- **Change**: added **time‑dependent mask** and optional **gap conditioning** to latent flow interpolator.  
  - `mask = sigmoid(mask_a + mask_b * (2*alpha - 1))`  
  - `gap_norm` can be concatenated as an extra input channel (and fed to residual refiner).  
- **Run A (time_mask=1, gap_cond=0, base=32)**  
  - Ckpt: `checkpoints/latent_flow_interp_wansynth_timemask/ckpt_final.pt`  
  - Eval (val, 1600 samples):  
    - Flow L1: **0.2295**  
    - LERP L1: **0.2422**  
    - Relative improve: **+5.2%**  
    - Unc corr: **0.812**  
- **Run B (time_mask=1, gap_cond=1)**  
  - Ckpt: `checkpoints/latent_flow_interp_wansynth_timegap/ckpt_final.pt`  
  - Eval: Flow L1 **0.2314** vs LERP **0.2432** (≈ **+4.8%**)  
- **Run C (time_mask=1, gap_cond=1, gap_weight=0.5)**  
  - Ckpt: `checkpoints/latent_flow_interp_wansynth_timegap_gw05/ckpt_final.pt`  
  - Eval: Flow L1 **0.2344** vs LERP **0.2461** (≈ **+4.8%**)  
- **Run D (time_mask=1, base=48)**  
  - Ckpt: `checkpoints/latent_flow_interp_wansynth_timemask_c48/ckpt_final.pt`  
  - Eval: Flow L1 **0.2246** vs LERP **0.2432** (≈ **+7.6%**)  
  - Unc corr: **0.820**  
  - Gap buckets (flow vs lerp): 2–3 **0.1621 vs 0.1670**, 4–6 **0.2002 vs 0.2148**, 7–10 **0.2373 vs 0.2578**, 11–20 **0.2813 vs 0.3086**  
- **Run E (time_mask=1, residual_blocks=0)**  
  - Ckpt: `checkpoints/latent_flow_interp_wansynth_timemask_r0/ckpt_final.pt`  
  - Eval: Flow L1 **0.2334** vs LERP **0.2402** (≈ **+2.8%**)  
- **Run F (time_mask=1, residual_blocks=4)**  
  - Ckpt: `checkpoints/latent_flow_interp_wansynth_timemask_rb4/ckpt_final.pt`  
  - Eval: Flow L1 **0.2314** vs LERP **0.2441** (≈ **+5.2%**)  
- **Takeaway**: time‑dependent mask **consistently beats LERP**; **gap conditioning did not help**. Best so far is **base_channels=48** with time‑mask only.  

### 2026-02-03 (PM) — Interpolator RGB Decode Eval (Wan VAE)
- **Script**: `scripts/eval_flow_interpolator_wansynth_rgb.py`  
- **Data**: val shards `shard-0000[8-9].tar`, `num_batches=50`, `B=4` (200 samples)  
- **Checkpoint**: `checkpoints/latent_flow_interp_wansynth_timemask_c48/ckpt_final.pt`  
- **VAE**: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (subfolder `vae`)  
- **Results (RGB)**:  
  - PSNR (flow): **20.1254 dB**  
  - PSNR (lerp): **20.5339 dB**  
  - SSIM (flow): **0.77772**  
  - SSIM (lerp): **0.77393**  
  - **Δ**: PSNR **-0.41 dB** (flow worse), SSIM **+0.0038** (flow slightly better)  
- **Note**: latent L1 improved vs LERP, but decoded PSNR is slightly worse on this subset; SSIM slightly better.  

### 2026-02-03 (PM) — Interpolator Cost‑Volume + Structural Losses (Wan synth)
- **Change**: added **low‑res cost volume** input + **structure‑biased losses**:  
  - Cost volume: downsample 2×, radius=2, normalized dot‑product (channels=(2r+1)^2).  
  - Losses: gradient/edge L1 + multi‑scale latent L1 (scales 2,4).  
- **Train**: `src/train/train_flow_interpolator_wansynth.py`  
  - Train shards: `shard-0000[0-7].tar`  
  - Val shards: `shard-0000[8-9].tar`  
  - `steps=10000`, `batch=8`, `bf16`  
  - `time_mask=1`, `base_channels=48`, `residual_blocks=2`  
  - `cost_volume=1`, `cv_radius=2`, `cv_downscale=2`, `cv_norm=1`  
  - `edge_weight=0.1`, `ms_weight=0.1`, `ms_scales=2,4`  
  - Logs: `runs/latent_flow_interp_wansynth_c48_cv`  
  - Ckpt: `checkpoints/latent_flow_interp_wansynth_c48_cv/ckpt_final.pt`  
- **Eval (val, latent L1)**:  
  - Flow L1: **0.2227**  
  - LERP L1: **0.2441**  
  - Relative improve: **+8.8%** (best so far)  
  - Gap buckets (flow vs lerp): 2–3 **0.1729 vs 0.1787**, 4–6 **0.1943 vs 0.2119**, 7–10 **0.2393 vs 0.2637**, 11–20 **0.2695 vs 0.3008**  
- **Eval (val, RGB decode)**:  
  - PSNR (flow): **19.9715 dB**  
  - PSNR (lerp): **20.2620 dB**  
  - SSIM (flow): **0.77283**  
  - SSIM (lerp): **0.76688**  
  - **Δ**: PSNR **-0.29 dB**, SSIM **+0.0060**  
- **Takeaway**: latent structure improved; decoded SSIM improves slightly but PSNR still trails LERP.  

### 2026-02-03 (PM) — LDMVFI Teacher Distillation (Planned)
- **Decision**: Distill the cheap latent flow interpolator from **LDMVFI** (latent diffusion VFI) using **teacher‑generated mid‑frames**; avoid FP4 training.  
- **Teacher repo** cloned at `tmp_repos/LDMVFI` (MIT license; pretrained weights available per README).  
- **New code**:
  - `src/teachers/ldmvfi_teacher.py`: wrapper to load LDMVFI, run DDIM sampling, and decode interpolated RGB frames.  
  - `scripts/datasets/wan_synth/precompute_ldmvfi_teacher.py`: precompute LDMVFI teacher outputs, encode with Wan VAE, and store `teacher.pt` + `teacher_idx.pt` in WebDataset shards.  
  - `src/data/wan_synth.py`: added `create_wan_synth_teacher_dataloader` to join base data with teacher shards by key.  
  - `src/train/train_flow_interpolator_wansynth.py`: added `--teacher_pattern`, `--teacher_weight`, `--gt_weight`; training uses teacher latents when provided.  
- **Open**: install LDMVFI deps (omegaconf, pytorch‑lightning, timm, taming‑transformers, clip) and download pretrained ckpt to actually run teacher precompute.  

### 2026-02-03 (PM) — LDMVFI Teacher Distillation (Setup + Debug)
- **Venv**: created `/.venv_ldmvfi` (separate env for LDMVFI) with deps: `omegaconf`, `pytorch-lightning==2.2.4`, `cupy-cuda12x`, `gdown` (+ taming‑transformers from git).  
- **Checkpoint**: downloaded LDMVFI pretrained ckpt to `checkpoints/ldmvfi/ldmvfi-vqflow-f32-c256-concat_max.ckpt` (≈3.4GB).  
- **Patches**:
  - `src/teachers/ldmvfi_teacher.py`:  
    - Added Lightning `rank_zero_only` compatibility shim.  
    - Added fallback to locate `taming-transformers` in venv.  
    - Skip `ema_scope()` on assertion failure.  
  - `scripts/datasets/wan_synth/precompute_ldmvfi_teacher.py`:  
    - Fixed even-gap sampling (avoid invalid randint).  
    - Write WebDataset fields as `teacher.pth` + `teacher_idx.pth`.  
  - `src/data/wan_synth.py`:  
    - Teacher dataloader now maps `teacher.pth` / `teacher_idx.pth`.  
  - `src/train/train_flow_interpolator_wansynth.py`:  
    - Teacher dataloader uses `allow_missing=True` to avoid join buffer overflow on partial teacher shards.  
- **Debug precompute**:  
  - Ran `scripts/datasets/wan_synth/precompute_ldmvfi_teacher.py` with `ddim_steps=20`, `batch=1`, `max_batches=2`, `max_hw=256`.  
  - Output: `data/wan_synth_ldmvfi_teacher_debug2/teacher-000000.tar`.  
- **Debug train**:  
  - `train_flow_interpolator_wansynth.py` with `--teacher_pattern data/wan_synth_ldmvfi_teacher_debug2/teacher-*.tar` ran 5 steps successfully.  

### 2026-02-03 (PM) — LDMVFI Teacher Distillation (Full Precompute Settings)
- **Confirmed settings**: `ddim_steps=30`, `max_hw=256` for LDMVFI teacher precompute on Wan2.1 synth data.  
- **Plan**: run full teacher precompute with these settings, then train the flow interpolator with `--teacher_pattern` against the teacher latents.  

### 2026-02-03 (PM) — LDMVFI Precompute Throughput Calibration + Storage Bug
- **Calibration** (50 batches on shard-0000*, `ddim_steps=30`, `max_hw=256`, `num_workers=8`):  
  - **B=8**: ~**1.84 s/it** → **~4.35 samples/s**  
  - **B=16**: ~**2.59 s/it** → **~6.18 samples/s**  
  - **B=32**: ~**4.02 s/it** → **~7.96 samples/s** (best)  
  - **ETA @ 250k samples**: ~**8.7 hours** at B=32 (linear scaling).  
- **Found bug**: each `teacher.pth` in calibration shards was ~6.3MB because `zt[i]` was a view into the batch storage; `torch.save` serialized the **entire batch** for every sample.  
- **Fix**: clone per-sample tensors before write (`zt[i].contiguous().clone()`, `t_info[i].contiguous().clone()`) in `scripts/datasets/wan_synth/precompute_ldmvfi_teacher.py`.  
- **Action**: re-run full teacher precompute after the fix (and optionally delete large calibration shards).  

### 2026-02-03 (PM) — LDMVFI Teacher Precompute (5k staged)
- **Started** staged teacher precompute to avoid full 250k upfront.  
- Settings: `ddim_steps=30`, `max_hw=256`, `batch=32`, `num_workers=8`, `max_batches=157` (~5,024 samples).  
- Output: `data/wan_synth_ldmvfi_teacher_5k_b32`, log: `logs/ldmvfi_teacher_5k_b32.log`.  

### 2026-02-03 (PM) — Training Curves as PNGs
- Added `scripts/plot_tensorboard_scalars.py` to export TensorBoard scalars into PNGs (default output: `<log_dir>/plots/`).  

### 2026-02-03 (PM) — Flow Interpolator Training (LDMVFI 5k)
- **Run started**: LDMVFI‑distilled flow interpolator on 5k teacher set.  
- Train cmd: `train_flow_interpolator_wansynth.py`  
- Settings: `steps=5000`, `batch=64`, `model_dtype=bf16`, `time_mask=1`, `gap_cond=1`, `cost_volume=1`, `cv_radius=2`, `cv_downscale=2`.  
- Data: `train_pattern=shard-000[0-1]*.tar` (first ~20 shards), `val_pattern=shard-0096*.tar` (held‑out).  
- Outputs: `runs/latent_flow_interp_ldmvfi_5k_b32`, `checkpoints/latent_flow_interp_ldmvfi_5k_b32`.  
- Post‑run: auto‑export PNG curves + latent & RGB eval logs.  

### 2026-02-03 (PM) — LDMVFI 5k Run (Results)
- **Teacher dataset**: regenerated as **self‑contained** shards (includes `latent.pth`, `embed.pth`, `prompt.txt`, `teacher.pth`, `teacher_idx.pth`) to avoid key‑join issues.  
- Size: **~41 GB** for 5k samples.  
- **Training** (GPU): 5,000 steps completed successfully.  
- **Eval (val, latent L1 vs GT)**:  
  - `mean L1 (flow)` **0.5078**  
  - `mean L1 (lerp)` **0.2334**  
  - **Result**: flow is **much worse** than lerp on GT.  
- **Diagnosis**: model **fits teacher** but teacher is **far from GT**.  
  - L1 vs teacher ≈ **0.167**  
  - L1 vs GT ≈ **0.535**  
  → LDMVFI teacher outputs are **not aligned** with Wan synth GT latents.  
- **Implication**: LDMVFI is not a good teacher for Wan2.1 synth latents (domain/scale mismatch). Need a different teacher or distillation target.  

### 2026-02-03 (PM) — Cleanup + Pivot Discussion
- **Deleted** the 5k self‑contained teacher dataset (`data/wan_synth_ldmvfi_teacher_5k_b32`, ~41 GB).  
- **Consensus**: latent interpolation likely suffers from **geometry mismatch** (latent space not Euclidean), per discussion of non‑Euclidean shape space / geodesics in the ML thread.  
- **Potential next directions**:  
  - Train a **cheap bridge residual** (LERP + tiny residual) directly on **GT mid‑latents**, optionally with curvature/acceleration penalties.  
  - Consider **latent straightening** (learn an invertible linear map so LERP approximates geodesics).  

### 2026-02-03 (PM) — Teacher Key Order Issue (Fix)
- **Issue**: training failed immediately (`StopIteration`) because teacher shards were generated with `num_workers=8`, causing key order to diverge from base dataset.  
- **Symptom**: `_KeyJoinIterableDataset` overflowed buffer (`max_buffer=2000`) when trying to match keys (teacher keys far out of order).  
- **Fix**: re‑precompute the 5k teacher set with **`num_workers=0`** to preserve sequential order (deleted old unordered shards).  
- **New run**: `logs/ldmvfi_teacher_5k_b32_ordered.log`, output `data/wan_synth_ldmvfi_teacher_5k_b32`.  

### 2026-02-03 (PM) — Shard Ordering Bug (Root Cause)
- **Root cause**: `glob.glob()` returned shards in filesystem order (not sorted), so dataloaders started at ~`shard-0096x` (keys ≈ 247k).  
- **Symptom**: teacher and base streams were aligned but started near the end, causing key‑join mismatch with any expectation of low‑index keys.  
- **Fix**: sort shard lists in `create_wan_synth_dataloader`, `create_wan_synth_anchor_dataloader`, and `create_wan_synth_teacher_dataloader`.  

## 2026-02-06

### Sinkhorn Warper — PhaseCorr Robustness + Outlier Diagnostics
- **Context**: sinkhorn correspondence interpolator showed heavy-tail failures (some samples much worse than LERP), likely from bad global alignment and then applying large warps despite low confidence.
- **Code**:
  - `src/models/sinkhorn_warp.py`:
    - `phasecorr_mode=multi`: multi-channel phase correlation (sum cross-power across channels) instead of mean score-map.
    - `phasecorr_level=latent`: compute global (rotation + shift) via phasecorr on full latent maps `[C=16,H=60,W=104]`, then convert pixel shift to token units using `align_corners` scaling.
  - `scripts/diagnose_sinkhorn_outliers_wansynth.py`: outlier scanner (rank by `sinkhorn_mse - lerp_mse`) and dump summary stats.
- **Commits**:
  - `f5e45f5`: multi-channel phasecorr + outlier diagnostic.
  - `bf97233`: latent-level phasecorr option (global shift on 60x104 maps).
  - `7d39ff1`: diagnostic option `--scale_flow_by_conf` (see below).

### Sinkhorn Pipeline Training (Freeze Straightener, Train Matcher Only)
- **Script**: `src/train/train_sinkhorn_interp_wansynth.py`
- **Run**: `runs/sinkhorn_interp_optD_pcmulti_latent_w5_s3_r1_d64_b128_2k_warpz`
- **Ckpt**: `checkpoints/sinkhorn_interp_optD_pcmulti_latent_w5_s3_r1_d64_b128_2k_warpz/ckpt_final.pt`
- **Settings**:
  - Init straightener: `checkpoints/latent_straightener_optD_h448_b5_10k/ckpt_final.pt`
  - `freeze_straightener=1` (only trains matcher: token projection + tau + dustbin)
  - Global: `sinkhorn_global_mode=phasecorr`, `sinkhorn_phasecorr_mode=multi`, `sinkhorn_phasecorr_level=latent`
  - Local: `win=5`, `stride=3`, `d_match=64`, `proj_mode=linear`, `learn_tau=1`, `learn_dustbin=1`
  - Priors: `spatial_gamma=0.5`, `spatial_radius=1`, `fb_sigma=0.5`
  - `warp_space=z`
- **Val @ step=1500**:
  - `sinkhorn_mse=0.164625`
  - `lerp_mse=0.165021`
  - `straight_lerp_mse=0.154459` (still much better; straightener is doing most of the work here)

### Outlier Diagnosis (Why Tail Failures Persist)
- **Scan (val shards, 30 batches x 64 = 1920 samples)**, ckpt above, `phasecorr_mode=multi`, `phasecorr_level=latent`:
  - Original behavior had **catastrophic tail**: `max(delta_vs_lerp) ~ +0.76` and `delta>0.5` count `6/1920`.
- **Root cause**: when both confidences are near-zero (`denom ~ 0`), the code fell back to a **linear blend of warped endpoints**, which is unstable when the estimated flow is invalid.
- **Fix (implemented)**: fall back to **unwarped LERP** when `denom` is tiny (no reliable correspondences).
  - Commit: `c56de0d` ("avoid warping when confidence is zero") in:
    - `src/models/sinkhorn_warp.py`
    - `src/train/train_sinkhorn_interp_wansynth.py`
    - `scripts/diagnose_sinkhorn_outliers_wansynth.py`
  - After fix on the same 1920-sample scan:
    - `mean sinkhorn_mse` improved from `~0.1712` to `~0.1662` (LERP mean `~0.1704`)
    - `delta>0.5` tail removed: `0/1920`, `max(delta_vs_lerp) ~ +0.38`
- **Extra diagnostic (optional)**: `--scale_flow_by_conf 1` scales flow by confidence before warping; this also removes the `delta>0.5` tail but needs care during training to avoid trivial `conf -> 0` collapse.

### Sinkhorn Pipeline Training (End-to-End, Warp In Straightened Space)
- **Goal**: train the *full* “phasecorr (rot+shift) + local Sinkhorn + dustbin” interpolator **through** the straightener, warping/blending in `s` and decoding back to `z`.
- **Script**: `src/train/train_sinkhorn_interp_wansynth.py`
- **Run**: `runs/sinkhorn_interp_optD_pcmulti_latent_w5_s3_r1_d64_b64_10k_warps`
- **Ckpt**: `checkpoints/sinkhorn_interp_optD_pcmulti_latent_w5_s3_r1_d64_b64_10k_warps/ckpt_final.pt`
- **Settings**:
  - Init straightener: `checkpoints/latent_straightener_optD_h448_b5_10k/ckpt_final.pt`
  - `freeze_straightener=0` (train straightener + matcher)
  - Global: `sinkhorn_global_mode=phasecorr`, `sinkhorn_phasecorr_mode=multi`, `sinkhorn_phasecorr_level=latent`
  - Local: `win=5`, `stride=3`, `d_match=64`, `proj_mode=linear`, `learn_tau=1`, `learn_dustbin=1`
  - Priors: `spatial_gamma=0.5`, `spatial_radius=1`, `fb_sigma=0.5`
  - `warp_space=s`, `min_gap=2`, `steps=10000`, `batch=64`, `num_workers=8`
- **Val during training** (random triplets; MSE in `z`):
  - Step 1000: `sinkhorn_mse=0.156593`, `lerp_mse=0.167893`, `straight_lerp_mse=0.151825`
  - Step 9000: `sinkhorn_mse=0.154611`, `lerp_mse=0.167893`, `straight_lerp_mse=0.150652`
  - Observation: Sinkhorn consistently beats raw `z`-LERP, but is not clearly better than “straightened LERP” (`decode((1-a)S(z0)+aS(z1))`).

### Sinkhorn Diagnostics (ckpt_final, 30x64=1920 samples)
- **Script**: `scripts/diagnose_sinkhorn_outliers_wansynth.py` (run with `PYTHONPATH=.`).
- **Dataset**: val shards `shard-0000[8-9].tar`, `warp_space=s`, `min_gap=2`.
- **Summary stats** from `tmp_sinkhorn_outliers_ckpt_final_warps/cases.jsonl`:
  - `sinkhorn_mse`: mean **0.15116**, median **0.08791**
  - `lerp_mse`: mean **0.17045**, median **0.10426**
  - `straight_lerp_mse`: mean **0.15295**, median **0.09335**
  - `sinkhorn - lerp` (delta): mean **-0.01929**, p99 **+0.0870**, max **+0.5979**
  - Gap trend: mean `sinkhorn - lerp` becomes more negative as gap increases (e.g. gap>=10 mean delta **-0.0282**), but there remains a small tail of failures.

### Straightener (New Option): Token-Grid Transformer
- Added a new straightener architecture: `LatentStraightenerTokenTransformer` (patchify to 15x26=390 tokens; transformer over tokens; unpatchify back).
- **Code**:
  - `src/models/latent_straightener.py`: new `token_transformer` arch + checkpoint loader dispatch via `meta['arch']`.
  - `src/train/train_latent_straightener_wansynth.py`: add `--arch token_transformer` and transformer hyperparams.
  - `src/train/train_sinkhorn_interp_wansynth.py`: training-time loader/meta updated to support transformer straighteners.
  - `scripts/diagnose_sinkhorn_outliers_wansynth.py`: backward-compatible matcher loading for older checkpoints.
- **Commit**: `63c3a4d` ("straightener: add token-transformer arch").
