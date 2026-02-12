# Project Log Phase 2 (Video Refinement / Imputation)

This file tracks Phase 2 work done in parallel to Phase 1.

Phase 2 is a diffusion-style refiner whose forward process is **keyframes + interpolation (+ optional noise)** rather
than Gaussian noise. It can be used both:
- as part of the full pipeline: text -> phase-1 keyframes -> interpolator -> phase-2 refinement
- as a standalone module: storyboard keyframes (+ optional text prompt) -> full video (useful even if phase 1 is not ready)

Process: update after each experiment run and after major design discussions.
Include: date, command/script, dataset/checkpoints, key settings, and observed results.

## 2026-02-09

### Repo Fork Setup (Parallel Work)
- (Superseded) initial local worktree/clone setup for parallel work.

## 2026-02-10

### Dev Repo Setup (Parallel Work)
- Phase-2 development repo: `/workspace/Interpolated_Diffusion_phase2` (branch `phase2`, pushed to `origin/phase2`)
- Shared artifacts (symlinks to avoid copying multi-TB directories; write new outputs locally by default):
  - `data_shared` -> `/workspace/Interpolated_Diffusion/data`
  - `data/wan_synth` -> `../data_shared/wan_synth` (keep derived artifacts like anchors local under `data/`)
  - `checkpoints_shared` -> `/workspace/Interpolated_Diffusion/checkpoints`
  - `tmp_repos_shared` -> `/workspace/Interpolated_Diffusion/tmp_repos`

### Implementation Fixes
- `src/corruptions/video_keyframes.py`:
  - `build_video_token_interp_adjacent_batch` now accepts `sinkhorn_warper` (previously training crashed due to unexpected kwarg).
  - Fixed warper dtype fallback paths that referenced undefined variables when `warper.parameters()` is empty.
  - Commit: `220dee8`

### Throughput Reference (B200, Wan2.1 1.3B + SageSLA)
- From prior calibration runs (`runs/interp_levels_wansynth_calib_sla_long`):
  - Avg `step_time_sec` ≈ **2.64s** at `B=2`, `T=21` (≈ **0.76 samples/s**, **15–16 frames/s**)
  - Rough wall-time estimate: `steps * 2.64s` (e.g. **20k steps ≈ 14.7 hours**)

### Phase 2, Standalone Definition (Assume GT Keyframes)
Assume we are given ground-truth (or user-provided) keyframes (a storyboard) plus an interpolator.
- Forward/corruption process: pick a nested set of keyframe indices (levels). For a level `s`, reconstruct the full
  sequence by interpolating between the selected keyframes; optionally add small noise mainly on non-keyframes.
- Reverse/denoising process: a model predicts a correction from interpolated tokens toward the ground-truth tokens (or
  predicts the clean tokens directly).
- Optional scheduled-sampling hook: independently replace some keyframes with phase-1 outputs with probability `p` to
  make phase 2 robust to off-manifold keyframes. (Disabled for the standalone GT-keyframe/imputation setting.)

Key motivation vs phase 1: phase 2 is naturally an **imputation/inpainting** problem because keyframes are treated as
ground-truth constraints; phase-1 keyframes should not be treated as ground truth at inference, only as high-confidence
proposals.

### Existing Code Paths (as of commit `a975394`)
- Video (Wan2.1-synth):
  - Training: `src/train/train_interp_levels_wansynth.py`
  - Corruption (keyframes + interp + noise + scheduled sampling): `src/corruptions/video_keyframes.py`
  - Nested keyframe schedules/masks: `src/corruptions/keyframes.py`
  - Debug pipeline wrapper: `scripts/run_wansynth_pipeline_debug.sh`
- Toy video sanity:
  - Training: `src/train/train_interp_levels_toy_video.py`
- Maze/trajectory stage-2 analogue (not video, but same concept):
  - Training: `src/train/train_interp_levels.py`

### Immediate Phase-2-Standalone Experiments (Planned)
1. GT-keyframes only (no phase-1 anchors):
   - Use `train_interp_levels_wansynth.py` with no `--anchor_pattern` and `--student_replace_prob 0`.
   - Compare `corrupt_mode` (`none`, `dist`, `gauss`) and sigma settings.
2. Imputation with explicit "given keyframes":
   - Provide `anchor_idx`/`anchor_values` from GT keyframes (or storyboard images encoded to latents).
   - Ensure corruption never perturbs/clobbers the given keyframes; add noise only to missing frames.
3. Storyboard conditioning:
   - Condition phase 2 on keyframe content explicitly (beyond just treating them as fixed tokens).

### Open Questions / Design Notes
- Index nesting policy: fixed uniform schedules vs learned selector; for storyboard use, a fixed schedule may be acceptable.
- What should "noise level" mean: number of provided frames vs additive noise amplitude vs both.
- One-step distillation (rCM) for phase 2: once we have a reliable multi-step refiner, distill to 1 step for practical use.

## 2026-02-11

### Drifting Direction (Agreed Scope)
- We are prioritizing **one-step correction via drifting** for Phase 2.
- Refiner input/output for this line:
  - input: `z_s` from keyframes + interpolation corruption
  - output: `x_hat` one-step corrected full video tokens/latents
- Loss location:
  - drifting loss in **feature space** (`phi_j`) only
  - no paired reconstruction term in the first implementation (drifting-only objective)
- Conditioning:
  - use **discrete pseudo-classes** for conditional drifting behavior (class-aware positive/negative sampling)
  - pseudo-classes are obtained by quantizing a joint condition embedding from text + keyframes

### Representation Plan (First Pass)
- Feature encoder for drifting:
  - start with a **latent-space MAE encoder** trained on Wan-synth latents, then freeze it
  - extract multi-scale/multi-location features and sum drifting losses across feature vectors
- Quantized condition classes:
  - compute `h = g(text, keyframes)`
  - quantize `h` to class id `c` using a learned codebook (k-means/VQ)
  - use `c` for class-conditional sampling in drifting field computation

### Pseudo-Class Count (K) Planning
- Current dataset is the Wan-synth 250K set (`970` shards in local storage).
- First-shard estimate method (agreed):
  - `shard-00000.tar` contains `256` clips (`256 latent + 256 embed + 256 prompt files`).
  - total clips estimate: `256 * 970 = 248,320`.
- Initial single guess (no sweep for now):
  - use a higher-capacity class partition for semantic resolution: **`K=1024`** as the first run.
  - fallback only if occupancy/collapse diagnostics are poor: **`K=512`**.
- Keep monitoring:
  - non-empty classes
  - mean/median occupancy
  - occupancy p10/p90
  - fraction of singleton or near-empty classes
- If occupancy/collapse metrics indicate bottleneck, revise `K` afterward.

### Mandatory Representation Evaluation (for every quantizer/encoder experiment)
- Separation metrics:
  - intra-class distance vs inter-class distance (ratio)
  - silhouette score (on sampled subset)
  - Davies-Bouldin index
- Retrieval diagnostics:
  - nearest-neighbor retrieval in feature space
  - top-k retrieval consistency by pseudo-class
  - text/keyframe agreement checks on retrieved neighbors
- Collapse diagnostics:
  - class occupancy histogram
  - entropy of class usage
  - max-class dominance fraction
- Drifting-kernel diagnostics:
  - distribution of kernel logits/distances
  - effective neighbor count per sample
  - positive vs negative kernel mass ratio
- Visualization:
  - UMAP/t-SNE plots colored by pseudo-class and by prompt family
  - per-class exemplar grids (keyframes and decoded clips where feasible)

### Experiment Logging Contract (must fill before/after each run)
- Header:
  - date/time, experiment id, branch, commit hash, script/command
- Data:
  - data pattern, shard range, estimated sample count
- Model/loss config:
  - encoder checkpoint + freeze status
  - quantizer method and `K`
  - positive/negative sampling policy
  - feature sets used (`phi_j` definition), temperatures, normalization settings
- Runtime:
  - batch size, steps, throughput, GPU type, peak memory
- Results:
  - training curves
  - representation-separation metrics above
  - qualitative retrieval/visual diagnostics
  - decision: keep/drop candidate and why

### Image-Pivot Implementation (Keypatch Drifting, for low-cost validation)
- Rationale:
  - Before expensive video-scale runs, validate the core idea (key anchors + interpolation + one-step drifting refinement) in image space where we can compare more directly to ImageNet-style drifting settings.
- Added modules:
  - `src/utils/image_patches.py`
    - patchify/unpatchify image tokens
    - nested keypatch mask sampling
    - inverse-distance keypatch interpolation for missing patches
  - `src/models/image_patch_refiner.py`
    - class-conditioned + level-conditioned transformer refiner over patch tokens
  - `src/models/drifting_feature_encoder.py`
    - frozen ResNet feature extractor with multi-feature vector extraction (global stats, optional local/patch stats)
  - `src/losses/drifting.py`
    - paper-style kernelized drifting field (`compute_drift`)
    - feature/drift normalization and multi-temperature loss aggregation
    - drift diagnostics (distance/kernel mass/effective neighbors)
  - `src/train/train_drifting_image_keypatch.py`
    - class-balanced batch sampler
    - per-class positive/negative feature queues
    - keypatch corruption + one-step refinement + feature-space drifting loss
    - representation diagnostics logging (intra/inter ratio, silhouette/DBI if sklearn, queue occupancy/entropy/collapse stats)
  - `scripts/run_drifting_image_keypatch.sh`
    - default launcher for ImageFolder/ImageNet-style runs with configurable drift/feature settings
- Interpolation baseline update:
  - added **feature-space edge-aware harmonic/Laplacian interpolation** on the 2D keypatch graph (`interp_mode=laplacian`), with IDW init and tunable edge weighting (`sigma_feature`, `sigma_space`, `lambda_reg`).
  - interpolation diagnostics now logged: `interp/mse_missing`, `interp/mse_all`.
- Anchor behavior update:
  - anchor clamping is now optional in the image refiner path; default is **not** clamping anchors (`keep_anchors_fixed=0`), per current design decision.

### Quick B200 Profiling (sanity only; no real training run)
- Environment: single **NVIDIA B200**, fake dataset, image size 64.
- Command family: `python -m src.train.train_drifting_image_keypatch ... --dataset fake ...`
- Observed:
  - Light feature set (9 feature vectors; no patch stats): ~`0.25–0.7 s/step` at batch `32`.
  - Heavier multi-feature set (125 vectors with patch stats): ~`1.8–2.7 s/step` at batch `32`.
  - Laplacian interpolation mode shows similar steady-state cost to IDW in this setup (initial step dominates due warm-up/model init).
- Takeaway:
  - Image-space validation is much cheaper than video-space Wan runs and is suitable for rapid ablations on representation choices and drifting diagnostics.

## 2026-02-12

### Drift-Speed Refactor (Objective-Preserving)
- Goal:
  - keep full feature set and class conditioning unchanged
  - speed up only via implementation changes (no loss/objective change)
- Implemented in `src/losses/drifting.py`:
  - class-batched drifting loss path (`drifting_loss_single_feature_class_batched`)
  - single `cdist` reuse across all temperatures per feature
  - new stats-free fast kernel (`compute_drift_from_distances_nostats`) for non-log steps
  - existing per-class path retained for compatibility
- Implemented in `src/train/train_drifting_image_keypatch.py`:
  - class-ragged packing helper to batch classes per feature (`_pack_feature_class_batch`)
  - training loop now computes drifting loss per-feature with class batching
  - BF16 autocast toggle (`--amp_bf16`)
  - optional compiled fast path (`--compile_drift`, `--compile_mode`)
  - drift stats are computed only on logging steps
- Launcher update in `scripts/run_drifting_image_keypatch.sh`:
  - added env/args for `AMP_BF16`, `COMPILE_DRIFT`, `COMPILE_MODE`

### Validation / Benchmarks (B200, fake data)
- Smoke checks:
  - non-compiled training smoke run completed (`steps=5`)
  - compiled training smoke run completed (`steps=2`, `compile_mode=reduce-overhead`)
- Controlled benchmark (same synthetic setup, full 125-feature config):
  - old per-class drifting path: ~`3.08 s/step`
  - new class-batched drifting path (fp32): ~`0.50 s/step`
  - new class-batched drifting path (bf16): ~`0.50 s/step`
  - speedup vs old: about **6.1x** in this setup
- Drift-only micro-benchmark (loss stage + backward on feature tensors):
  - old per-class: ~`2.81 s`
  - new class-batched: ~`0.32 s`
  - speedup: about **8.8x** on drifting stage itself

### Notes on `torch.compile`
- Compiled fast path works in short training smoke runs.
- With highly dynamic shapes (class occupancy variation), compile may recompile often depending on mode.
- In our local synthetic benchmark, compile mode did not provide a stable speed win and could be much slower due recompilation overhead.
- Current recommendation:
  - keep `--compile_drift=0` by default for stability
  - enable compile only for longer fixed-shape runs and profile first

### Generative-Drifting Paper-Alignment Fixes (Implemented)
- Scope:
  - align training path with requested items:
    1. unconditional/CFG negative branch
    2. vanilla drifting branch (without `phi`)
    3. spatial normalization sharing for multi-location features
    4. sample queue without replacement
    5. index-aligned self-negative masking

- `src/train/train_drifting_image_keypatch.py`:
  - queue sampling policy changed to **without replacement** (`PerClassFeatureQueue.sample`).
  - added CFG controls:
    - `--cfg_alpha_values`
    - `--cfg_alpha_probs`
    - `--drift_n_unc`
    - `--queue_unc`
  - sampled per-class `alpha` each step; fed into refiner as conditioning.
  - added unconditional real-feature queue and weighted unconditional negatives (`w`) following paper CFG formulation.
  - `_pack_feature_class_batch` now:
    - prepends current generated features for class (index-aligned negatives),
    - appends queued generated negatives,
    - appends unconditional negatives,
    - returns per-negative weights and per-class `self_neg_k`.
  - added feature grouping by location prefix and used group drifting loss with shared normalization for location groups.
  - added optional vanilla no-`phi` branch (`vanilla_token_mean`) with `--vanilla_loss_weight`.

- `src/models/image_patch_refiner.py`:
  - added `alpha` conditioning path (`alpha_proj`) and forward arg `alpha`.

- `src/losses/drifting.py`:
  - `compute_drift_from_distances*` now support:
    - `y_neg_weights` (for unconditional negative weighting),
    - `self_neg_k` (index-aligned diagonal self masking).
  - `drifting_loss_single_feature_class_batched` now accepts weighted negatives and self-mask controls.
  - added `drifting_loss_feature_group_class_batched`:
    - shared feature normalization and drift normalization across grouped location features (paper-style policy),
    - per-feature losses still computed independently; only normalization scales are shared within group.

- `scripts/run_drifting_image_keypatch.sh`:
  - added env/arg wiring for CFG/unconditional queue/spatial sharing/vanilla branch.

### Validation (Smoke)
- Command family:
  - `python -m src.train.train_drifting_image_keypatch --dataset fake ... --steps 3 ...`
- Result:
  - end-to-end smoke run passed on B200 with:
    - CFG alpha conditioning
    - unconditional weighted negatives
    - group-shared normalization
    - vanilla branch enabled
  - no runtime errors after BF16 diagnostics fix in separation metrics (`feat.float()` before `cdist`).

### Quick Throughput Re-check (B200, fake data, 126 features)
- Command family:
  - `python -m src.train.train_drifting_image_keypatch --dataset fake --image_size 64 --steps 30 --n_classes_step 8 --n_gen_per_class 4 --feature_include_patch_stats 1 ...`
- Observed logged step times:
  - step 0 log: `1.748s` (warmup-heavy)
  - step 10 log: `0.847s`
  - step 20 log: `1.043s`
- Takeaway:
  - after warmup, this aligned implementation remains around ~`0.85-1.05 s/step` in this synthetic setup (single B200).

### Resolution Sensitivity Check (B200, fake data, 256x256 with patch_size=16)
- Command family:
  - `python -m src.train.train_drifting_image_keypatch --dataset fake --image_size 256 --patch_size 16 --steps 12 --n_classes_step 8 --n_gen_per_class 4 ...`
- Observed logged step times:
  - step 0 log: `3.097s` (warmup-heavy)
  - step 6 log: `1.900s`
- Observed feature count:
  - `feat=242` (more features than the `64x64` run due additional spatial feature vectors)
- Takeaway:
  - for same token count (`N=256`) but higher image resolution/features, steady-state is around ~`1.9s/step` on single B200 in this synthetic benchmark.

### 8xB200 Readiness: DDP + Muon + Launch Scripts
- Added optimizer factory with AdamW/Muon split:
  - `src/optim/factory.py`
  - Muon applies to matrix-like hidden weights; Adam branch handles norm/bias/embedding-like params.
  - Muon dependency: `pip install git+https://github.com/KellerJordan/Muon.git`
- Upgraded drifting trainer for multi-GPU effective-batch runs:
  - `src/train/train_drifting_image_keypatch.py`
  - new support:
    - DDP init via `torchrun`
    - `--grad_accum`
    - `--optimizer {adamw,muon}` + Muon hyperparameters
    - effective batch metadata/logging
    - latent input mode (`--input_space latent`) using SD-VAE encoder
    - latent feature-space mode (`--feature_space latent_mae`) via frozen latent-MAE checkpoint
- Added latent-MAE model and trainer:
  - `src/models/latent_mae.py`
  - `src/models/latent_mae_feature_encoder.py`
  - `src/train/train_latent_mae_imagenet.py`
  - trainer supports DDP + grad accumulation + AdamW/Muon.
- Added 8xB200 launch scripts:
  - `scripts/run_8xb200_latent_mae.sh`
  - `scripts/run_8xb200_drifting_latent.sh`
  - both scripts are configured for effective batch 4096 defaults.
- Added RunPod containerization files:
  - `docker/runpod/Dockerfile`
  - `docker/runpod/entrypoint.sh`
  - `docker/runpod/README.md`
  - image installs nightly Torch CUDA 12.8 wheels + Muon package.
