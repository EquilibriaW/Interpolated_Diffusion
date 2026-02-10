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
