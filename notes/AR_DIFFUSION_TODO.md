# AR Diffusion (Future Work)

Goal: support variable-length trajectories for AR/diffusion-forcing training and inference.

Plan:
- Dataset prep: store full episodes + lengths (no windowing).
  - Save `episodes.npz` with:
    - `x`: concatenated episodes or list of arrays
    - `lengths`: per-episode lengths
    - `occ`, optional `sdf`
- Dataset loader: `PreparedEpisodeDataset`
  - Returns variable-length sequences.
  - Provide `collate_pad` to pad and return an attention mask.
- Full-seq training: keep fixed-T by slicing/resampling from episodes.
- AR training: use padded batches + causal mask, or chunked streaming with masks.
- Inference: allow variable horizon at sample time (length L), with mask-aware decoding.

Notes:
- Keep dataset construction separate from training logic.
- For fixed-length full-seq experiments, continue using prepared fixed-T datasets.

## Anchor/Interpolation Diffusion Notes (Planning → Video)

Goal: build a two-stage model where Stage-1 generates sparse anchors (keypoints/keyframes) and
Stage-2 refines an interpolated trajectory through multiple corruption levels, with inference
matching training distribution.

Ideas tried / current status:
- **Student vs teacher anchors mismatch**:
  - Inference only sees Stage-1 anchors; training used mixed GT anchors.
  - Fix: training forward process now replaces *coarsest* anchors with Stage-1 samples
    with probability `bootstrap_replace_prob` (default 0.5), keeping GT endpoints.
- **Confidence masks (soft anchors)**:
  - Add a confidence channel: teacher=0.95, student=0.5, endpoints=1.0, missing=0.0.
  - This is fed into Stage-2 as an extra input channel and used as loss weights.
  - Inference uses soft clamp with confidence (optional).
- **Annealed confidence (inference)**:
  - Fixed conf=0.5 keeps anchors noisy even at fine steps → diffuse outputs.
  - Added per-level anneal: `conf_s = conf + (1-conf) * (1 - s/levels)` (linear or cosine).
  - Keeps early flexibility and tightens toward the end.
- **Phase-2 corruption schedule**:
  - Added distance-to-anchor noise + anchor jitter (`--corrupt_mode dist`).
  - Noise scales by anchor density: `sigma_s = sigma_max * (K_min/K_s)^pow`.
  - Intended to keep successive levels close (Soft Diffusion principle).
- **Temporal index jitter (anchor value jitter)**:
  - New corruption: keep anchor indices fixed, but sample anchor values from nearby timesteps.
  - Implemented as `idx_j = idx + delta` (small delta), used only for gathering anchor values.
  - Masks remain nested; endpoints can stay exact.
  - CLI: `--corrupt_index_jitter_max`, `--corrupt_index_jitter_prob`, `--corrupt_index_jitter_pow`.

Video transfer intuition:
- Anchors become keyframes / sparse latent tokens.
- Confidence channel generalizes to "keyframe reliability."
- Annealed confidence should map to "lock in" keyframes at later refinement steps.

Open questions:
- Whether to replace *all* anchors vs only coarsest anchors with Stage-1 samples.
- How to learn per-anchor uncertainty from Stage-1 (variance prediction) rather than fixed 0.5.
- Best corruption schedule for video latents (may need smaller sigma and content-aware noise).

Recent observation:
- Stage-2 remains conservative (tracks Phase-1 interpolation too closely).
  Hypotheses:
  - Corruption between adjacent levels still too mild -> model learns small deltas only.
  - Soft clamp/confidence too strong -> keeps anchors/interp path fixed.
  Next tests:
  - Increase corruption (`sigma_max`, `anchor_frac`), reduce `anchor_conf_student`,
    reduce/disable soft clamp, lower `w_anchor`.
  - Try x0-pred mode or hybrid fine-tune (adj → x0) for larger corrections.
Update:
- Sampling with `--soft_anchor_clamp 1 --soft_clamp_max 0.3` noticeably improves recovery from bad interp,
  but still fails on some hard cases. Suggest testing 0.2 vs 0.4 and/or cosine confidence anneal.
