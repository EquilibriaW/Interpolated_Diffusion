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
