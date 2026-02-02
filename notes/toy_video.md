# Toy Video Experiments (Moving Shapes)

Goal: create a small, fully modular toy video dataset + corruption pipeline to test
keypoint diffusion + interpolation‑corruption denoising, in a setting closer to video.

Status:
- Dataset: `src/data/toy_video.py` (MovingShapesVideoDataset)
  - RGB frames (64x64), 1–3 moving shapes, bouncing on borders.
  - Latents are downsampled to 16x16 and flattened to D=768.
- Interpolation / corruption: `src/corruptions/video_keyframes.py`
  - Linear, smooth (temporal kernel), or learned interpolation.
  - Corruption supports time‑distance noise (`corrupt_mode="dist"`) and anchor noise.
  - Optional “student anchor” replacement (noisy teacher) for bootstrap.
- Training scripts are dataset‑specific:
  - `src/train/train_keypoints_toy_video.py`
  - `src/train/train_interp_levels_toy_video.py`
  - `src/train/train_video_interpolator.py`
- Sampling script: `src/sample/sample_toy_video.py` (compares oracle/pred + stage2)

Notes / ideas:
- Use soft anchors (confidence channel) to let stage‑2 correct keypoints.
- For video, consider stronger temporal corruption but keep steps small (soft diffusion principle).
- Learnable interpolator can be used as an ablation vs linear/smooth.
- Future: evaluate on held‑out seeds or different object dynamics to avoid trivial memorization.
