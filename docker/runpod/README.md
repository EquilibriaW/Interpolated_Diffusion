# RunPod Image (8xB200)

## Build and push

```bash
cd /workspace/Interpolated_Diffusion_phase2
docker build -f docker/runpod/Dockerfile -t <registry>/<repo>:phase2-8xb200 .
docker push <registry>/<repo>:phase2-8xb200
```

## Launch on RunPod

Set container image to `<registry>/<repo>:phase2-8xb200`.

Recommended env vars:

```bash
LAUNCH_SCRIPT=scripts/run_8xb200_drifting_latent.sh
DATA_ROOT=/runpod-volume/imagenet/train
LATENT_FEATURE_CKPT=/runpod-volume/checkpoints/latent_mae_imagenet_8xb200/ckpt_final.pt
```

For latent-MAE pretrain first:

```bash
LAUNCH_SCRIPT=scripts/run_8xb200_latent_mae.sh
DATA_ROOT=/runpod-volume/imagenet/train
```

## Notes

- Torch is installed from nightly CUDA 12.8 wheels in this image (`TORCH_INDEX_URL` build arg).
- Muon is installed from `https://github.com/KellerJordan/Muon.git`.
- If you want a stable Torch release instead of nightly, change `TORCH_INDEX_URL` in `docker/runpod/Dockerfile`.

