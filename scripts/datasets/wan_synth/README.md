# Wan2.1 synthetic dataset (TurboDiffusion)

TurboDiffusion provides **Wan2.1‑synthesized** latent datasets on HuggingFace:

- Repo: `worstcoder/Wan_datasets`
- Example dataset: `Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K`

Each sample is stored in webdataset `.tar` shards with:

- `*.latent.pt` → latents (C=16, T=21, H=60, W=104 for 480p 16:9)
- `*.embed.pt` → umT5 text embeddings
- `*.prompt.txt` → prompt string

## Download

```bash
python scripts/datasets/wan_synth/download_wan_synth.py \
  --dataset Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K \
  --out_dir data/wan_synth
```

To download only a few shards:

```bash
python scripts/datasets/wan_synth/download_wan_synth.py \
  --dataset Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K \
  --shard_pattern "shard-0000*.tar" \
  --out_dir data/wan_synth
```

## Loader

Use the WebDataset loader:

```python
from src.data.wan_synth import create_wan_synth_dataloader

loader = create_wan_synth_dataloader(
    "data/wan_synth/Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K/shard*.tar",
    batch_size=4,
)
```

Requires:
- `webdataset`
- `huggingface_hub`
