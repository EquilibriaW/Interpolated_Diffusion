#!/usr/bin/env bash
set -euo pipefail

CMD="python -u -m src.train.train_latent_straightener_wansynth \
  --data_pattern 'data/wan_synth/Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K/shard-0000*.tar' \
  --train_pattern 'data/wan_synth/Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K/shard-0000[0-7].tar' \
  --val_pattern 'data/wan_synth/Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K/shard-0000[8-9].tar' \
  --steps 10000 \
  --batch 256 \
  --hidden_channels 448 \
  --blocks 5 \
  --log_dir runs/latent_straightener_optD_h448_b5_10k \
  --ckpt_dir checkpoints/latent_straightener_optD_h448_b5_10k \
  --eval_every 1000 \
  --val_batches 50 \
  --save_every 1000 \
  --prefetch_to_gpu 1 \
  --prefetch_factor 2 \
  --num_workers 4 \
  --persistent_workers 1 \
  --resampled 1 \
  |& tee logs/latent_straightener_optD_h448_b5_10k.log"

FALLBACK_CMD="python -u -m src.train.train_latent_straightener_wansynth \
  --data_pattern 'data/wan_synth/Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K/shard-0000*.tar' \
  --train_pattern 'data/wan_synth/Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K/shard-0000[0-7].tar' \
  --val_pattern 'data/wan_synth/Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K/shard-0000[8-9].tar' \
  --steps 10000 \
  --batch 256 \
  --hidden_channels 448 \
  --blocks 5 \
  --log_dir runs/latent_straightener_optD_h448_b5_10k \
  --ckpt_dir checkpoints/latent_straightener_optD_h448_b5_10k \
  --eval_every 1000 \
  --val_batches 50 \
  --save_every 1000 \
  --prefetch_to_gpu 0 \
  --prefetch_factor 2 \
  --num_workers 0 \
  --persistent_workers 0 \
  --resampled 1 \
  |& tee logs/latent_straightener_optD_h448_b5_10k.log"

python -u scripts/watchdog_train.py \
  --tmux-session straightener_h448_b5_10k \
  --log-path logs/latent_straightener_optD_h448_b5_10k.log \
  --ckpt-dir checkpoints/latent_straightener_optD_h448_b5_10k \
  --cmd "$CMD" \
  --fallback-cmd "$FALLBACK_CMD" \
  --stall-minutes 10 \
  --check-every 60 \
  --ram-max-frac 0.90 \
  --use-fallback-after 1
