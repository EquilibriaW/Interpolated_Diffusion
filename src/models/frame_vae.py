from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

try:  # optional dependency
    from diffusers import AutoencoderKL

    _DIFFUSERS_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    AutoencoderKL = None
    _DIFFUSERS_AVAILABLE = False


class FrameAutoencoderKL(nn.Module):
    def __init__(
        self,
        model_name: str = "stabilityai/sd-vae-ft-mse",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
        scale: float = 0.18215,
        use_mean: bool = True,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        if not _DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers is required for FrameAutoencoderKL")
        self.vae = AutoencoderKL.from_pretrained(model_name, use_safetensors=True)
        self.scale = float(scale)
        self.use_mean = bool(use_mean)
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.vae.to(self.device, dtype=self.dtype)
        if freeze:
            for param in self.vae.parameters():
                param.requires_grad = False
            self.vae.eval()

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [B,T,3,H,W] or [B,3,H,W] in [0,1]
        orig_shape = frames.shape
        if frames.dim() == 5:
            b, t, c, h, w = orig_shape
            frames = frames.view(b * t, c, h, w)
        frames = frames.to(self.device, dtype=self.dtype)
        frames = frames * 2.0 - 1.0
        dist = self.vae.encode(frames).latent_dist
        latents = dist.mean if self.use_mean else dist.sample()
        latents = latents * self.scale
        if len(orig_shape) == 5:
            latents = latents.view(b, t, latents.shape[1], latents.shape[2], latents.shape[3])
        return latents

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        # latents: [B,T,4,H,W] or [B,4,H,W]
        orig_shape = latents.shape
        if latents.dim() == 5:
            b, t, c, h, w = orig_shape
            latents = latents.view(b * t, c, h, w)
        latents = latents.to(self.device, dtype=self.dtype) / self.scale
        frames = self.vae.decode(latents).sample
        frames = (frames + 1.0) * 0.5
        frames = frames.clamp(0.0, 1.0)
        if len(orig_shape) == 5:
            frames = frames.view(b, t, frames.shape[1], frames.shape[2], frames.shape[3])
        return frames


__all__ = ["FrameAutoencoderKL"]
