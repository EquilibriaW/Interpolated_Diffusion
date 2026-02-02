from __future__ import annotations

from typing import Tuple

import torch


def patchify_latents(latents: torch.Tensor, patch_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Patchify VAE latents into per-frame tokens.

    Args:
        latents: [B,T,C,H,W]
        patch_size: spatial patch size on latent grid

    Returns:
        tokens: [B,T,N,D]
        spatial_shape: (H_p, W_p)
    """
    if latents.dim() != 5:
        raise ValueError("latents must have shape [B,T,C,H,W]")
    B, T, C, H, W = latents.shape
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError("latent H/W must be divisible by patch_size")
    H_p = H // patch_size
    W_p = W // patch_size
    z = latents.view(B, T, C, H_p, patch_size, W_p, patch_size)
    z = z.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
    tokens = z.view(B, T, H_p * W_p, C * patch_size * patch_size)
    return tokens, (H_p, W_p)


def unpatchify_tokens(tokens: torch.Tensor, patch_size: int, spatial_shape: Tuple[int, int]) -> torch.Tensor:
    """Reconstruct latents from per-frame tokens.

    Args:
        tokens: [B,T,N,D]
        patch_size: spatial patch size on latent grid
        spatial_shape: (H_p, W_p)

    Returns:
        latents: [B,T,C,H,W]
    """
    if tokens.dim() != 4:
        raise ValueError("tokens must have shape [B,T,N,D]")
    B, T, N, D = tokens.shape
    H_p, W_p = spatial_shape
    if N != H_p * W_p:
        raise ValueError("spatial_shape does not match token count")
    if D % (patch_size * patch_size) != 0:
        raise ValueError("token dim must be divisible by patch_size**2")
    C = D // (patch_size * patch_size)
    z = tokens.view(B, T, H_p, W_p, C, patch_size, patch_size)
    z = z.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    return z.view(B, T, C, H_p * patch_size, W_p * patch_size)


__all__ = ["patchify_latents", "unpatchify_tokens"]
