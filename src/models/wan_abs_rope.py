from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class WanRotaryPosEmbedAbsTime(nn.Module):
    """Wan rotary position embedding with optional absolute-time indices.

    Diffusers' `WanTransformer3DModel` calls `rotary_emb = self.rope(hidden_states)` with no arguments,
    so we stash optional per-sample frame indices on the module and use them when provided.

    `frame_indices` should be absolute frame indices in the *original* video coordinate system, with shape:
      - [B, num_frames] or [num_frames]
    """

    def __init__(self, base_rope: nn.Module) -> None:
        super().__init__()
        # We wrap the existing rope so we can fall back to its exact behavior.
        self.base_rope = base_rope
        self.frame_indices: Optional[torch.Tensor] = None

    @property
    def attention_head_dim(self) -> int:  # pragma: no cover - simple passthrough
        return int(getattr(self.base_rope, "attention_head_dim"))

    @property
    def patch_size(self) -> tuple[int, int, int]:  # pragma: no cover - simple passthrough
        return tuple(int(x) for x in getattr(self.base_rope, "patch_size"))

    @property
    def max_seq_len(self) -> int:  # pragma: no cover - simple passthrough
        return int(getattr(self.base_rope, "max_seq_len"))

    @property
    def t_dim(self) -> int:  # pragma: no cover - simple passthrough
        return int(getattr(self.base_rope, "t_dim"))

    @property
    def h_dim(self) -> int:  # pragma: no cover - simple passthrough
        return int(getattr(self.base_rope, "h_dim"))

    @property
    def w_dim(self) -> int:  # pragma: no cover - simple passthrough
        return int(getattr(self.base_rope, "w_dim"))

    @property
    def freqs_cos(self) -> torch.Tensor:  # pragma: no cover - simple passthrough
        return getattr(self.base_rope, "freqs_cos")

    @property
    def freqs_sin(self) -> torch.Tensor:  # pragma: no cover - simple passthrough
        return getattr(self.base_rope, "freqs_sin")

    def set_frame_indices(self, frame_indices: Optional[torch.Tensor]) -> None:
        if frame_indices is None:
            self.frame_indices = None
            return
        if not isinstance(frame_indices, torch.Tensor):
            raise TypeError("frame_indices must be a torch.Tensor or None")
        if frame_indices.dtype != torch.long:
            frame_indices = frame_indices.to(dtype=torch.long)
        self.frame_indices = frame_indices

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.frame_indices is None:
            return self.base_rope(hidden_states)

        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        if num_frames % p_t != 0 or height % p_h != 0 or width % p_w != 0:
            raise ValueError("hidden_states must be divisible by rope patch_size")

        ppf = num_frames // p_t
        pph = height // p_h
        ppw = width // p_w

        idx = self.frame_indices
        if idx.dim() == 1:
            idx = idx.view(1, -1)
        if idx.dim() != 2:
            raise ValueError("frame_indices must be [B,T] or [T]")
        if idx.shape[0] not in (1, batch_size):
            raise ValueError(f"frame_indices batch mismatch: got {idx.shape[0]} expected 1 or {batch_size}")
        if idx.shape[1] != num_frames:
            raise ValueError(f"frame_indices length mismatch: got {idx.shape[1]} expected {num_frames}")
        if idx.shape[0] == 1 and batch_size != 1:
            idx = idx.expand(batch_size, -1)

        if p_t != 1:
            if torch.any(idx % p_t != 0):
                raise ValueError("frame_indices must be multiples of temporal patch size p_t")
            idx = idx // p_t
        if int(idx.max().item()) >= self.max_seq_len or int(idx.min().item()) < 0:
            raise ValueError(f"frame_indices out of range for max_seq_len={self.max_seq_len}")

        # Split the precomputed 1D rotary tables into (t, h, w).
        split_sizes = [self.t_dim, self.h_dim, self.w_dim]
        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        # Time: per-sample absolute indices.
        idx = idx.to(device=freqs_cos[0].device)
        t_cos = freqs_cos[0].index_select(0, idx.reshape(-1)).view(batch_size, ppf, -1)
        t_sin = freqs_sin[0].index_select(0, idx.reshape(-1)).view(batch_size, ppf, -1)
        t_cos = t_cos.view(batch_size, ppf, 1, 1, -1).expand(batch_size, ppf, pph, ppw, -1)
        t_sin = t_sin.view(batch_size, ppf, 1, 1, -1).expand(batch_size, ppf, pph, ppw, -1)

        # Space: shared across batch.
        h_cos = freqs_cos[1][:pph].view(1, 1, pph, 1, -1).expand(batch_size, ppf, pph, ppw, -1)
        h_sin = freqs_sin[1][:pph].view(1, 1, pph, 1, -1).expand(batch_size, ppf, pph, ppw, -1)
        w_cos = freqs_cos[2][:ppw].view(1, 1, 1, ppw, -1).expand(batch_size, ppf, pph, ppw, -1)
        w_sin = freqs_sin[2][:ppw].view(1, 1, 1, ppw, -1).expand(batch_size, ppf, pph, ppw, -1)

        freqs_cos_out = torch.cat([t_cos, h_cos, w_cos], dim=-1).reshape(batch_size, ppf * pph * ppw, 1, -1)
        freqs_sin_out = torch.cat([t_sin, h_sin, w_sin], dim=-1).reshape(batch_size, ppf * pph * ppw, 1, -1)
        return freqs_cos_out, freqs_sin_out


def enable_wan_absolute_time_rope(model: nn.Module) -> None:
    """Wrap `model.rope` so we can set absolute-time indices for short temporal inputs."""
    rope = getattr(model, "rope", None)
    if rope is None:
        raise ValueError("Model has no .rope attribute; expected diffusers WanTransformer3DModel")
    if isinstance(rope, WanRotaryPosEmbedAbsTime):
        return
    model.rope = WanRotaryPosEmbedAbsTime(rope)


def set_wan_frame_indices(model: nn.Module, frame_indices: Optional[torch.Tensor]) -> None:
    """Set per-batch absolute frame indices used by the rotary embedding (or clear with None)."""
    rope = getattr(model, "rope", None)
    if rope is None:
        raise ValueError("Model has no .rope attribute; expected diffusers WanTransformer3DModel")
    if not isinstance(rope, WanRotaryPosEmbedAbsTime):
        enable_wan_absolute_time_rope(model)
        rope = model.rope
    rope.set_frame_indices(frame_indices)


__all__ = ["WanRotaryPosEmbedAbsTime", "enable_wan_absolute_time_rope", "set_wan_frame_indices"]
