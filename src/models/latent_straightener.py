from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class _StraightenerNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        blocks: int,
        *,
        use_residual: bool = True,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if blocks < 1:
            raise ValueError("blocks must be >= 1")
        pad = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=pad),
            nn.SiLU(),
        ]
        for _ in range(blocks - 1):
            layers.extend(
                [
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=pad),
                    nn.SiLU(),
                ]
            )
        layers.append(nn.Conv2d(hidden_channels, in_channels, 1))
        self.net = nn.Sequential(*layers)
        self.use_residual = bool(use_residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if self.use_residual:
            return x + y
        return y


class LatentStraightener(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        blocks: int = 2,
        *,
        use_residual: bool = True,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.blocks = int(blocks)
        self.kernel_size = int(kernel_size)
        self.use_residual = bool(use_residual)
        self.encoder = _StraightenerNet(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            blocks=self.blocks,
            use_residual=self.use_residual,
            kernel_size=self.kernel_size,
        )
        self.decoder = _StraightenerNet(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            blocks=self.blocks,
            use_residual=self.use_residual,
            kernel_size=self.kernel_size,
        )

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        return self.encoder(z)

    def decode(self, s: torch.Tensor) -> torch.Tensor:
        return self.decoder(s)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(z))

    @torch.no_grad()
    def interpolate_pair(
        self, z0: torch.Tensor, z1: torch.Tensor, alpha: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if alpha.dim() == 1:
            alpha = alpha.view(-1, 1, 1, 1)
        alpha = alpha.to(dtype=z0.dtype, device=z0.device).clamp(0.0, 1.0)
        s0 = self.encode(z0)
        s1 = self.encode(z1)
        s = (1.0 - alpha) * s0 + alpha * s1
        z_hat = self.decode(s)
        return z_hat, s


def load_latent_straightener(
    ckpt_path: str, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
) -> Tuple[LatentStraightener, dict]:
    payload = torch.load(ckpt_path, map_location="cpu")
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    in_channels = int(meta.get("in_channels", 16))
    hidden_channels = int(meta.get("hidden_channels", 64))
    blocks = int(meta.get("blocks", 2))
    kernel_size = int(meta.get("kernel_size", 3))
    use_residual = bool(meta.get("use_residual", True))
    if dtype is None:
        meta_dtype = meta.get("model_dtype")
        if meta_dtype:
            name = str(meta_dtype).lower()
            if name in {"bf16", "bfloat16"}:
                dtype = torch.bfloat16
            elif name in {"fp16", "float16"}:
                dtype = torch.float16
            elif name in {"fp32", "float32"}:
                dtype = torch.float32
    model = LatentStraightener(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        blocks=blocks,
        use_residual=use_residual,
        kernel_size=kernel_size,
    )
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=True)
    if device is not None or dtype is not None:
        model.to(device=device, dtype=dtype)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, meta


__all__ = ["LatentStraightener", "load_latent_straightener"]
