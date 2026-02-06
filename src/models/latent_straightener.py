from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.transformer import TransformerEncoder
from src.utils.video_tokens import patchify_latents, unpatchify_tokens


def _sincos_1d_pos_embed(length: int, dim: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Returns [length, dim] 1D sin-cos embeddings (dim must be even)."""
    if dim <= 0:
        raise ValueError("dim must be positive")
    if dim % 2 != 0:
        raise ValueError("dim must be even for sin-cos embedding")
    pos = torch.arange(int(length), device=device, dtype=torch.float32).unsqueeze(1)  # [L,1]
    half = dim // 2
    omega = torch.arange(half, device=device, dtype=torch.float32) / float(max(half, 1))
    omega = 1.0 / (10000.0**omega)  # [half]
    out = pos * omega.unsqueeze(0)  # [L,half]
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # [L,dim]
    return emb.to(dtype=dtype)


def _sincos_2d_pos_embed(hp: int, wp: int, dim: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Returns [1, hp*wp, dim] 2D sin-cos embeddings."""
    if dim <= 0:
        raise ValueError("dim must be positive")
    # Split dims between y and x. Ensure each side is even (sin+cos).
    dim_y = dim // 2
    dim_x = dim - dim_y
    if dim_y % 2 != 0:
        dim_y -= 1
        dim_x += 1
    if dim_x % 2 != 0:
        dim_x -= 1
        dim_y += 1
    dim_y = max(dim_y, 2) if dim >= 4 else max(dim_y, 0)
    dim_x = max(dim_x, 2) if dim >= 4 else max(dim_x, 0)
    if dim_y > 0:
        emb_y = _sincos_1d_pos_embed(hp, dim_y, device=device, dtype=dtype)  # [hp,dim_y]
    else:
        emb_y = torch.zeros((hp, 0), device=device, dtype=dtype)
    if dim_x > 0:
        emb_x = _sincos_1d_pos_embed(wp, dim_x, device=device, dtype=dtype)  # [wp,dim_x]
    else:
        emb_x = torch.zeros((wp, 0), device=device, dtype=dtype)

    yy, xx = torch.meshgrid(
        torch.arange(hp, device=device),
        torch.arange(wp, device=device),
        indexing="ij",
    )
    pos = torch.cat([emb_y[yy], emb_x[xx]], dim=-1)  # [hp,wp,dim_y+dim_x]
    pos = pos.view(1, hp * wp, -1)
    if pos.shape[-1] < dim:
        pad = torch.zeros((1, hp * wp, dim - pos.shape[-1]), device=device, dtype=dtype)
        pos = torch.cat([pos, pad], dim=-1)
    elif pos.shape[-1] > dim:
        pos = pos[..., :dim]
    return pos


class _TokenTransformerStraightener(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int,
        patch_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        use_residual: bool,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.patch_size = int(patch_size)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.d_ff = int(d_ff)
        self.dropout = float(dropout)
        self.use_residual = bool(use_residual)

        self.in_proj: nn.Module | None = None
        self.out_proj: nn.Module | None = None
        if self.d_model != self.token_dim:
            self.in_proj = nn.Linear(self.token_dim, self.d_model, bias=False)
            self.out_proj = nn.Linear(self.d_model, self.token_dim, bias=False)

        self.tr = TransformerEncoder(
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout=self.dropout,
            cond_dim=None,
            causal=False,
            use_checkpoint=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
        tokens, (hp, wp) = patchify_latents(x.unsqueeze(1), self.patch_size)  # [B,1,N,token_dim]
        tok = tokens[:, 0]  # [B,N,token_dim]
        h = tok
        if self.in_proj is not None:
            h = self.in_proj(h)
        pos = _sincos_2d_pos_embed(hp, wp, self.d_model, device=h.device, dtype=h.dtype)
        h = h + pos
        h = self.tr(h)
        if self.out_proj is not None:
            h = self.out_proj(h)

        # Interpret transformer output as a delta in token space, then unpatchify.
        delta = unpatchify_tokens(h.unsqueeze(1), self.patch_size, (hp, wp))[:, 0]  # [B,C,H,W]
        if self.use_residual:
            return x + delta
        return delta


class LatentStraightenerTokenTransformer(nn.Module):
    """Token-grid transformer straightener.

    Patchifies z into (Hp*Wp) tokens, processes with a transformer, then unpatchifies back.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        patch_size: int = 4,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.0,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.d_ff = int(d_ff)
        self.dropout = float(dropout)
        self.use_residual = bool(use_residual)

        token_dim = int(self.in_channels) * int(self.patch_size) * int(self.patch_size)
        self.token_dim = int(token_dim)

        self.encoder = _TokenTransformerStraightener(
            token_dim=self.token_dim,
            patch_size=self.patch_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout=self.dropout,
            use_residual=self.use_residual,
        )
        self.decoder = _TokenTransformerStraightener(
            token_dim=self.token_dim,
            patch_size=self.patch_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout=self.dropout,
            use_residual=self.use_residual,
        )

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        return self.encoder(z)

    def decode(self, s: torch.Tensor) -> torch.Tensor:
        return self.decoder(s)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(z))


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
) -> Tuple[nn.Module, dict]:
    payload = torch.load(ckpt_path, map_location="cpu")
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    arch = str(meta.get("arch", "conv"))
    in_channels = int(meta.get("in_channels", 16))
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
    if arch in ("conv", "", "cnn"):
        hidden_channels = int(meta.get("hidden_channels", 64))
        blocks = int(meta.get("blocks", 2))
        kernel_size = int(meta.get("kernel_size", 3))
        model: nn.Module = LatentStraightener(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            blocks=blocks,
            use_residual=use_residual,
            kernel_size=kernel_size,
        )
    elif arch in ("token_transformer", "toktr"):
        model = LatentStraightenerTokenTransformer(
            in_channels=in_channels,
            patch_size=int(meta.get("patch_size", 4)),
            d_model=int(meta.get("d_model", 256)),
            n_layers=int(meta.get("n_layers", 6)),
            n_heads=int(meta.get("n_heads", 8)),
            d_ff=int(meta.get("d_ff", 1024)),
            dropout=float(meta.get("dropout", 0.0)),
            use_residual=use_residual,
        )
    else:
        raise ValueError(f"Unknown latent straightener arch '{arch}' in checkpoint meta")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=True)
    if device is not None or dtype is not None:
        model.to(device=device, dtype=dtype)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, meta  # type: ignore[return-value]


__all__ = [
    "LatentStraightener",
    "LatentStraightenerTokenTransformer",
    "load_latent_straightener",
]
