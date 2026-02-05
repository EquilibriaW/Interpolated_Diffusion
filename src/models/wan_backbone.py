from __future__ import annotations

from typing import Optional

import torch


def resolve_dtype(name: str | None) -> Optional[torch.dtype]:
    if not name:
        return None
    name = str(name).lower()
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


def load_wan_transformer(
    repo_id: str,
    *,
    subfolder: str = "transformer",
    torch_dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
):
    from diffusers import WanTransformer3DModel

    model = WanTransformer3DModel.from_pretrained(repo_id, subfolder=subfolder, torch_dtype=torch_dtype)
    if device is not None:
        model.to(device)
    return model


__all__ = ["load_wan_transformer", "resolve_dtype"]
