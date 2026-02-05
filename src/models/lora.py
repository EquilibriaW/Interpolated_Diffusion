from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float = 1.0, dropout: float = 0.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, base.out_features, bias=False)
        # Ensure LoRA params live on the same device/dtype as the base layer.
        self.lora_A.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_B.to(device=base.weight.device, dtype=base.weight.dtype)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        lora = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return out + lora


@dataclass
class LoRAConfig:
    rank: int = 0
    alpha: float = 1.0
    dropout: float = 0.0
    targets: Sequence[str] = ("attn", "mlp")


def _resolve_parent(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora(
    model: nn.Module,
    config: LoRAConfig,
    module_filter: Iterable[str] | None = None,
) -> List[str]:
    if config.rank <= 0:
        return []
    targets = tuple(config.targets)
    replaced: List[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if module_filter is not None:
            if name not in module_filter:
                continue
        if targets and not any(t in name for t in targets):
            continue
        parent, child = _resolve_parent(model, name)
        setattr(parent, child, LoRALinear(module, config.rank, config.alpha, config.dropout))
        replaced.append(name)
    return replaced


def mark_only_lora_trainable(model: nn.Module) -> int:
    trainable = 0
    for name, param in model.named_parameters():
        requires_grad = "lora_" in name
        param.requires_grad = requires_grad
        if requires_grad:
            trainable += param.numel()
    return trainable


__all__ = ["LoRALinear", "LoRAConfig", "inject_lora", "mark_only_lora_trainable"]
