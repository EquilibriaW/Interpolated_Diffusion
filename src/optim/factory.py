from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch


@dataclass
class OptimizerBuildInfo:
    name: str
    n_total: int
    n_muon: int
    n_adam: int


def _split_muon_adam_params(
    named_params: Iterable[Tuple[str, torch.nn.Parameter]],
) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """Heuristic parameter split for Muon+Adam.

    Muon is applied to matrix-like hidden weights (ndim>=2) and AdamW is used for
    scalar/bias/norm/embedding-like parameters.
    """
    muon_params: List[torch.nn.Parameter] = []
    adam_params: List[torch.nn.Parameter] = []
    for name, p in named_params:
        if not p.requires_grad:
            continue
        n = str(name).lower()
        is_norm_or_embed = any(k in n for k in ("norm", "ln", "bn", "bias", "embed", "emb", "pos", "level_emb", "class_emb"))
        if p.ndim >= 2 and (not is_norm_or_embed):
            muon_params.append(p)
        else:
            adam_params.append(p)
    return muon_params, adam_params


def build_optimizer(
    model: torch.nn.Module,
    *,
    optimizer_name: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    muon_weight_decay: float = 0.01,
    muon_adam_betas: Tuple[float, float] = (0.9, 0.95),
    muon_adam_eps: float = 1e-10,
) -> Tuple[torch.optim.Optimizer, OptimizerBuildInfo]:
    name = str(optimizer_name).lower()
    named_params = list(model.named_parameters())
    all_trainable = [p for _, p in named_params if p.requires_grad]
    if not all_trainable:
        raise ValueError("model has no trainable parameters")

    if name == "adamw":
        opt = torch.optim.AdamW(all_trainable, lr=float(lr), weight_decay=float(weight_decay))
        info = OptimizerBuildInfo(name="adamw", n_total=len(all_trainable), n_muon=0, n_adam=len(all_trainable))
        return opt, info

    if name != "muon":
        raise ValueError(f"unknown optimizer: {optimizer_name}")

    try:
        from muon import SingleDeviceMuonWithAuxAdam
    except Exception:
        from .muon_fallback import SingleDeviceMuonWithAuxAdam

    muon_params, adam_params = _split_muon_adam_params(named_params)
    if not muon_params:
        raise ValueError("Muon optimizer requested but no Muon-eligible parameters were found")

    param_groups = []
    if muon_params:
        param_groups.append(
            {
                "params": muon_params,
                "lr": float(muon_lr),
                "momentum": float(muon_momentum),
                "weight_decay": float(muon_weight_decay),
                "use_muon": True,
            }
        )
    if adam_params:
        param_groups.append(
            {
                "params": adam_params,
                "lr": float(lr),
                "betas": (float(muon_adam_betas[0]), float(muon_adam_betas[1])),
                "eps": float(muon_adam_eps),
                "weight_decay": float(weight_decay),
                "use_muon": False,
            }
        )
    opt = SingleDeviceMuonWithAuxAdam(param_groups)
    info = OptimizerBuildInfo(
        name="muon",
        n_total=len(all_trainable),
        n_muon=len(muon_params),
        n_adam=len(adam_params),
    )
    return opt, info


__all__ = ["OptimizerBuildInfo", "build_optimizer"]
