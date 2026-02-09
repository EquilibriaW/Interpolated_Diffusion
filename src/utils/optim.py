from __future__ import annotations

import inspect
from typing import Iterable

import torch


def _filter_kwargs(callable_obj, kwargs: dict) -> dict:
    """Filter kwargs by inspecting a callable signature.

    This lets us support optimizers whose constructor signature may differ across
    PyTorch versions (e.g. new optimizers introduced in later releases).
    """

    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):  # pragma: no cover
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def create_optimizer(
    name: str,
    params: Iterable[torch.nn.Parameter],
    *,
    lr: float,
    weight_decay: float,
    muon_momentum: float = 0.95,
    muon_nesterov: bool = False,
) -> torch.optim.Optimizer:
    name = str(name).lower().strip()
    if name in {"adamw"}:
        return torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))

    if name in {"muon"}:
        opt_cls = getattr(torch.optim, "Muon", None)
        if opt_cls is None:
            raise RuntimeError(
                "Requested optimizer=muon but torch.optim.Muon is not available in this PyTorch build."
            )
        kwargs = {
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "momentum": float(muon_momentum),
            "nesterov": bool(muon_nesterov),
        }
        kwargs = _filter_kwargs(opt_cls.__init__, kwargs)
        return opt_cls(params, **kwargs)

    raise ValueError(f"Unknown optimizer: {name}")


__all__ = ["create_optimizer"]

