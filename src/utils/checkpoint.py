from typing import Optional

import torch


def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int,
                    ema: Optional[object] = None):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    if ema is not None:
        payload["ema"] = ema.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
                    ema: Optional[object] = None, map_location: Optional[str] = None):
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if ema is not None and "ema" in payload:
        ema.load_state_dict(payload["ema"])
    return payload.get("step", 0)
