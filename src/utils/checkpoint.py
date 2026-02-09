from typing import Optional, Tuple, Union

import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    step: int,
    ema: Optional[object] = None,
    meta: Optional[dict] = None,
    *,
    save_optimizer: bool = True,
):
    payload = {
        "model": model.state_dict(),
        "step": step,
    }
    if save_optimizer and optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if ema is not None:
        payload["ema"] = ema.state_dict()
    if meta is not None:
        payload["meta"] = meta
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema: Optional[object] = None,
    map_location: Optional[str] = None,
    return_payload: bool = False,
) -> Union[int, Tuple[int, dict]]:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if ema is not None and "ema" in payload:
        ema.load_state_dict(payload["ema"])
    step = payload.get("step", 0)
    if return_payload:
        return step, payload
    return step
