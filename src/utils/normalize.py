import torch


def logit_pos(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Apply logit transform to position dims (0:2), leaving other dims unchanged."""
    if x.shape[-1] < 2:
        return x
    out = x.clone()
    pos = out[..., :2].clamp(eps, 1.0 - eps)
    out[..., :2] = torch.log(pos / (1.0 - pos))
    return out


def sigmoid_pos(x: torch.Tensor) -> torch.Tensor:
    """Apply sigmoid to position dims (0:2), leaving other dims unchanged."""
    if x.shape[-1] < 2:
        return x
    out = x.clone()
    out[..., :2] = torch.sigmoid(out[..., :2])
    return out
