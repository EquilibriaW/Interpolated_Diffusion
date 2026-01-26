import torch


def apply_clamp(x_hat: torch.Tensor, x_ref: torch.Tensor, clamp_mask: torch.Tensor, clamp_dims: str) -> torch.Tensor:
    if clamp_mask is None:
        return x_hat
    if clamp_dims == "pos":
        x_hat[:, :, :2] = torch.where(clamp_mask.unsqueeze(-1), x_ref[:, :, :2], x_hat[:, :, :2])
        return x_hat
    return torch.where(clamp_mask.unsqueeze(-1), x_ref, x_hat)


def apply_soft_clamp(
    x_hat: torch.Tensor,
    x_ref: torch.Tensor,
    conf: torch.Tensor,
    lam: float,
    clamp_dims: str,
) -> torch.Tensor:
    if conf is None:
        return x_hat
    if lam <= 0.0:
        return x_hat
    if conf.dim() == 2:
        w = conf.unsqueeze(-1)
    else:
        w = conf
    w = w * float(lam)
    if clamp_dims == "pos":
        x_hat[:, :, :2] = x_hat[:, :, :2] + w * (x_ref[:, :, :2] - x_hat[:, :, :2])
        return x_hat
    return x_hat + w * (x_ref - x_hat)
