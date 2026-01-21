import torch


def apply_clamp(x_hat: torch.Tensor, x_ref: torch.Tensor, clamp_mask: torch.Tensor, clamp_dims: str) -> torch.Tensor:
    if clamp_mask is None:
        return x_hat
    if clamp_dims == "pos":
        x_hat[:, :, :2] = torch.where(clamp_mask.unsqueeze(-1), x_ref[:, :, :2], x_hat[:, :, :2])
        return x_hat
    return torch.where(clamp_mask.unsqueeze(-1), x_ref, x_hat)
