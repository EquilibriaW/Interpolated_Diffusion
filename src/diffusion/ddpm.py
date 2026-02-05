from typing import Dict, Optional, Tuple

import torch


def _gather(vec: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    out = vec.to(t.device)[t]
    if t.dim() == 1:
        return out.view(-1, 1, 1)
    if t.dim() == 2:
        return out.view(t.shape[0], t.shape[1], 1)
    raise ValueError("t must be 1D or 2D")


def q_sample(r0: torch.Tensor, t: torch.Tensor, schedule: Dict[str, torch.Tensor], noise: Optional[torch.Tensor] = None):
    if noise is None:
        noise = torch.randn_like(r0)
    sqrt_alpha_bar = _gather(schedule["sqrt_alpha_bar"], t)
    sqrt_one_minus = _gather(schedule["sqrt_one_minus_alpha_bar"], t)
    while sqrt_alpha_bar.dim() < r0.dim():
        sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
    while sqrt_one_minus.dim() < r0.dim():
        sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
    return sqrt_alpha_bar * r0 + sqrt_one_minus * noise, noise


def predict_x0_from_eps(rt: torch.Tensor, eps: torch.Tensor, t: torch.Tensor, schedule: Dict[str, torch.Tensor]):
    sqrt_alpha_bar = _gather(schedule["sqrt_alpha_bar"], t)
    sqrt_one_minus = _gather(schedule["sqrt_one_minus_alpha_bar"], t)
    while sqrt_alpha_bar.dim() < rt.dim():
        sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
    while sqrt_one_minus.dim() < rt.dim():
        sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
    return (rt - sqrt_one_minus * eps) / torch.clamp(sqrt_alpha_bar, min=1e-8)


def ddim_step(rt: torch.Tensor, eps: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor, schedule: Dict[str, torch.Tensor], eta: float = 0.0):
    # Deterministic DDIM by default (eta=0).
    alpha_bar_t = _gather(schedule["alpha_bar"], t)
    alpha_bar_prev = _gather(schedule["alpha_bar"], t_prev)
    while alpha_bar_t.dim() < rt.dim():
        alpha_bar_t = alpha_bar_t.unsqueeze(-1)
    while alpha_bar_prev.dim() < rt.dim():
        alpha_bar_prev = alpha_bar_prev.unsqueeze(-1)
    x0 = (rt - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
    if eta == 0.0:
        rt_prev = torch.sqrt(alpha_bar_prev) * x0 + torch.sqrt(1.0 - alpha_bar_prev) * eps
        return rt_prev
    # Stochastic DDIM
    sigma = (
        eta
        * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
        * torch.sqrt(1.0 - alpha_bar_t / alpha_bar_prev)
    )
    noise = torch.randn_like(rt)
    rt_prev = torch.sqrt(alpha_bar_prev) * x0 + torch.sqrt(1.0 - alpha_bar_prev - sigma**2) * eps + sigma * noise
    return rt_prev


def ddpm_step(rt: torch.Tensor, eps: torch.Tensor, t: torch.Tensor, schedule: Dict[str, torch.Tensor]):
    betas = _gather(schedule["betas"], t)
    alphas = _gather(schedule["alphas"], t)
    alpha_bar = _gather(schedule["alpha_bar"], t)
    while betas.dim() < rt.dim():
        betas = betas.unsqueeze(-1)
    while alphas.dim() < rt.dim():
        alphas = alphas.unsqueeze(-1)
    while alpha_bar.dim() < rt.dim():
        alpha_bar = alpha_bar.unsqueeze(-1)
    sqrt_one_minus = torch.sqrt(1.0 - alpha_bar)
    mean = (1.0 / torch.sqrt(alphas)) * (rt - (betas / sqrt_one_minus) * eps)
    if torch.all(t == 0):
        return mean
    noise = torch.randn_like(rt)
    var = betas
    return mean + torch.sqrt(var) * noise


def _timesteps(n_train: int, steps: int, schedule: str = "linear") -> torch.Tensor:
    device = torch.device("cpu")
    if steps <= 1:
        return torch.tensor([n_train - 1, 0], dtype=torch.long, device=device)
    if steps >= n_train:
        return torch.arange(n_train - 1, -1, -1, dtype=torch.long, device=device)
    if schedule == "quadratic":
        t = torch.linspace(0.0, 1.0, steps, device=device)
        times = (t * t * (n_train - 1)).long()
    elif schedule == "sqrt":
        t = torch.linspace(0.0, 1.0, steps, device=device)
        times = (torch.sqrt(t) * (n_train - 1)).long()
    else:
        times = torch.linspace(0, n_train - 1, steps, device=device).long()
    times = torch.unique(times)
    if times[0].item() != 0:
        times = torch.cat([torch.tensor([0], dtype=torch.long, device=device), times], dim=0)
    if times[-1].item() != n_train - 1:
        tail = torch.tensor([n_train - 1], dtype=torch.long, device=device)
        times = torch.cat([times, tail], dim=0)
    return torch.flip(times, dims=[0])


def ddim_sample(
    model,
    schedule: Dict[str, torch.Tensor],
    y: torch.Tensor,
    mask: torch.Tensor,
    cond: Dict[str, torch.Tensor],
    steps: int,
    eta: float = 0.0,
) -> torch.Tensor:
    device = y.device
    B, T, D = y.shape
    n_train = schedule["alpha_bar"].shape[0]
    times = _timesteps(n_train, steps)
    rt = torch.randn((B, T, D), device=device)
    rt = rt * (~mask).unsqueeze(-1)  # zero at keyframes
    for i in range(len(times) - 1):
        t = torch.full((B,), int(times[i]), device=device, dtype=torch.long)
        t_prev = torch.full((B,), int(times[i + 1]), device=device, dtype=torch.long)
        eps = model(rt, t, y, mask, cond)
        rt = ddim_step(rt, eps, t, t_prev, schedule, eta=eta)
        rt = rt * (~mask).unsqueeze(-1)
    return rt


def ddpm_sample(
    model,
    schedule: Dict[str, torch.Tensor],
    y: torch.Tensor,
    mask: torch.Tensor,
    cond: Dict[str, torch.Tensor],
) -> torch.Tensor:
    device = y.device
    B, T, D = y.shape
    n_train = schedule["alpha_bar"].shape[0]
    rt = torch.randn((B, T, D), device=device)
    rt = rt * (~mask).unsqueeze(-1)
    for t_int in range(n_train - 1, -1, -1):
        t = torch.full((B,), t_int, device=device, dtype=torch.long)
        eps = model(rt, t, y, mask, cond)
        rt = ddpm_step(rt, eps, t, schedule)
        rt = rt * (~mask).unsqueeze(-1)
    return rt
