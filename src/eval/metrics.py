from typing import Dict, Optional

import torch


def _to_tensor(x, device: Optional[torch.device] = None) -> torch.Tensor:
    if torch.is_tensor(x):
        return x if device is None or x.device == device else x.to(device)
    t = torch.as_tensor(x)
    return t if device is None else t.to(device)


def _pos_to_cell(pos: torch.Tensor, h: int, w: int):
    x = pos[..., 0]
    y = pos[..., 1]
    oob = (x < 0) | (x > 1) | (y < 0) | (y > 1)
    j = torch.round(x * w).long()
    i = torch.round(y * h).long()
    i = torch.clamp(i, 0, h - 1)
    j = torch.clamp(j, 0, w - 1)
    return i, j, oob


def collision_rate(occ, traj) -> float:
    traj_t = _to_tensor(traj)
    occ_t = _to_tensor(occ, traj_t.device)
    h, w = occ_t.shape
    i, j, oob = _pos_to_cell(traj_t, h, w)
    coll = occ_t[i, j] > 0.5
    coll = coll | oob
    return float(coll.float().mean().item())


def goal_distance(goal, traj) -> float:
    traj_t = _to_tensor(traj)
    goal_t = _to_tensor(goal, traj_t.device)
    return float(torch.norm(traj_t[-1] - goal_t).item())


def success(goal, traj, h: int, w: int) -> float:
    cell_w = 1.0 / float(w)
    return float(goal_distance(goal, traj) < cell_w)


def path_length(traj) -> float:
    traj_t = _to_tensor(traj)
    return float(torch.norm(traj_t[1:] - traj_t[:-1], dim=-1).sum().item())


def smoothness(traj) -> float:
    traj_t = _to_tensor(traj)
    if traj_t.shape[0] < 3:
        return 0.0
    acc = traj_t[2:] - 2 * traj_t[1:-1] + traj_t[:-2]
    return float(torch.norm(acc, dim=-1).mean().item())


def mse_to_gt(traj, gt: Optional[object]) -> Optional[float]:
    if gt is None:
        return None
    traj_t = _to_tensor(traj)
    gt_t = _to_tensor(gt, traj_t.device)
    return float(((traj_t - gt_t) ** 2).mean().item())


def compute_metrics_batch(occ, traj, goal, gt: Optional[object] = None) -> Dict[str, torch.Tensor]:
    traj_t = _to_tensor(traj)
    device = traj_t.device
    occ_t = _to_tensor(occ, device)
    goal_t = _to_tensor(goal, device)

    if traj_t.dim() == 2:
        traj_t = traj_t.unsqueeze(0)
    if occ_t.dim() == 2:
        occ_t = occ_t.unsqueeze(0)
    if goal_t.dim() == 1:
        goal_t = goal_t.unsqueeze(0)

    if occ_t.shape[0] != traj_t.shape[0]:
        if occ_t.shape[0] == 1:
            occ_t = occ_t.expand(traj_t.shape[0], -1, -1)
        else:
            raise ValueError("occ batch size does not match traj batch size")
    if goal_t.shape[0] != traj_t.shape[0]:
        if goal_t.shape[0] == 1:
            goal_t = goal_t.expand(traj_t.shape[0], -1)
        else:
            raise ValueError("goal batch size does not match traj batch size")

    gt_t = None
    if gt is not None:
        gt_t = _to_tensor(gt, device)
        if gt_t.dim() == 2:
            gt_t = gt_t.unsqueeze(0)
        if gt_t.shape[0] != traj_t.shape[0]:
            if gt_t.shape[0] == 1:
                gt_t = gt_t.expand(traj_t.shape[0], -1, -1)
            else:
                raise ValueError("gt batch size does not match traj batch size")

    h, w = occ_t.shape[-2:]
    i, j, oob = _pos_to_cell(traj_t, h, w)
    b_idx = torch.arange(traj_t.shape[0], device=device).view(-1, 1).expand_as(i)
    coll = occ_t[b_idx, i, j] > 0.5
    coll = coll | oob
    collision = coll.float().mean(dim=1)

    goal_dist = torch.norm(traj_t[:, -1] - goal_t, dim=-1)
    success_t = (goal_dist < (1.0 / float(w))).float()
    path_len = torch.norm(traj_t[:, 1:] - traj_t[:, :-1], dim=-1).sum(dim=1)
    if traj_t.shape[1] < 3:
        smooth = torch.zeros_like(goal_dist)
    else:
        acc = traj_t[:, 2:] - 2 * traj_t[:, 1:-1] + traj_t[:, :-2]
        smooth = torch.norm(acc, dim=-1).mean(dim=1)

    out = {
        "collision_rate": collision,
        "goal_dist": goal_dist,
        "success": success_t,
        "path_length": path_len,
        "smoothness": smooth,
    }
    if gt_t is not None:
        out["mse_to_gt"] = ((traj_t - gt_t) ** 2).mean(dim=(1, 2))
    return out


def compute_metrics(
    occ,
    traj,
    goal,
    gt: Optional[object] = None,
) -> Dict[str, float]:
    batch = compute_metrics_batch(occ, traj, goal, gt)
    out = {k: float(v[0].item()) for k, v in batch.items()}
    return out
