from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _sample_velocity(rng: np.random.RandomState, max_speed: int = 2) -> Tuple[int, int]:
    speeds = [s for s in range(-max_speed, max_speed + 1) if s != 0]
    vx = int(rng.choice(speeds))
    vy = int(rng.choice(speeds))
    return vx, vy


class MovingShapesVideoDataset(Dataset):
    """Procedural toy video dataset with moving squares/circles."""

    def __init__(
        self,
        T: int = 16,
        H: int = 64,
        W: int | None = None,
        n_samples: int = 100000,
        seed: int = 0,
        n_objects_range: Tuple[int, int] = (1, 3),
        latent_size: int = 16,
    ):
        self.T = int(T)
        self.H = int(H)
        self.W = int(W) if W is not None else int(H)
        self.n_samples = int(n_samples)
        self.seed = int(seed)
        self.n_objects_range = n_objects_range
        self.latent_size = int(latent_size)
        self.data_dim = 3 * self.latent_size * self.latent_size

    def __len__(self) -> int:
        return self.n_samples

    def _render_frame(self, obj_states, H: int, W: int) -> np.ndarray:
        frame = np.zeros((H, W, 3), dtype=np.float32)
        for obj in obj_states:
            x, y, size, shape, color = obj["x"], obj["y"], obj["size"], obj["shape"], obj["color"]
            x0 = max(0, x - size)
            x1 = min(W - 1, x + size)
            y0 = max(0, y - size)
            y1 = min(H - 1, y + size)
            if shape == "square":
                frame[y0 : y1 + 1, x0 : x1 + 1] = color
            else:
                yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
                mask = (xx - x) ** 2 + (yy - y) ** 2 <= size**2
                frame[y0 : y1 + 1, x0 : x1 + 1][mask] = color
        return frame

    def _simulate(self, rng: np.random.RandomState) -> Tuple[np.ndarray, int]:
        H, W = self.H, self.W
        n_min, n_max = self.n_objects_range
        n_obj = int(rng.randint(n_min, n_max + 1))
        obj_states = []
        for _ in range(n_obj):
            shape = "square" if rng.rand() < 0.5 else "circle"
            size = int(rng.randint(3, 9))
            x = int(rng.randint(size, W - size))
            y = int(rng.randint(size, H - size))
            vx, vy = _sample_velocity(rng, max_speed=2)
            color = rng.uniform(0.2, 1.0, size=(3,)).astype(np.float32)
            obj_states.append({"x": x, "y": y, "vx": vx, "vy": vy, "size": size, "shape": shape, "color": color})

        frames = []
        for _ in range(self.T):
            frames.append(self._render_frame(obj_states, H, W))
            for obj in obj_states:
                x = obj["x"] + obj["vx"]
                y = obj["y"] + obj["vy"]
                if x < obj["size"] or x > W - 1 - obj["size"]:
                    obj["vx"] *= -1
                    x = obj["x"] + obj["vx"]
                if y < obj["size"] or y > H - 1 - obj["size"]:
                    obj["vy"] *= -1
                    y = obj["y"] + obj["vy"]
                obj["x"], obj["y"] = int(x), int(y)
        frames = np.stack(frames, axis=0)  # [T,H,W,3]
        return frames, n_obj

    def __getitem__(self, idx: int) -> Dict[str, object]:
        rng = np.random.RandomState(self.seed + int(idx))
        frames, n_obj = self._simulate(rng)
        # [T,3,H,W]
        frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        z = F.interpolate(frames_t, size=(self.latent_size, self.latent_size), mode="bilinear", align_corners=False)
        z_flat = z.reshape(self.T, -1).contiguous()
        start_goal = torch.cat([z_flat[0], z_flat[-1]], dim=0)
        return {
            "x": z_flat,
            "cond": {"start_goal": start_goal},
            "meta": {"n_objects": int(n_obj)},
        }


def _infer_latent_size(D: int) -> int:
    size = int(round((D / 3) ** 0.5))
    if 3 * size * size != D:
        raise ValueError(f"Cannot infer latent size from D={D}")
    return size


def decode_latents(z_flat: torch.Tensor, out_size: int = 64) -> torch.Tensor:
    """Decode flattened latents back to RGB frames for visualization."""
    if z_flat.dim() == 2:
        T = z_flat.shape[0]
        size = _infer_latent_size(z_flat.shape[-1])
        z = z_flat.view(T, 3, size, size)
        x = F.interpolate(z, size=(out_size, out_size), mode="bilinear", align_corners=False)
        return x
    if z_flat.dim() == 3:
        B, T, D = z_flat.shape
        size = _infer_latent_size(D)
        z = z_flat.view(B * T, 3, size, size)
        x = F.interpolate(z, size=(out_size, out_size), mode="bilinear", align_corners=False)
        return x.view(B, T, 3, out_size, out_size)
    raise ValueError("z_flat must have shape [T,D] or [B,T,D]")
