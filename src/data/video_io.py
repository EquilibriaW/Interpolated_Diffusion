from __future__ import annotations

import os
from typing import Optional

import math
import numpy as np
import torch
import torch.nn.functional as F

try:  # optional dependency
    import decord  # type: ignore

    _DECORD_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    _DECORD_AVAILABLE = False

try:  # imageio v2 preferred for stable API
    import imageio.v2 as imageio  # type: ignore
except Exception:  # pragma: no cover - optional
    import imageio  # type: ignore


def resolve_video_path(video_root: str, video_name: str) -> Optional[str]:
    base, ext = os.path.splitext(video_name)
    candidates = [
        video_name,
        f"{video_name}.mp4",
        f"{video_name}.avi",
        f"{video_name}.mkv",
    ]
    if ext:
        candidates.extend(
            [
                f"{base}.mp4",
                f"{base}.avi",
                f"{base}.mkv",
            ]
        )
    seen = set()
    ordered = []
    for name in candidates:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    for name in ordered:
        path = os.path.join(video_root, name)
        if os.path.exists(path):
            return path
    return None


def _center_crop(frames: torch.Tensor) -> torch.Tensor:
    # frames: [T,C,H,W]
    _, _, h, w = frames.shape
    if h == w:
        return frames
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    return frames[:, :, top : top + side, left : left + side]


def _resize_frames(frames: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(frames, size=(size, size), mode="bilinear", align_corners=False)


def _sample_indices(start: int, end: int, num_frames: int) -> np.ndarray:
    if end <= start:
        return np.full((num_frames,), max(0, start), dtype=np.int64)
    idx = np.linspace(start, end - 1, num_frames)
    return np.clip(np.round(idx), start, max(start, end - 1)).astype(np.int64)


def _read_with_decord(path: str, start_sec: float, end_sec: float, num_frames: int) -> np.ndarray:
    vr = decord.VideoReader(path)
    fps = float(getattr(vr, "get_avg_fps", lambda: 30.0)())
    total = len(vr)
    start_f = int(max(0.0, start_sec) * fps)
    end_f = int(max(start_sec, end_sec) * fps)
    start_f = min(start_f, max(0, total - 1))
    end_f = min(max(start_f + 1, end_f), total)
    idx = _sample_indices(start_f, end_f, num_frames)
    batch = vr.get_batch(idx).asnumpy()
    return batch


def _read_with_imageio(path: str, start_sec: float, end_sec: float, num_frames: int) -> np.ndarray:
    reader = imageio.get_reader(path, "ffmpeg")
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 30.0))
    nframes_raw = meta.get("nframes", 0)
    if nframes_raw is None or (isinstance(nframes_raw, (float, int)) and not math.isfinite(nframes_raw)):
        nframes = 0
    else:
        try:
            nframes = int(nframes_raw)
        except (TypeError, ValueError):
            nframes = 0
    start_f = int(max(0.0, start_sec) * fps)
    end_f = int(max(start_sec, end_sec) * fps)
    if nframes > 0:
        start_f = min(start_f, max(0, nframes - 1))
        end_f = min(max(start_f + 1, end_f), nframes)
    idx = _sample_indices(start_f, end_f, num_frames)
    frames = []
    for i in idx:
        try:
            frames.append(reader.get_data(int(i)))
        except Exception:
            break
    reader.close()
    if len(frames) == 0:
        raise RuntimeError(f"Failed to read frames from {path}")
    return np.stack(frames, axis=0)


def read_video_clip(
    path: str,
    start_sec: float,
    end_sec: float,
    num_frames: int,
    resize: Optional[int] = None,
    center_crop: bool = True,
) -> torch.Tensor:
    if _DECORD_AVAILABLE:
        frames = _read_with_decord(path, start_sec, end_sec, num_frames)
    else:
        frames = _read_with_imageio(path, start_sec, end_sec, num_frames)
    if frames.shape[0] < num_frames:
        pad = np.repeat(frames[-1:], num_frames - frames.shape[0], axis=0)
        frames = np.concatenate([frames, pad], axis=0)
    elif frames.shape[0] > num_frames:
        frames = frames[:num_frames]
    # frames: [T,H,W,3] uint8/float
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    if center_crop:
        frames_t = _center_crop(frames_t)
    if resize is not None:
        frames_t = _resize_frames(frames_t, int(resize))
    return frames_t


__all__ = ["read_video_clip", "resolve_video_path"]
