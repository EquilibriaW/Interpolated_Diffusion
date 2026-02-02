from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .video_io import read_video_clip, resolve_video_path


def _load_json(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _mode_time_pair(times: List[List[int]]) -> Tuple[int, int]:
    pairs = [tuple(t) for t in times]
    if len(pairs) == 0:
        return 0, 0
    counts = Counter(pairs)
    (start, end), _ = counts.most_common(1)[0]
    return int(start), int(end)


def _clip_window(
    start_sec: float,
    end_sec: float,
    clip_seconds: Optional[float],
    rng: np.random.RandomState,
    strategy: str,
) -> Tuple[float, float]:
    if clip_seconds is None:
        return start_sec, end_sec
    seg_len = max(0.0, end_sec - start_sec)
    if clip_seconds >= seg_len or seg_len == 0.0:
        return start_sec, end_sec
    if strategy == "random":
        offset = rng.uniform(0.0, seg_len - clip_seconds)
    else:
        offset = 0.5 * (seg_len - clip_seconds)
    return start_sec + offset, start_sec + offset + clip_seconds


class DiDeMoVideoDataset(Dataset):
    """DiDeMo clips (2-5s) with captions."""

    def __init__(
        self,
        data_dir: str,
        video_dir: str,
        split: str = "train",
        T: int = 16,
        frame_size: int = 64,
        clip_seconds: float = 5.0,
        single_segment_only: bool = True,
        time_strategy: str = "mode",
        clip_strategy: str = "center",
        seed: int = 0,
        max_items: Optional[int] = None,
        verify_files: bool = True,
        max_retries: int = 3,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split {split}")
        self.data_dir = data_dir
        self.video_dir = video_dir
        self.split = split
        self.T = int(T)
        self.frame_size = int(frame_size)
        self.clip_seconds = float(clip_seconds)
        self.single_segment_only = bool(single_segment_only)
        self.time_strategy = time_strategy
        self.clip_strategy = clip_strategy
        self.seed = int(seed)
        self.max_retries = int(max_retries)

        split_path = os.path.join(data_dir, f"{split}_data.json")
        items = _load_json(split_path)

        filtered: List[Dict[str, object]] = []
        for item in items:
            times = item.get("times") or []
            start_idx, end_idx = _mode_time_pair(times)
            if self.single_segment_only and start_idx != end_idx:
                continue
            num_segments = int(item.get("num_segments", 6))
            start_sec = 5.0 * float(start_idx)
            end_sec = 5.0 * float(end_idx + 1)
            max_sec = 5.0 * float(num_segments)
            end_sec = min(end_sec, max_sec)
            record = dict(item)
            record.update({"start_sec": start_sec, "end_sec": end_sec})
            filtered.append(record)

        if verify_files:
            verified = []
            for item in filtered:
                video = str(item["video"])
                if resolve_video_path(self.video_dir, video) is not None:
                    verified.append(item)
            filtered = verified

        if max_items is not None:
            filtered = filtered[: int(max_items)]

        self.items = filtered

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        if len(self.items) == 0:
            raise RuntimeError("DiDeMo dataset is empty after filtering")
        for attempt in range(self.max_retries):
            item = self.items[(idx + attempt) % len(self.items)]
            video = str(item["video"])
            path = resolve_video_path(self.video_dir, video)
            if path is None:
                continue
            rng = np.random.RandomState(self.seed + idx + attempt)
            start_sec = float(item["start_sec"])
            end_sec = float(item["end_sec"])
            clip_start, clip_end = _clip_window(start_sec, end_sec, self.clip_seconds, rng, self.clip_strategy)
            try:
                frames = read_video_clip(
                    path,
                    clip_start,
                    clip_end,
                    self.T,
                    resize=self.frame_size,
                    center_crop=True,
                )
            except Exception:
                continue
            return {
                "frames": frames,
                "text": str(item.get("description", "")),
                "meta": {
                    "annotation_id": int(item.get("annotation_id", -1)),
                    "video": video,
                    "start_sec": clip_start,
                    "end_sec": clip_end,
                },
            }
        raise RuntimeError(f"Failed to load video after {self.max_retries} retries")


__all__ = ["DiDeMoVideoDataset"]
