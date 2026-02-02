from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .video_io import read_video_clip, resolve_video_path


def _parse_timecode(ts: str) -> float:
    # Expected format: HH.MM.SS.mmm
    parts = ts.strip().split(".")
    if len(parts) != 4:
        raise ValueError(f"Invalid timecode: {ts}")
    h, m, s, ms = [int(p) for p in parts]
    return float(h * 3600 + m * 60 + s) + float(ms) / 1000.0


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


def _build_video_index(video_dir: str) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for root, _, files in os.walk(video_dir):
        for name in files:
            base, _ = os.path.splitext(name)
            if base not in index:
                index[base] = os.path.join(root, name)
    return index


class LSMDCVideoDataset(Dataset):
    """LSMDC clips with captions and time windows (uses padded_start/end by default)."""

    _SPLIT_FILES = {
        "train": "LSMDC16_annos_training_someone.csv",
        "val": "LSMDC16_annos_val_someone.csv",
        "test": "LSMDC16_annos_test_someone.csv",
        "blind": "LSMDC16_annos_blindtest.csv",
    }

    def __init__(
        self,
        data_dir: str,
        video_dir: str,
        split: str = "train",
        T: int = 16,
        frame_size: int = 64,
        clip_seconds: float = 5.0,
        clip_strategy: str = "center",
        use_padded: bool = True,
        seed: int = 0,
        max_items: Optional[int] = None,
        verify_files: bool = True,
        index_videos: bool = True,
        max_retries: int = 3,
    ) -> None:
        if split not in self._SPLIT_FILES:
            raise ValueError(f"Unknown split {split}")
        self.data_dir = data_dir
        self.video_dir = video_dir
        self.split = split
        self.T = int(T)
        self.frame_size = int(frame_size)
        self.clip_seconds = float(clip_seconds) if clip_seconds is not None else None
        self.clip_strategy = clip_strategy
        self.use_padded = bool(use_padded)
        self.seed = int(seed)
        self.max_retries = int(max_retries)
        self._video_index = _build_video_index(video_dir) if index_videos else None

        split_path = os.path.join(data_dir, "task1", self._SPLIT_FILES[split])
        items: List[Dict[str, object]] = []
        with open(split_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 6:
                    continue
                clip_id = row[0]
                start_raw = row[3] if self.use_padded else row[1]
                end_raw = row[4] if self.use_padded else row[2]
                try:
                    start_sec = _parse_timecode(start_raw)
                    end_sec = _parse_timecode(end_raw)
                except Exception:
                    continue
                text = row[5]
                items.append(
                    {
                        "video": clip_id,
                        "text": text,
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                    }
                )

        if verify_files:
            verified: List[Dict[str, object]] = []
            for item in items:
                video = str(item["video"])
                path = None
                if self._video_index is not None:
                    path = self._video_index.get(video)
                if path is None:
                    path = resolve_video_path(self.video_dir, video)
                if path is not None:
                    verified.append(item)
            items = verified

        if max_items is not None:
            items = items[: int(max_items)]

        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        if len(self.items) == 0:
            raise RuntimeError("LSMDC dataset is empty after filtering")
        for attempt in range(self.max_retries):
            item = self.items[(idx + attempt) % len(self.items)]
            video = str(item["video"])
            path = None
            if self._video_index is not None:
                path = self._video_index.get(video)
            if path is None:
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
                "text": str(item.get("text", "")),
                "meta": {
                    "video": video,
                    "start_sec": clip_start,
                    "end_sec": clip_end,
                },
            }
        raise RuntimeError(f"Failed to load video after {self.max_retries} retries")


__all__ = ["LSMDCVideoDataset"]
