from __future__ import annotations

import json
import os
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset


class CachedLSMDCDataset(Dataset):
    def __init__(self, cache_dir: str, split: str = "train") -> None:
        self.cache_dir = cache_dir
        self.split = split
        index_path = os.path.join(cache_dir, split, "index.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Cache index not found: {index_path}")
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        self.shards = index["shards"]
        self.total = int(index["total"])
        self._cum_counts = []
        running = 0
        for shard in self.shards:
            running += int(shard["count"])
            self._cum_counts.append(running)
        self._cached_shard_id: Optional[int] = None
        self._cached_payload: Optional[Dict[str, object]] = None

    def __len__(self) -> int:
        return self.total

    def _load_shard(self, shard_id: int) -> Dict[str, object]:
        shard = self.shards[shard_id]
        path = shard["path"]
        payload = torch.load(path, map_location="cpu")
        self._cached_shard_id = shard_id
        self._cached_payload = payload
        return payload

    def _get_shard_for_index(self, idx: int) -> tuple[int, int]:
        for shard_id, end in enumerate(self._cum_counts):
            if idx < end:
                start = 0 if shard_id == 0 else self._cum_counts[shard_id - 1]
                return shard_id, idx - start
        raise IndexError("index out of range")

    def __getitem__(self, idx: int) -> Dict[str, object]:
        if idx < 0 or idx >= self.total:
            raise IndexError("index out of range")
        shard_id, local_idx = self._get_shard_for_index(idx)
        if self._cached_shard_id != shard_id or self._cached_payload is None:
            payload = self._load_shard(shard_id)
        else:
            payload = self._cached_payload
        latents = payload["latents"][local_idx]
        text_embed = payload.get("text_embed")
        text = payload.get("text")
        meta = payload.get("meta")
        item = {
            "latents": latents,
        }
        if text_embed is not None:
            item["text_embed"] = text_embed[local_idx]
        if text is not None:
            item["text"] = text[local_idx]
        if meta is not None:
            item["meta"] = meta[local_idx]
        return item


__all__ = ["CachedLSMDCDataset"]
