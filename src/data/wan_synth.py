from __future__ import annotations

import glob
import warnings
from typing import Iterable, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset
import itertools


def _require_webdataset() -> None:
    try:
        import webdataset  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("webdataset is required. Install with `pip install webdataset`.") from exc


def _dict_collation_fn(samples):
    if not samples:
        return {}
    keys = samples[0].keys()
    batched = {k: [] for k in keys}
    for sample in samples:
        for k in keys:
            batched[k].append(sample[k])
    for k in keys:
        if isinstance(batched[k][0], torch.Tensor):
            batched[k] = torch.stack(batched[k])
    return batched


def _attach_keys(sample: dict) -> dict:
    if "__key__" in sample and "key" not in sample:
        sample["key"] = sample["__key__"]
    if "__url__" in sample and "url" not in sample:
        sample["url"] = sample["__url__"]
    return sample


_LATENTS_TRANSPOSED_ONCE = False


def _normalize_latents(sample: dict) -> dict:
    global _LATENTS_TRANSPOSED_ONCE
    latents = sample.get("latents")
    if isinstance(latents, torch.Tensor) and latents.dim() == 4:
        t_dim, c_dim = int(latents.shape[0]), int(latents.shape[1])
        # If latents are stored as [C,T,H,W], transpose to [T,C,H,W].
        if t_dim <= 32 and c_dim > t_dim:
            latents = latents.permute(1, 0, 2, 3).contiguous()
            sample["latents"] = latents
            if not _LATENTS_TRANSPOSED_ONCE:
                warnings.warn(
                    "Detected Wan synth latents in [C,T,H,W] order; transposed to [T,C,H,W].",
                    stacklevel=2,
                )
                _LATENTS_TRANSPOSED_ONCE = True
    return sample


class _ZippedIterableDataset(IterableDataset):
    def __init__(self, base, anchors, merge_fn):
        super().__init__()
        self.base = base
        self.anchors = anchors
        self.merge_fn = merge_fn

    def __iter__(self):
        base_iter = iter(self.base)
        anchor_iter = iter(self.anchors)
        for pair in zip(base_iter, anchor_iter):
            yield self.merge_fn(pair)


class _KeyJoinIterableDataset(IterableDataset):
    def __init__(self, base, anchors, merge_fn, max_buffer: int = 2000, allow_missing: bool = False):
        super().__init__()
        self.base = base
        self.anchors = anchors
        self.merge_fn = merge_fn
        self.max_buffer = int(max_buffer)
        self.allow_missing = bool(allow_missing)

    def __iter__(self):
        base_iter = iter(self.base)
        anchor_iter = iter(self.anchors)
        base_buf: dict[str, dict] = {}
        anchor_buf: dict[str, dict] = {}
        for base_sample, anchor_sample in itertools.zip_longest(base_iter, anchor_iter):
            if base_sample is not None:
                key = base_sample.get("key")
                if key is None:
                    raise ValueError("Base sample missing key")
                match = anchor_buf.pop(key, None)
                if match is not None:
                    yield self.merge_fn((base_sample, match))
                else:
                    base_buf[key] = base_sample
            if anchor_sample is not None:
                key = anchor_sample.get("key")
                if key is None:
                    raise ValueError("Anchor sample missing key")
                match = base_buf.pop(key, None)
                if match is not None:
                    yield self.merge_fn((match, anchor_sample))
                else:
                    anchor_buf[key] = anchor_sample
            if self.max_buffer and (len(base_buf) + len(anchor_buf)) > self.max_buffer:
                raise RuntimeError(
                    f"Anchor join buffer exceeded {self.max_buffer} entries. "
                    "Streams may be too far out of order."
                )
            if anchor_sample is None and self.allow_missing:
                break
        if not self.allow_missing and (base_buf or anchor_buf):
            raise RuntimeError(
                f"Anchor join ended with unmatched samples: base={len(base_buf)} anchor={len(anchor_buf)}"
            )


def _build_wan_synth_ops(
    shards: list[str],
    *,
    shuffle: bool,
    shardshuffle: bool,
    shuffle_buffer: int,
    return_keys: bool,
    keep_text_embed: bool,
    keep_text: bool,
    rename_map: Optional[dict],
    resampled: bool = False,
    seed: int = 0,
):
    _require_webdataset()
    import webdataset as wds

    if resampled:
        ops = [wds.ResampledShards(shards, seed=int(seed), deterministic=True)]
    else:
        ops = [wds.SimpleShardList(shards)]
        if shardshuffle:
            ops.append(wds.shuffle(1000))
    ops.extend([wds.split_by_node, wds.split_by_worker, wds.tarfile_to_samples()])

    # Drop unneeded fields before shuffling/decoding to keep memory bounded.
    drop_keys = set()
    if not keep_text_embed:
        drop_keys.update({"embed.pt", "embed.pth"})
    if not keep_text:
        drop_keys.update({"prompt.txt"})
    if drop_keys:

        def _drop(sample: dict) -> dict:
            for k in drop_keys:
                sample.pop(k, None)
            return sample

        ops.append(wds.map(_drop))
    if shuffle:
        buffer = max(1, int(shuffle_buffer))
        ops.append(wds.shuffle(buffer))
    ops.append(wds.decode(wds.handle_extension("pt", wds.torch_loads)))
    if return_keys:
        ops.append(wds.map(_attach_keys))
    if rename_map:
        ops.append(wds.rename(**rename_map))
    ops.append(wds.map(_normalize_latents))
    return ops


def create_wan_synth_dataloader(
    tar_path_pattern: str,
    batch_size: int,
    num_workers: int = 8,
    shuffle_buffer: int = 1000,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    pin_memory: bool = True,
    shuffle: bool = True,
    shardshuffle: bool = True,
    return_keys: bool = False,
    keep_text_embed: bool = True,
    keep_text: bool = True,
    resampled: bool = False,
    seed: int = 0,
) -> DataLoader:
    """WebDataset loader for TurboDiffusion synthetic Wan2.1 latents."""
    _require_webdataset()
    import webdataset as wds

    shards = sorted(glob.glob(tar_path_pattern))
    if not shards:
        raise FileNotFoundError(f"No files found with pattern '{tar_path_pattern}'")

    if persistent_workers and not resampled:
        raise ValueError("persistent_workers=True requires resampled=True for WebDataset to avoid iterator deadlocks.")

    ops = _build_wan_synth_ops(
        shards,
        shuffle=shuffle,
        shardshuffle=shardshuffle,
        shuffle_buffer=shuffle_buffer,
        return_keys=return_keys,
        keep_text_embed=keep_text_embed,
        keep_text=keep_text,
        rename_map={
            "latents": "latent.pt;latent.pth",
            **({"text_embed": "embed.pt;embed.pth"} if keep_text_embed else {}),
            **({"text": "prompt.txt"} if keep_text else {}),
        },
        resampled=resampled,
        seed=seed,
    )
    ops.append(wds.batched(batch_size, partial=False, collation_fn=_dict_collation_fn))
    dataset = wds.DataPipeline(*ops)

    effective_prefetch = None if num_workers == 0 else prefetch_factor
    effective_persist = bool(persistent_workers) and num_workers > 0
    loader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=effective_prefetch,
        persistent_workers=effective_persist,
    )
    return loader


def create_wan_synth_anchor_dataloader(
    tar_path_pattern: str,
    anchor_path_pattern: str,
    batch_size: int,
    num_workers: int = 8,
    shuffle_buffer: int = 1000,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    pin_memory: bool = True,
    shuffle: bool = True,
    join_by_key: bool = True,
    max_key_buffer: int = 2000,
    allow_missing: bool = False,
    keep_text_embed: bool = True,
    keep_text: bool = True,
    resampled: bool = False,
    seed: int = 0,
) -> DataLoader:
    """WebDataset loader for Wan2.1 latents paired with precomputed anchors."""
    _require_webdataset()
    import webdataset as wds

    shards = sorted(glob.glob(tar_path_pattern))
    if not shards:
        raise FileNotFoundError(f"No files found with pattern '{tar_path_pattern}'")
    anchor_shards = sorted(glob.glob(anchor_path_pattern))
    if not anchor_shards:
        raise FileNotFoundError(f"No files found with pattern '{anchor_path_pattern}'")

    if persistent_workers and not resampled:
        raise ValueError("persistent_workers=True requires resampled=True for WebDataset to avoid iterator deadlocks.")

    base_ops = _build_wan_synth_ops(
        shards,
        shuffle=False,
        shardshuffle=False,
        shuffle_buffer=shuffle_buffer,
        return_keys=True,
        keep_text_embed=keep_text_embed,
        keep_text=keep_text,
        rename_map={
            "latents": "latent.pt;latent.pth",
            **({"text_embed": "embed.pt;embed.pth"} if keep_text_embed else {}),
            **({"text": "prompt.txt"} if keep_text else {}),
        },
        resampled=resampled,
        seed=seed,
    )
    anchor_ops = _build_wan_synth_ops(
        anchor_shards,
        shuffle=False,
        shardshuffle=False,
        shuffle_buffer=shuffle_buffer,
        return_keys=True,
        keep_text_embed=False,
        keep_text=False,
        rename_map={"anchor": "anchor.pth", "anchor_idx": "anchor_idx.pth"},
        resampled=resampled,
        seed=seed,
    )
    base = wds.DataPipeline(*base_ops)
    anchors = wds.DataPipeline(*anchor_ops)

    def _merge_pair(pair):
        sample, anchor = pair
        key_s = sample.get("key")
        key_a = anchor.get("key")
        if key_s is not None and key_a is not None and key_s != key_a:
            raise ValueError(f"Anchor key mismatch: {key_s} != {key_a}")
        out = dict(sample)
        out.update(anchor)
        return out

    if join_by_key:
        joined = _KeyJoinIterableDataset(base, anchors, _merge_pair, max_buffer=max_key_buffer, allow_missing=allow_missing)
        pipeline = [joined]
    else:
        zipped = _ZippedIterableDataset(base, anchors, _merge_pair)
        pipeline = [zipped]
    if shuffle:
        pipeline.append(wds.shuffle(max(1, int(shuffle_buffer))))
    pipeline.append(wds.batched(batch_size, partial=False, collation_fn=_dict_collation_fn))
    dataset = wds.DataPipeline(*pipeline)

    effective_prefetch = None if num_workers == 0 else prefetch_factor
    effective_persist = bool(persistent_workers) and num_workers > 0
    loader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=effective_prefetch,
        persistent_workers=effective_persist,
    )
    return loader


def create_wan_synth_teacher_dataloader(
    tar_path_pattern: str,
    teacher_path_pattern: str,
    batch_size: int,
    num_workers: int = 8,
    shuffle_buffer: int = 1000,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    pin_memory: bool = True,
    shuffle: bool = True,
    join_by_key: bool = True,
    max_key_buffer: int = 2000,
    allow_missing: bool = False,
    keep_text_embed: bool = True,
    keep_text: bool = True,
    resampled: bool = False,
    seed: int = 0,
) -> DataLoader:
    """WebDataset loader for Wan2.1 latents paired with LDMVFI teacher outputs."""
    _require_webdataset()
    import webdataset as wds

    shards = sorted(glob.glob(tar_path_pattern))
    if not shards:
        raise FileNotFoundError(f"No files found with pattern '{tar_path_pattern}'")
    teacher_shards = sorted(glob.glob(teacher_path_pattern))
    if not teacher_shards:
        raise FileNotFoundError(f"No files found with pattern '{teacher_path_pattern}'")

    if persistent_workers and not resampled:
        raise ValueError("persistent_workers=True requires resampled=True for WebDataset to avoid iterator deadlocks.")

    base_ops = _build_wan_synth_ops(
        shards,
        shuffle=False,
        shardshuffle=False,
        shuffle_buffer=shuffle_buffer,
        return_keys=True,
        keep_text_embed=keep_text_embed,
        keep_text=keep_text,
        rename_map={
            "latents": "latent.pt;latent.pth",
            **({"text_embed": "embed.pt;embed.pth"} if keep_text_embed else {}),
            **({"text": "prompt.txt"} if keep_text else {}),
        },
        resampled=resampled,
        seed=seed,
    )
    teacher_ops = _build_wan_synth_ops(
        teacher_shards,
        shuffle=False,
        shardshuffle=False,
        shuffle_buffer=shuffle_buffer,
        return_keys=True,
        keep_text_embed=False,
        keep_text=False,
        rename_map={"teacher": "teacher.pth", "teacher_idx": "teacher_idx.pth"},
        resampled=resampled,
        seed=seed,
    )
    base = wds.DataPipeline(*base_ops)
    teacher = wds.DataPipeline(*teacher_ops)

    def _merge_pair(pair):
        sample, teach = pair
        key_s = sample.get("key")
        key_t = teach.get("key")
        if key_s is not None and key_t is not None and key_s != key_t:
            raise ValueError(f"Teacher key mismatch: {key_s} != {key_t}")
        out = dict(sample)
        out.update(teach)
        return out

    if join_by_key:
        joined = _KeyJoinIterableDataset(
            base, teacher, _merge_pair, max_buffer=max_key_buffer, allow_missing=allow_missing
        )
        pipeline = [joined]
    else:
        zipped = _ZippedIterableDataset(base, teacher, _merge_pair)
        pipeline = [zipped]
    if shuffle:
        pipeline.append(wds.shuffle(max(1, int(shuffle_buffer))))
    pipeline.append(wds.batched(batch_size, partial=False, collation_fn=_dict_collation_fn))
    dataset = wds.DataPipeline(*pipeline)

    effective_prefetch = None if num_workers == 0 else prefetch_factor
    effective_persist = bool(persistent_workers) and num_workers > 0
    loader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=effective_prefetch,
        persistent_workers=effective_persist,
    )
    return loader


__all__ = [
    "create_wan_synth_dataloader",
    "create_wan_synth_anchor_dataloader",
    "create_wan_synth_teacher_dataloader",
]
