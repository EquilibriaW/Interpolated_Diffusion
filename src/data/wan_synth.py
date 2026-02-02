from __future__ import annotations

import glob
from typing import Iterable

import torch
from torch.utils.data import DataLoader


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


def create_wan_synth_dataloader(
    tar_path_pattern: str,
    batch_size: int,
    num_workers: int = 8,
    shuffle_buffer: int = 1000,
    prefetch_factor: int = 2,
) -> DataLoader:
    """WebDataset loader for TurboDiffusion synthetic Wan2.1 latents."""
    _require_webdataset()
    import webdataset as wds

    shards = glob.glob(tar_path_pattern)
    if not shards:
        raise FileNotFoundError(f"No files found with pattern '{tar_path_pattern}'")

    dataset = wds.DataPipeline(
        wds.SimpleShardList(shards),
        wds.shuffle(1000),
        wds.split_by_node,
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.shuffle(shuffle_buffer),
        wds.decode(wds.handle_extension("pt", wds.torch_loads)),
        wds.rename(latents="latent.pt", text_embed="embed.pt", text="prompt.txt"),
        wds.batched(batch_size, partial=False, collation_fn=_dict_collation_fn),
    )

    loader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )
    return loader


__all__ = ["create_wan_synth_dataloader"]
