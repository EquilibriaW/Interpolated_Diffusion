from __future__ import annotations

import argparse
import os
from typing import List


def _require_hf() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("huggingface_hub is required. Install with `pip install huggingface_hub`.") from exc


def _snapshot_download(repo_id: str, out_dir: str, patterns: List[str]) -> str:
    from huggingface_hub import snapshot_download

    os.makedirs(out_dir, exist_ok=True)
    return snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=out_dir,
        local_dir_use_symlinks=False,
        allow_patterns=patterns,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="worstcoder/Wan_datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K",
    )
    parser.add_argument("--out_dir", type=str, default="data/wan_synth")
    parser.add_argument(
        "--shard_pattern",
        type=str,
        default="shard-*.tar",
        help="Glob pattern for shards within the dataset directory.",
    )
    parser.add_argument("--also_patterns", type=str, default="", help="Extra comma-separated patterns to download.")
    args = parser.parse_args()

    _require_hf()
    patterns = [f"{args.dataset}/{args.shard_pattern}"]
    if args.also_patterns:
        patterns.extend([p.strip() for p in args.also_patterns.split(",") if p.strip()])

    snapshot_dir = _snapshot_download(args.repo_id, args.out_dir, patterns)
    print(f"Downloaded to {snapshot_dir}")


if __name__ == "__main__":
    main()
