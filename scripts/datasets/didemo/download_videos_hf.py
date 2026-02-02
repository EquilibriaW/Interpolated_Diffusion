from __future__ import annotations

import argparse
import glob
import os
import shutil
import tarfile
from typing import Dict, List


def _require_hf() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required. Install with `pip install huggingface_hub`."
        ) from exc


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


def _group_parts(part_paths: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for path in part_paths:
        base = path.rsplit(".part-", 1)[0]
        groups.setdefault(base, []).append(path)
    for base, parts in groups.items():
        parts.sort()
        groups[base] = parts
    return groups


def _concat_parts(parts: List[str], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as dst:
        for path in parts:
            with open(path, "rb") as src:
                shutil.copyfileobj(src, dst)


def _extract_tar(tar_path: str, video_dir: str) -> None:
    os.makedirs(video_dir, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(video_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="friedrichor/DiDeMo")
    parser.add_argument("--cache_dir", type=str, default="data/didemo/hf_cache")
    parser.add_argument("--video_dir", type=str, default="data/didemo/videos")
    parser.add_argument(
        "--allow_patterns",
        type=str,
        default="*Videos*tar*",
        help="Comma-separated glob patterns to download.",
    )
    parser.add_argument("--assemble_dir", type=str, default="data/didemo/hf_cache/assembled")
    parser.add_argument("--skip_existing", type=int, default=1)
    parser.add_argument("--cleanup", type=int, default=0)
    parser.add_argument("--flatten", type=int, default=0)
    args = parser.parse_args()

    _require_hf()
    patterns = [p.strip() for p in args.allow_patterns.split(",") if p.strip()]
    snapshot_dir = _snapshot_download(args.repo_id, args.cache_dir, patterns)

    part_paths = glob.glob(os.path.join(snapshot_dir, "**/*.tar.part-*"), recursive=True)
    tar_paths = glob.glob(os.path.join(snapshot_dir, "**/*.tar"), recursive=True)

    assembled = []
    for base, parts in _group_parts(part_paths).items():
        out_tar = os.path.join(args.assemble_dir, os.path.basename(base))
        if args.skip_existing and os.path.exists(out_tar):
            assembled.append(out_tar)
            continue
        print(f"Assembling {len(parts)} parts -> {out_tar}")
        _concat_parts(parts, out_tar)
        assembled.append(out_tar)

    all_tars = sorted(set(tar_paths + assembled))
    if len(all_tars) == 0:
        raise RuntimeError("No tar files found after download. Check repo_id/patterns.")

    for tar_path in all_tars:
        print(f"Extracting {tar_path} -> {args.video_dir}")
        _extract_tar(tar_path, args.video_dir)
        if args.cleanup and tar_path in assembled:
            try:
                os.remove(tar_path)
            except OSError:
                pass

    if args.flatten:
        moved = 0
        for root, _, files in os.walk(args.video_dir):
            if root == args.video_dir:
                continue
            for name in files:
                if not name.lower().endswith((".mp4", ".avi", ".mkv")):
                    continue
                src = os.path.join(root, name)
                dst = os.path.join(args.video_dir, name)
                if os.path.exists(dst):
                    continue
                os.makedirs(args.video_dir, exist_ok=True)
                shutil.move(src, dst)
                moved += 1
        if moved > 0:
            print(f"Flattened {moved} videos into {args.video_dir}")


if __name__ == "__main__":
    main()
