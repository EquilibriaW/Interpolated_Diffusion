"""
Python3 port of LocalizingMoments/download/download_videos_AWS.py.

Usage:
  python download_videos_aws.py --download --video_directory /path/to/videos --data_dir data/didemo
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from typing import Dict, List

import json


MULTIMEDIA_TEMPLATE = "https://multimedia-commons.s3-us-west-2.amazonaws.com/data/videos/mp4/%s/%s/%s.mp4"


def read_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_hash(path: str) -> Dict[str, str]:
    yfcc_hash: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line_count, line in enumerate(lines):
        sys.stdout.write(f"\r{line_count}/{len(lines)}")
        line = line.strip().split("\t")
        if len(line) < 2:
            continue
        yfcc_hash[line[0]] = line[1]
    print("\n")
    return yfcc_hash


def get_aws_link(h: str) -> str:
    return MULTIMEDIA_TEMPLATE % (h[:3], h[3:6], h)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_directory",
        type=str,
        default="videos/",
        help="Where downloaded videos should be stored",
    )
    parser.add_argument("--data_dir", type=str, default="data/didemo")
    parser.add_argument("--hash_file", type=str, default="")
    parser.add_argument("--download", dest="download", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--append_mp4", type=int, default=1)
    parser.set_defaults(download=False)
    args = parser.parse_args()

    if args.download:
        os.makedirs(args.video_directory, exist_ok=True)

    splits = ["test", "val", "train"]
    caps: List[dict] = []
    for split in splits:
        caps.extend(read_json(os.path.join(args.data_dir, f"{split}_data.json")))
    videos = sorted({cap["video"] for cap in caps})
    if args.limit and args.limit > 0:
        videos = videos[: args.limit]

    hash_file = args.hash_file or os.path.join(args.data_dir, "yfcc100m_hash.txt")
    yfcc100m_hash = read_hash(hash_file)

    missing_videos: List[str] = []

    for video_count, video in enumerate(videos):
        sys.stdout.write(f"\rDownloading video: {video_count}/{len(videos)}")
        video_id = video.split("_")[1]
        if video_id not in yfcc100m_hash:
            missing_videos.append(video)
            continue
        link = get_aws_link(yfcc100m_hash[video_id])
        if args.download:
            try:
                response = urllib.request.urlopen(link)
                out_name = f"{video}.mp4" if args.append_mp4 else str(video)
                out_path = os.path.join(args.video_directory, out_name)
                urllib.request.urlretrieve(response.geturl(), out_path)
            except Exception:
                print(f"\nCould not download link: {link}\n")
        else:
            try:
                urllib.request.urlopen(link)
            except Exception:
                missing_videos.append(video)
                print(f"\nCould not find link: {link}\n")

    if len(missing_videos) > 0:
        missing_path = os.path.join(args.data_dir, "missing_videos.txt")
        with open(missing_path, "w", encoding="utf-8") as write_txt:
            for missing_video in missing_videos:
                write_txt.writelines(f"{missing_video}\n")


if __name__ == "__main__":
    main()
