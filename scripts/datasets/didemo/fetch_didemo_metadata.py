from __future__ import annotations

import argparse
import os
import urllib.request

BASE_URL = "https://raw.githubusercontent.com/LisaAnne/LocalizingMoments/master/data"
FILES = [
    "train_data.json",
    "val_data.json",
    "test_data.json",
    "yfcc100m_hash.txt",
    "video_licenses.txt",
]


def _download(url: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"Downloading {url} -> {out_path}")
    urllib.request.urlretrieve(url, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/didemo")
    args = parser.parse_args()

    for name in FILES:
        url = f"{BASE_URL}/{name}"
        out_path = os.path.join(args.out_dir, name)
        _download(url, out_path)


if __name__ == "__main__":
    main()
