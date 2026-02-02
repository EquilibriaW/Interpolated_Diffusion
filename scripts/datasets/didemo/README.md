# DiDeMo setup

1) Fetch metadata (annotations + hash list):

```bash
python scripts/datasets/didemo/fetch_didemo_metadata.py --out_dir data/didemo
```

2) Download videos from AWS (author script, ported to Python3):

```bash
python scripts/datasets/didemo/download_videos_aws.py \
  --download \
  --video_directory data/didemo/videos \
  --data_dir data/didemo
```

2b) (Alternative) Download videos from HuggingFace (mp4 tar parts):

```bash
python scripts/datasets/didemo/download_videos_hf.py \
  --repo_id friedrichor/DiDeMo \
  --cache_dir data/didemo/hf_cache \
  --video_dir data/didemo/videos \
  --flatten 1
```

Or run the combined setup:

```bash
bash scripts/datasets/didemo/setup_didemo_hf.sh
```

For a small smoke test, add `--limit 50` to download only the first 50 videos.

Notes:
- The AWS script stores files as `<video_name>.mp4` (this matches the original script’s behavior).
- Some videos are missing on AWS; the script writes a `missing_videos.txt` list under `data/didemo`.
 - If metadata filenames end in `.avi` but your download is `.mp4`, `resolve_video_path` will handle the extension swap.
