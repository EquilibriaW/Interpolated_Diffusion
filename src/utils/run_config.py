import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import torch


def _get_repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _get_git_info(repo_root: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
        info["dirty"] = (
            subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, text=True).strip()
            != ""
        )
    except Exception:
        return {}
    return info


def _get_env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    else:
        info["device_count"] = 0
        info["devices"] = []
    return info


def _load_dataset_meta(prepared_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not prepared_path:
        return None
    meta_path = os.path.join(os.path.dirname(prepared_path), "meta.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_run_config(
    log_dir: Optional[str],
    args: Any,
    writer: Optional[Any] = None,
    prepared_path: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    if log_dir is None:
        return None
    os.makedirs(log_dir, exist_ok=True)
    repo_root = _get_repo_root()
    payload: Dict[str, Any] = {
        "cmd": " ".join(sys.argv),
        "args": vars(args),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "env": _get_env_info(),
        "git": _get_git_info(repo_root),
    }
    dataset_meta = _load_dataset_meta(prepared_path)
    if dataset_meta is not None:
        payload["dataset_meta"] = dataset_meta
    if extra:
        payload["extra"] = extra
    out_path = os.path.join(log_dir, "run_config.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    if writer is not None and hasattr(writer, "add_text"):
        try:
            writer.add_text("run/config", json.dumps(payload, indent=2, default=str), 0)
        except Exception:
            pass
    return out_path
