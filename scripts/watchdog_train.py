#!/usr/bin/env python3
import argparse
import glob
import os
import re
import shlex
import subprocess
import time


def _run(cmd, check=False):
    return subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _tmux_has_session(name: str) -> bool:
    return _run(["tmux", "has-session", "-t", name]).returncode == 0


def _tmux_kill(name: str) -> None:
    _run(["tmux", "kill-session", "-t", name])


def _tmux_start(name: str, cmd: str) -> None:
    _run(["tmux", "new-session", "-d", "-s", name, "bash", "-lc", cmd], check=True)


def _latest_ckpt(ckpt_dir: str) -> str | None:
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt")))
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)


def _inject_resume(cmd: str, ckpt_path: str | None) -> str:
    if not ckpt_path:
        return cmd
    m = re.search(r"\|&|\|", cmd)
    if m:
        main = cmd[: m.start()].strip()
        tail = cmd[m.start() :].lstrip()
    else:
        main = cmd.strip()
        tail = ""
    if "--resume" in main:
        main = re.sub(r"--resume\\s+\\S+", f"--resume {shlex.quote(ckpt_path)}", main)
    else:
        main = main + f" --resume {shlex.quote(ckpt_path)}"
    return f"{main} {tail}".strip()


def _tail_lines(path: str, n: int = 200) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()
    return lines[-n:]


def _parse_last_step(lines: list[str]) -> int | None:
    step_pat = re.compile(r"(\\d+)\\s*/\\s*(\\d+)")
    for line in reversed(lines):
        m = step_pat.search(line)
        if m:
            return int(m.group(1))
    return None


def _has_error(lines: list[str]) -> bool:
    err_markers = ("Traceback", "RuntimeError", "CUDA out of memory", "ValueError")
    for line in lines[-50:]:
        if any(tok in line for tok in err_markers):
            return True
    return False


def _ram_used_frac() -> float:
    total_kb = None
    avail_kb = None
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    avail_kb = int(line.split()[1])
                if total_kb is not None and avail_kb is not None:
                    break
    except OSError:
        return 0.0
    if not total_kb or avail_kb is None:
        return 0.0
    used_kb = max(total_kb - avail_kb, 0)
    return float(used_kb) / float(total_kb)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmux-session", required=True)
    ap.add_argument("--log-path", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--cmd", required=True, help="Training command to launch")
    ap.add_argument("--fallback-cmd", default="", help="Fallback command after a failure")
    ap.add_argument("--stall-minutes", type=float, default=10.0)
    ap.add_argument("--check-every", type=float, default=60.0)
    ap.add_argument("--use-fallback-after", type=int, default=1)
    ap.add_argument("--max-restarts", type=int, default=5)
    ap.add_argument("--ram-max-frac", type=float, default=0.0, help="Restart if RAM usage exceeds this fraction (0 disables)")
    args = ap.parse_args()

    last_step = None
    last_progress_time = time.time()
    restarts = 0
    use_fallback = False

    while True:
        lines = _tail_lines(args.log_path)
        step = _parse_last_step(lines)
        has_err = _has_error(lines)
        now = time.time()
        if step is not None and step != last_step:
            last_step = step
            last_progress_time = now

        if not _tmux_has_session(args.tmux_session):
            cmd = args.fallback_cmd if (use_fallback and args.fallback_cmd) else args.cmd
            cmd = _inject_resume(cmd, _latest_ckpt(args.ckpt_dir))
            _tmux_start(args.tmux_session, cmd)
            last_progress_time = now
            time.sleep(args.check_every)
            continue

        stalled = (now - last_progress_time) > float(args.stall_minutes) * 60.0
        ram_high = args.ram_max_frac > 0.0 and _ram_used_frac() >= float(args.ram_max_frac)
        if (has_err or stalled or ram_high) and restarts < args.max_restarts:
            ckpt = _latest_ckpt(args.ckpt_dir)
            cmd = args.fallback_cmd if (use_fallback and args.fallback_cmd) else args.cmd
            cmd = _inject_resume(cmd, ckpt)
            _tmux_kill(args.tmux_session)
            _tmux_start(args.tmux_session, cmd)
            restarts += 1
            if ram_high:
                use_fallback = True
            if restarts >= int(args.use_fallback_after):
                use_fallback = True
            last_progress_time = now

        time.sleep(args.check_every)


if __name__ == "__main__":
    main()
