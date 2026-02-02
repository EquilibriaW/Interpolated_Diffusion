from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

from src.data.maze import sdf_from_occupancy


def _resize_occ_stretch(occ: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    if occ.ndim == 3:
        if occ.shape[0] != 1:
            raise ValueError("Expected occ shape [1, H, W] for prepared datasets.")
        occ = occ[0]
    if occ.ndim != 2:
        raise ValueError("occ must be 2D")
    h, w = target_hw
    occ_t = torch.from_numpy(occ.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    occ_rs = F.interpolate(occ_t, size=(h, w), mode="nearest")
    return occ_rs[0, 0].cpu().numpy()


def _compute_isotropic_resize(
    h: int, w: int, target_h: int, target_w: int
) -> Tuple[int, int, int, int, float, float]:
    if h <= 1 or w <= 1:
        raise ValueError("occ must be at least 2x2 for resize")
    ratio_h = (target_h - 1) / float(h - 1)
    ratio_w = (target_w - 1) / float(w - 1)
    scale = min(ratio_h, ratio_w)
    new_h = int(np.floor((h - 1) * scale)) + 1
    new_w = int(np.floor((w - 1) * scale)) + 1
    new_h = max(2, min(target_h, new_h))
    new_w = max(2, min(target_w, new_w))
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    pad_top = pad_h // 2
    pad_left = pad_w // 2
    scale_y = (new_h - 1) / float(h - 1)
    scale_x = (new_w - 1) / float(w - 1)
    return new_h, new_w, pad_top, pad_left, scale_y, scale_x


def _resize_occ_pad(
    occ: np.ndarray,
    target_hw: Tuple[int, int],
    pad_value: float,
    scale_mode: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if occ.ndim == 3:
        if occ.shape[0] != 1:
            raise ValueError("Expected occ shape [1, H, W] for prepared datasets.")
        occ = occ[0]
    if occ.ndim != 2:
        raise ValueError("occ must be 2D")
    target_h, target_w = target_hw
    h, w = occ.shape
    if scale_mode == "none":
        if h > target_h or w > target_w:
            raise ValueError("pad scale_mode=none requires target size >= input size")
        new_h, new_w = int(h), int(w)
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        pad_top = pad_h // 2
        pad_left = pad_w // 2
        scale_y = 1.0
        scale_x = 1.0
        occ_rs = torch.from_numpy(occ.astype(np.float32))
    else:
        new_h, new_w, pad_top, pad_left, scale_y, scale_x = _compute_isotropic_resize(h, w, target_h, target_w)
        occ_t = torch.from_numpy(occ.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        occ_rs = F.interpolate(occ_t, size=(new_h, new_w), mode="nearest")[0, 0]
    pad_bottom = target_h - new_h - pad_top
    pad_right = target_w - new_w - pad_left
    occ_pad = F.pad(
        occ_rs,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="constant",
        value=float(pad_value),
    )
    info = {
        "orig_h": int(h),
        "orig_w": int(w),
        "new_h": int(new_h),
        "new_w": int(new_w),
        "pad_top": int(pad_top),
        "pad_left": int(pad_left),
        "pad_bottom": int(pad_bottom),
        "pad_right": int(pad_right),
        "scale_y": float(scale_y),
        "scale_x": float(scale_x),
        "scale_mode": str(scale_mode),
    }
    return occ_pad.cpu().numpy(), info


def _transform_positions(
    x: np.ndarray,
    h: int,
    w: int,
    target_h: int,
    target_w: int,
    scale_y: float,
    scale_x: float,
    pad_top: int,
    pad_left: int,
) -> np.ndarray:
    if x.size == 0:
        return x
    out = x.copy()
    denom_h = float(target_h - 1)
    denom_w = float(target_w - 1)
    if denom_h <= 0 or denom_w <= 0:
        return out
    y = out[..., 1]
    x0 = out[..., 0]
    y = (y * float(h - 1)) * scale_y + float(pad_top)
    x0 = (x0 * float(w - 1)) * scale_x + float(pad_left)
    out[..., 1] = y / denom_h
    out[..., 0] = x0 / denom_w
    if out.shape[-1] >= 4:
        out[..., 2] = out[..., 2] * scale_x * float(w - 1) / denom_w
        out[..., 3] = out[..., 3] * scale_y * float(h - 1) / denom_h
    return out


def _load_meta(path: str) -> dict:
    meta_path = os.path.join(os.path.dirname(path), "meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_unified(
    inputs: List[str],
    out_dir: str,
    target_hw: Tuple[int, int] | None,
    use_sdf: bool,
    resize_mode: str,
    pad_value: float,
    pad_scale_mode: str,
) -> None:
    xs = []
    sgs = []
    diffs = []
    occs = []
    sdfs = []
    env_ids = []
    metas = []
    transforms = []

    T_ref = None
    D_ref = None
    flip_y_ref = None
    if target_hw is None:
        max_h = 0
        max_w = 0
        for path in inputs:
            data = np.load(path)
            occ = data["occ"]
            if occ.ndim == 3:
                occ = occ[0]
            max_h = max(max_h, occ.shape[-2])
            max_w = max(max_w, occ.shape[-1])
        target_hw = (max_h, max_w)

    for path in inputs:
        data = np.load(path)
        x = data["x"].astype(np.float32)
        start_goal = data["start_goal"].astype(np.float32)
        difficulty = data.get("difficulty")
        occ = data["occ"].astype(np.float32)
        meta = _load_meta(path)
        metas.append(meta)
        env_id = meta.get("env_id", os.path.basename(os.path.dirname(path)))

        if T_ref is None:
            T_ref = x.shape[1]
        if D_ref is None:
            D_ref = x.shape[2]
        if x.shape[1] != T_ref or x.shape[2] != D_ref:
            raise ValueError(f"Mismatch in T/D for {path}: got {x.shape[1:]}, expected {(T_ref, D_ref)}")

        flip_y = meta.get("d4rl_flip_y")
        if flip_y_ref is None:
            flip_y_ref = flip_y
        elif flip_y is not None and flip_y_ref is not None and bool(flip_y) != bool(flip_y_ref):
            raise ValueError("All inputs must share the same d4rl_flip_y setting for unified dataset.")

        if resize_mode == "stretch":
            occ_rs = _resize_occ_stretch(occ, target_hw)
            sdf_rs = sdf_from_occupancy(occ_rs) if use_sdf else None
            x_rs = x
            sg_rs = start_goal
            transform = {
                "mode": "stretch",
                "orig_h": int(occ.shape[-2]) if occ.ndim >= 2 else None,
                "orig_w": int(occ.shape[-1]) if occ.ndim >= 2 else None,
                "target_h": int(target_hw[0]),
                "target_w": int(target_hw[1]),
            }
        else:
            occ_rs, info = _resize_occ_pad(occ, target_hw, pad_value, pad_scale_mode)
            sdf_rs = sdf_from_occupancy(occ_rs) if use_sdf else None
            h0 = int(occ.shape[-2]) if occ.ndim >= 2 else int(occ.shape[1])
            w0 = int(occ.shape[-1]) if occ.ndim >= 2 else int(occ.shape[2])
            x_rs = _transform_positions(
                x,
                h0,
                w0,
                target_hw[0],
                target_hw[1],
                info["scale_y"],
                info["scale_x"],
                info["pad_top"],
                info["pad_left"],
            )
            sg = start_goal.reshape(-1, 2)
            sg_rs = _transform_positions(
                sg,
                h0,
                w0,
                target_hw[0],
                target_hw[1],
                info["scale_y"],
                info["scale_x"],
                info["pad_top"],
                info["pad_left"],
            ).reshape(start_goal.shape)
            transform = {"mode": "pad", **info}

        n = x.shape[0]
        xs.append(x_rs.astype(np.float32))
        sgs.append(sg_rs.astype(np.float32))
        if difficulty is not None:
            diffs.append(difficulty.astype(np.int64))
        env_ids.append(np.full((n,), str(env_id), dtype=str))
        occs.append(np.repeat(occ_rs[None, ...], n, axis=0))
        if sdf_rs is not None:
            sdfs.append(np.repeat(sdf_rs[None, ...], n, axis=0))
        transforms.append(transform)

    x_all = np.concatenate(xs, axis=0)
    sg_all = np.concatenate(sgs, axis=0)
    occ_all = np.concatenate(occs, axis=0)
    env_all = np.concatenate(env_ids, axis=0)
    diff_all = None
    if diffs:
        diff_all = np.concatenate(diffs, axis=0)
    sdf_all = None
    if sdfs:
        sdf_all = np.concatenate(sdfs, axis=0)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dataset.npz")
    meta_path = os.path.join(out_dir, "meta.json")

    save_kwargs = {
        "x": x_all,
        "start_goal": sg_all,
        "occ": occ_all,
        "env_id": env_all,
    }
    if diff_all is not None:
        save_kwargs["difficulty"] = diff_all
    if sdf_all is not None:
        save_kwargs["sdf"] = sdf_all
    np.savez_compressed(out_path, **save_kwargs)

    meta_out = {
        "env_ids": [m.get("env_id", "") for m in metas],
        "num_samples": int(x_all.shape[0]),
        "T": int(T_ref),
        "data_dim": int(D_ref),
        "target_h": int(target_hw[0]),
        "target_w": int(target_hw[1]),
        "use_sdf": bool(use_sdf),
        "d4rl_flip_y": bool(flip_y_ref) if flip_y_ref is not None else False,
        "inputs": inputs,
        "resize_mode": resize_mode,
        "pad_value": float(pad_value),
        "pad_scale_mode": pad_scale_mode,
        "transforms": transforms,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)
    print(f"Wrote {out_path}")
    print(f"Wrote {meta_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--target_h", type=int, default=0)
    p.add_argument("--target_w", type=int, default=0)
    p.add_argument("--use_sdf", type=int, default=1)
    p.add_argument("--resize_mode", type=str, default="pad", choices=["pad", "stretch"])
    p.add_argument("--pad_value", type=float, default=1.0)
    p.add_argument("--pad_scale_mode", type=str, default="none", choices=["none", "fit"])
    args = p.parse_args()

    target_hw = None
    if args.target_h > 0 and args.target_w > 0:
        target_hw = (int(args.target_h), int(args.target_w))
    build_unified(
        inputs=args.inputs,
        out_dir=args.out_dir,
        target_hw=target_hw,
        use_sdf=bool(args.use_sdf),
        resize_mode=args.resize_mode,
        pad_value=float(args.pad_value),
        pad_scale_mode=args.pad_scale_mode,
    )


if __name__ == "__main__":
    main()
