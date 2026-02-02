import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .maze import generate_maze, sdf_from_occupancy
from .trajectories import path_to_trajectory


def _cell_to_xy(cell, h, w):
    i, j = cell
    return np.array([(j + 0.5) / w, (i + 0.5) / h], dtype=np.float32)


def _maybe_import_d4rl():
    mujoco_candidates = [
        os.environ.get("MUJOCO_PY_MUJOCO_PATH", ""),
        "/workspace/mujoco210",
        os.path.expanduser("~/.mujoco/mujoco210"),
    ]
    for candidate in mujoco_candidates:
        if candidate and os.path.isdir(candidate):
            os.environ.setdefault("MUJOCO_PY_MUJOCO_PATH", candidate)
            bin_path = os.path.join(candidate, "bin")
            ld_paths = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
            if bin_path not in ld_paths:
                ld_paths.append(bin_path)
                os.environ["LD_LIBRARY_PATH"] = ":".join(ld_paths)
            break
    try:
        import gym
        import d4rl  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("D4RL is not installed. Install it to use the D4RL dataset.") from exc
    return gym


def _parse_maze_spec(maze_str: str) -> np.ndarray:
    lines = maze_str.strip().split("\\")
    width, height = len(lines), len(lines[0])
    arr = np.zeros((width, height), dtype=np.int32)
    for w in range(width):
        row = lines[w]
        for h in range(height):
            tile = row[h]
            if tile == "#":
                arr[w, h] = 10
            elif tile == "G":
                arr[w, h] = 12
            else:
                arr[w, h] = 11
    return arr


def _extract_maze_map(env):
    candidates = [env, getattr(env, "unwrapped", env)]
    for obj in candidates:
        if hasattr(obj, "get_maze_map"):
            maze_map = obj.get_maze_map()
            if maze_map is not None:
                return maze_map
        for attr in ["maze_arr", "maze_map", "maze", "str_maze_spec", "maze_spec"]:
            if hasattr(obj, attr):
                maze_map = getattr(obj, attr)
                if hasattr(maze_map, "maze_map"):
                    maze_map = maze_map.maze_map
                if isinstance(maze_map, str):
                    return _parse_maze_spec(maze_map)
                return maze_map
    return None


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _get_geom_name(model, geom_id: int) -> Optional[str]:
    try:
        name = model.geom_names[geom_id]
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        return name
    except Exception:
        pass
    try:
        import mujoco  # type: ignore

        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        return name
    except Exception:
        return None


def _extract_mujoco_walls(env) -> Optional[List[np.ndarray]]:
    sim = getattr(env, "sim", None)
    if sim is None and hasattr(env, "unwrapped"):
        sim = getattr(env.unwrapped, "sim", None)
    model = None
    if sim is not None:
        model = getattr(sim, "model", None)
    if model is None and hasattr(env, "model"):
        model = getattr(env, "model", None)
    if model is None:
        return None

    geom_type = getattr(model, "geom_type", None)
    geom_size = getattr(model, "geom_size", None)
    geom_pos = getattr(model, "geom_pos", None)
    geom_quat = getattr(model, "geom_quat", None)
    if geom_type is None or geom_size is None or geom_pos is None or geom_quat is None:
        return None

    try:
        import mujoco_py  # type: ignore

        box_type = mujoco_py.generated.const.GEOM_BOX
    except Exception:
        box_type = 6

    wall_ids: List[int] = []
    for i in range(model.ngeom):
        name = _get_geom_name(model, i) or ""
        name_l = name.lower()
        if any(k in name_l for k in ["wall", "block", "maze", "obstacle"]) and not any(
            k in name_l for k in ["floor", "ground", "plane", "base"]
        ):
            wall_ids.append(i)
    if len(wall_ids) == 0:
        for i in range(model.ngeom):
            name = _get_geom_name(model, i) or ""
            name_l = name.lower()
            if any(k in name_l for k in ["floor", "ground", "plane", "base"]):
                continue
            if int(geom_type[i]) == box_type:
                wall_ids.append(i)

    candidates: List[Tuple[np.ndarray, float, float]] = []
    max_height = 0.0
    for i in wall_ids:
        if int(geom_type[i]) != box_type:
            continue
        size = np.array(geom_size[i], dtype=np.float32)
        if size.shape[0] < 3:
            continue
        pos = np.array(geom_pos[i], dtype=np.float32)
        quat = np.array(geom_quat[i], dtype=np.float32)
        sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
        if sx <= 0 or sy <= 0:
            continue
        max_height = max(max_height, sz)
        corners = np.array(
            [[sx, sy, 0.0], [sx, -sy, 0.0], [-sx, -sy, 0.0], [-sx, sy, 0.0]], dtype=np.float32
        )
        rot = _quat_to_rotmat(quat)
        world = corners @ rot.T + pos[None, :]
        area = 4.0 * sx * sy
        candidates.append((world[:, :2], area, sz))

    if len(candidates) == 0:
        return None

    # Drop very thin boxes (likely the floor).
    if max_height > 0:
        min_height = 0.05 * max_height
        filtered = [c for c in candidates if c[2] >= min_height]
        if len(filtered) > 0:
            candidates = filtered

    # Drop extremely large boxes compared to the median area (likely floor/ground plane).
    areas = np.array([c[1] for c in candidates], dtype=np.float32)
    if areas.size > 0:
        med = float(np.median(areas))
        if med > 0:
            keep = areas <= (6.0 * med)
            if np.any(keep):
                candidates = [c for c, k in zip(candidates, keep.tolist()) if k]

    walls = [c[0] for c in candidates]
    return walls if len(walls) > 0 else None


def _maze_map_to_occ(maze_map) -> np.ndarray:
    if isinstance(maze_map, np.ndarray):
        arr = np.asarray(maze_map)
        if arr.ndim != 2:
            raise ValueError("Unsupported maze_map format")
        uniq = np.unique(arr)
        if set(uniq.tolist()).issubset({0, 1}):
            return (arr > 0).astype(np.float32)
        if set(uniq.tolist()).issubset({10, 11, 12}):
            # D4RL pointmaze uses WALL=10, EMPTY=11, GOAL=12 with arr indexed as [x, y].
            return (arr == 10).astype(np.float32).T
        return (arr > 0).astype(np.float32)
    if isinstance(maze_map, (list, tuple)) and len(maze_map) > 0:
        if isinstance(maze_map[0], (list, tuple, np.ndarray)):
            arr = np.array(maze_map)
            if arr.ndim != 2:
                raise ValueError("Unsupported maze_map format")
            uniq = np.unique(arr)
            if set(uniq.tolist()).issubset({0, 1}):
                return (arr > 0).astype(np.float32)
            if set(uniq.tolist()).issubset({10, 11, 12}):
                return (arr == 10).astype(np.float32).T
            return (arr > 0).astype(np.float32)
        if isinstance(maze_map[0], str):
            h = len(maze_map)
            w = len(maze_map[0])
            occ = np.zeros((h, w), dtype=np.float32)
            wall_chars = {"#", "1", "X"}
            for i, row in enumerate(maze_map):
                for j, ch in enumerate(row):
                    if ch in wall_chars:
                        occ[i, j] = 1.0
            return occ
    raise ValueError("Unsupported maze_map format")


def _resample_sequence(seq: torch.Tensor, T: int, mode: str = "linear") -> torch.Tensor:
    L = seq.shape[0]
    if L == T:
        return seq
    if L == 1:
        return seq.repeat(T, 1)
    idx = torch.linspace(0, L - 1, T, device=seq.device)
    if mode == "nearest":
        idx_n = torch.round(idx).long()
        idx_n = torch.clamp(idx_n, 0, L - 1)
        return seq[idx_n]
    idx0 = torch.floor(idx).long()
    idx1 = torch.clamp(idx0 + 1, max=L - 1)
    w = (idx - idx0.float()).unsqueeze(-1)
    return seq[idx0] + w * (seq[idx1] - seq[idx0])


class ParticleMazeDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 100000,
        h: int = 21,
        w: int = 21,
        T: int = 64,
        p_wall_min: float = 0.15,
        p_wall_max: float = 0.30,
        with_velocity: bool = False,
        use_sdf: bool = False,
        cache_dir: Optional[str] = None,
        shard_size: int = 10000,
        seed: int = 123,
    ):
        self.num_samples = num_samples
        self.h = h
        self.w = w
        self.T = T
        self.p_wall_min = p_wall_min
        self.p_wall_max = p_wall_max
        self.with_velocity = with_velocity
        self.use_sdf = use_sdf
        self.cache_dir = cache_dir
        self.shard_size = shard_size
        self.seed = seed
        self._cache = {"idx": None, "data": None}
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return self.num_samples

    def _shard_path(self, shard_idx: int) -> str:
        return os.path.join(self.cache_dir, f"shard_{shard_idx:05d}.npz")

    def _generate_sample(self, rng: np.random.RandomState):
        p_wall = rng.uniform(self.p_wall_min, self.p_wall_max)
        occ, start, goal, path = generate_maze(rng, self.h, self.w, p_wall=p_wall)
        x = path_to_trajectory(path, self.h, self.w, self.T, with_velocity=self.with_velocity)
        sdf = None
        if self.use_sdf:
            sdf = sdf_from_occupancy(occ).astype(np.float32)
        start_xy = _cell_to_xy(start, self.h, self.w)
        goal_xy = _cell_to_xy(goal, self.h, self.w)
        start_goal = np.concatenate([start_xy, goal_xy], axis=0).astype(np.float32)
        return x, occ.astype(np.float32), sdf, start_goal

    def _build_shard(self, shard_idx: int) -> Dict[str, np.ndarray]:
        rng = np.random.RandomState(self.seed + shard_idx)
        start = shard_idx * self.shard_size
        end = min(self.num_samples, start + self.shard_size)
        n = end - start
        D = 4 if self.with_velocity else 2
        x_arr = np.zeros((n, self.T, D), dtype=np.float32)
        occ_arr = np.zeros((n, 1, self.h, self.w), dtype=np.float32)
        sdf_arr = None
        if self.use_sdf:
            sdf_arr = np.zeros((n, 1, self.h, self.w), dtype=np.float32)
        sg_arr = np.zeros((n, 4), dtype=np.float32)
        for i in range(n):
            x, occ, sdf, start_goal = self._generate_sample(rng)
            x_arr[i] = x
            occ_arr[i, 0] = occ
            if self.use_sdf and sdf is not None:
                sdf_arr[i, 0] = sdf
            sg_arr[i] = start_goal
        data = {"x": x_arr, "occ": occ_arr, "start_goal": sg_arr}
        if self.use_sdf and sdf_arr is not None:
            data["sdf"] = sdf_arr
        return data

    def _load_shard(self, shard_idx: int) -> Dict[str, np.ndarray]:
        if self._cache["idx"] == shard_idx:
            return self._cache["data"]
        data = None
        if self.cache_dir is not None:
            path = self._shard_path(shard_idx)
            if os.path.exists(path):
                with np.load(path) as f:
                    data = {k: f[k] for k in f.files}
            else:
                data = self._build_shard(shard_idx)
                np.savez_compressed(path, **data)
        else:
            data = self._build_shard(shard_idx)
        self._cache["idx"] = shard_idx
        self._cache["data"] = data
        return data

    def __getitem__(self, idx: int):
        shard_idx = idx // self.shard_size
        offset = idx % self.shard_size
        data = self._load_shard(shard_idx)
        x = data["x"][offset]
        occ = data["occ"][offset]
        start_goal = data["start_goal"][offset]
        sdf = data.get("sdf")
        if sdf is not None:
            sdf = sdf[offset]
        sample = {
            "x": torch.from_numpy(x),
            "cond": {
                "occ": torch.from_numpy(occ),
                "start_goal": torch.from_numpy(start_goal),
            },
        }
        if sdf is not None:
            sample["cond"]["sdf"] = torch.from_numpy(sdf)
        return sample


class D4RLMazeDataset(Dataset):
    def __init__(
        self,
        env_id: str = "maze2d-medium-v1",
        num_samples: int = 100000,
        T: int = 64,
        with_velocity: bool = False,
        use_sdf: bool = False,
        seed: int = 123,
        flip_y: bool = True,
        swap_xy: bool = False,
        max_collision_rate: Optional[float] = None,
        max_resample_tries: int = 50,
        min_goal_dist: Optional[float] = None,
        min_path_len: Optional[float] = None,
        window_mode: str = "end",
        goal_mode: str = "env",
        min_tortuosity: Optional[float] = None,
        min_turns: Optional[int] = None,
        turn_angle_deg: float = 30.0,
        episode_split_mod: Optional[int] = None,
        episode_split_val: int = 0,
        require_accept: bool = False,
    ):
        gym = _maybe_import_d4rl()
        env = gym.make(env_id)
        dataset = env.get_dataset()

        self.env_id = env_id
        self.num_samples = num_samples
        self.T = T
        self.with_velocity = with_velocity
        self.use_sdf = use_sdf
        self.seed = seed
        self.flip_y = flip_y
        self.swap_xy = swap_xy
        self.max_collision_rate = max_collision_rate
        self.max_resample_tries = max_resample_tries
        self.min_goal_dist = min_goal_dist
        self.min_path_len = min_path_len
        self.window_mode = window_mode
        self.goal_mode = goal_mode
        self.min_tortuosity = min_tortuosity
        self.min_turns = min_turns
        self.turn_angle_deg = turn_angle_deg
        self.episode_split_mod = episode_split_mod
        self.episode_split_val = episode_split_val
        self.require_accept = bool(require_accept)

        self.obs = dataset["observations"].astype(np.float32)
        terminals = dataset.get("terminals")
        timeouts = dataset.get("timeouts")
        if terminals is None:
            terminals = dataset.get("dones")
        terminals = terminals.astype(np.bool_) if terminals is not None else np.zeros(len(self.obs), dtype=np.bool_)
        timeouts = timeouts.astype(np.bool_) if timeouts is not None else np.zeros(len(self.obs), dtype=np.bool_)
        done = terminals | timeouts
        self.episodes = []
        start = 0
        for i in range(len(done)):
            if done[i]:
                self.episodes.append((start, i + 1))
                start = i + 1
        if start < len(done):
            self.episodes.append((start, len(done)))
        if len(self.episodes) == 0:
            raise RuntimeError("No episodes found in D4RL dataset")
        if self.episode_split_mod is not None:
            mod = int(self.episode_split_mod)
            val = int(self.episode_split_val)
            if mod <= 0:
                raise ValueError("episode_split_mod must be > 0")
            self.episodes = [ep for i, ep in enumerate(self.episodes) if (i % mod) == val]
            if len(self.episodes) == 0:
                raise RuntimeError("No episodes left after split filter")

        maze_map = _extract_maze_map(env)
        self.maze_map = maze_map
        self.maze_size_scaling = getattr(getattr(env, "unwrapped", env), "maze_size_scaling", None)
        if self.maze_size_scaling is None:
            self.maze_size_scaling = getattr(getattr(env, "unwrapped", env), "maze_size_scale", None)
        self.mj_walls = _extract_mujoco_walls(env)
        if maze_map is None:
            warnings.warn("Could not extract maze_map from env; using empty occupancy grid.")
            occ = np.zeros((21, 21), dtype=np.float32)
        else:
            occ = _maze_map_to_occ(maze_map)
        if self.flip_y and occ is not None:
            occ = np.flipud(occ).copy()
        self.goal_arr = self._extract_goal_array(dataset)
        # Use occupancy-grid bounds for normalization to keep coordinates aligned with occ.
        self.pos_low, self.pos_high = self._infer_pos_bounds(env, occ, None)
        self.pos_scale = torch.clamp(self.pos_high - self.pos_low, min=1e-6)
        self.occ = torch.from_numpy(occ).unsqueeze(0)
        if self.use_sdf:
            sdf = sdf_from_occupancy(occ).astype(np.float32)
            self.sdf = torch.from_numpy(sdf).unsqueeze(0)
        else:
            self.sdf = None

        env.close()

    def __len__(self):
        return self.num_samples

    def _extract_goal_array(self, dataset) -> Optional[np.ndarray]:
        keys = ["infos/goal", "infos/desired_goal", "goal", "desired_goal"]
        for key in keys:
            if key in dataset:
                arr = dataset[key]
                if arr is None:
                    continue
                arr = np.asarray(arr)
                if arr.ndim >= 2 and arr.shape[-1] >= 2:
                    return arr[:, :2].astype(np.float32)
        return None

    def _infer_pos_bounds(
        self,
        env,
        occ: np.ndarray,
        walls: Optional[List[np.ndarray]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        low = None
        high = None
        if walls:
            mins = []
            maxs = []
            for poly in walls:
                arr = np.asarray(poly, dtype=np.float32)
                if arr.ndim != 2 or arr.shape[1] < 2:
                    continue
                mins.append(arr[:, :2].min(axis=0))
                maxs.append(arr[:, :2].max(axis=0))
            if len(mins) > 0:
                low = np.min(np.stack(mins, axis=0), axis=0)
                high = np.max(np.stack(maxs, axis=0), axis=0)

        if low is None or high is None:
            scale = getattr(getattr(env, "unwrapped", env), "maze_size_scaling", None)
            if scale is None:
                scale = getattr(getattr(env, "unwrapped", env), "maze_size_scale", None)
            if scale is None:
                scale = 1.0
            if occ is not None:
                h, w = occ.shape
                if np.any(occ > 0):
                    # D4RL pointmaze observations are in grid index coordinates (0..W-1),
                    # not offset by +0.5 like the wall geom centers.
                    low = np.array([0.0, 0.0], dtype=np.float32)
                    high = np.array([(w - 1) * float(scale), (h - 1) * float(scale)], dtype=np.float32)

        if low is None or high is None:
            try:
                obs_space = env.observation_space
                low = obs_space.low[:2]
                high = obs_space.high[:2]
                if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
                    low, high = None, None
            except Exception:
                low, high = None, None

        if low is None or high is None:
            low = self.obs[:, :2].min(axis=0)
            high = self.obs[:, :2].max(axis=0)

        low_t = torch.from_numpy(np.asarray(low, dtype=np.float32))
        high_t = torch.from_numpy(np.asarray(high, dtype=np.float32))
        return low_t, high_t

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        pos = obs[:, :2]
        pos = (pos - self.pos_low) / self.pos_scale
        if self.flip_y:
            pos[:, 1] = 1.0 - pos[:, 1]
        if self.swap_xy:
            pos = pos[:, [1, 0]]
        if not self.with_velocity:
            return pos
        vel = obs[:, 2:4] if obs.shape[1] >= 4 else torch.zeros_like(pos)
        vel = vel / self.pos_scale
        if self.flip_y:
            vel[:, 1] = -vel[:, 1]
        if self.swap_xy:
            vel = vel[:, [1, 0]]
        return torch.cat([pos, vel], dim=-1)

    def __getitem__(self, idx: int):
        from src.eval.metrics import collision_rate

        def _draw(gen_local: torch.Generator):
            ep_idx = int(torch.randint(0, len(self.episodes), (1,), generator=gen_local).item())
            start, end = self.episodes[ep_idx]
            length = end - start
            if self.window_mode == "episode":
                base_idx = start
                obs = self.obs[start:end]
            elif length >= self.T:
                if self.window_mode == "random":
                    max_start = end - self.T
                    offset = int(torch.randint(0, max_start - start + 1, (1,), generator=gen_local).item())
                    base_idx = start + offset
                    obs = self.obs[base_idx : base_idx + self.T]
                else:
                    base_idx = end - self.T
                    obs = self.obs[base_idx:end]
            else:
                base_idx = start
                obs = self.obs[start:end]
            end_idx = end - 1
            obs_t = torch.from_numpy(obs)
            if obs_t.shape[0] != self.T:
                obs_t = _resample_sequence(obs_t, self.T, mode="nearest")
            x = self._normalize_obs(obs_t)

            start_pos = x[0, :2]
            if self.goal_mode == "window_end":
                goal_t = x[-1, :2]
            elif self.goal_arr is not None:
                goal_raw = self.goal_arr[end_idx]
                goal_t = torch.from_numpy(goal_raw)
                goal_t = (goal_t - self.pos_low) / self.pos_scale
                if self.flip_y:
                    goal_t[1] = 1.0 - goal_t[1]
                if self.swap_xy:
                    goal_t = goal_t[[1, 0]]
            else:
                goal_t = x[-1, :2]
            start_goal = torch.cat([start_pos, goal_t], dim=0)

            sample = {
                "x": x,
                "cond": {
                    "occ": self.occ,
                    "start_goal": start_goal,
                },
            }
            if self.sdf is not None:
                sample["cond"]["sdf"] = self.sdf
            return sample

        def _check(sample):
            stats = {}
            accept = True
            if self.max_collision_rate is not None:
                coll = collision_rate(sample["cond"]["occ"][0], sample["x"])
                stats["collision_rate"] = float(coll)
                if coll > float(self.max_collision_rate):
                    accept = False
            if accept and self.min_goal_dist is not None:
                start_goal = sample["cond"]["start_goal"]
                sg = start_goal.clone()
                sg[:2] = sg[:2] * self.pos_scale[:2] + self.pos_low[:2]
                sg[2:] = sg[2:] * self.pos_scale[:2] + self.pos_low[:2]
                goal_dist = torch.norm(sg[:2] - sg[2:]).item()
                stats["goal_dist"] = float(goal_dist)
                if goal_dist < float(self.min_goal_dist):
                    accept = False
            if accept and self.min_path_len is not None:
                traj = sample["x"][:, :2] * self.pos_scale[:2] + self.pos_low[:2]
                path_len = torch.norm(traj[1:] - traj[:-1], dim=-1).sum().item()
                stats["path_len"] = float(path_len)
                if path_len < float(self.min_path_len):
                    accept = False
            if accept and (self.min_tortuosity is not None or self.min_turns is not None):
                traj = sample["x"][:, :2] * self.pos_scale[:2] + self.pos_low[:2]
                diffs = traj[1:] - traj[:-1]
                seg_lens = torch.norm(diffs, dim=-1)
                path_len = seg_lens.sum().item()
                straight = torch.norm(traj[-1] - traj[0]).item()
                if self.min_tortuosity is not None:
                    tort = path_len / max(straight, 1e-6)
                    stats["tortuosity"] = float(tort)
                    if tort < float(self.min_tortuosity):
                        accept = False
                if accept and self.min_turns is not None:
                    v1 = diffs[:-1]
                    v2 = diffs[1:]
                    n1 = torch.norm(v1, dim=-1)
                    n2 = torch.norm(v2, dim=-1)
                    valid = (n1 > 1e-6) & (n2 > 1e-6)
                    if valid.any():
                        v1n = v1[valid] / n1[valid].unsqueeze(-1)
                        v2n = v2[valid] / n2[valid].unsqueeze(-1)
                        dots = (v1n * v2n).sum(dim=-1).clamp(-1.0, 1.0)
                        angles = torch.acos(dots)
                        thresh = float(self.turn_angle_deg) * np.pi / 180.0
                        turns = int((angles >= thresh).sum().item())
                    else:
                        turns = 0
                    stats["turns"] = int(turns)
                    if turns < int(self.min_turns):
                        accept = False
            return accept, stats

        gen = torch.Generator()
        gen.manual_seed(self.seed + idx)
        sample = _draw(gen)
        if (
            self.max_collision_rate is None
            and self.min_goal_dist is None
            and self.min_path_len is None
            and self.min_tortuosity is None
            and self.min_turns is None
        ):
            return sample

        last_stats = {}
        for attempt in range(self.max_resample_tries):
            accept, last_stats = _check(sample)
            if accept:
                return sample
            gen.manual_seed(self.seed + idx + (attempt + 1) * 1000003)
            sample = _draw(gen)
        if self.require_accept:
            raise RuntimeError(
                f"D4RLMazeDataset: failed to satisfy constraints after {self.max_resample_tries} tries. "
                f"stats={last_stats}"
            )
        return sample


class PreparedTrajectoryDataset(Dataset):
    def __init__(self, path: str, use_sdf: bool = False):
        data = np.load(path)
        self.x = data["x"].astype(np.float32)
        self.start_goal = data["start_goal"].astype(np.float32)
        occ = data["occ"].astype(np.float32)
        self.difficulty = data.get("difficulty")
        if occ.ndim == 2:
            occ = occ[None, ...]
        self.occ = occ
        self.occ_per_sample = occ.ndim == 3 and occ.shape[0] == self.x.shape[0]
        self.sdf = None
        if use_sdf and "sdf" in data:
            sdf = data["sdf"].astype(np.float32)
            if sdf.ndim == 2:
                sdf = sdf[None, ...]
            self.sdf = sdf
        self.sdf_per_sample = False
        if self.sdf is not None and self.sdf.ndim == 3 and self.sdf.shape[0] == self.x.shape[0]:
            self.sdf_per_sample = True
        self.kp_idx = data.get("kp_idx")
        if self.kp_idx is not None:
            self.kp_idx = self.kp_idx.astype(np.int64)
        self.kp_feat = data.get("kp_feat")
        if self.kp_feat is not None:
            self.kp_feat = self.kp_feat.astype(np.float32)
        self.kp_mask_levels = data.get("kp_mask_levels")
        if self.kp_mask_levels is not None:
            self.kp_mask_levels = self.kp_mask_levels.astype(np.bool_)
        self.kp_mask_levels_per_sample = False
        if self.kp_mask_levels is not None and self.kp_mask_levels.ndim == 3 and self.kp_mask_levels.shape[0] == self.x.shape[0]:
            self.kp_mask_levels_per_sample = True

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        occ = self.occ[idx] if self.occ_per_sample else self.occ
        if occ.ndim == 2:
            occ = occ[None, ...]
        sdf = None
        if self.sdf is not None:
            sdf = self.sdf[idx] if self.sdf_per_sample else self.sdf
            if sdf.ndim == 2:
                sdf = sdf[None, ...]
        sample = {
            "x": torch.from_numpy(self.x[idx]),
            "cond": {
                "occ": torch.from_numpy(occ),
                "start_goal": torch.from_numpy(self.start_goal[idx]),
            },
        }
        if self.kp_idx is not None:
            sample["cond"]["kp_idx"] = torch.from_numpy(self.kp_idx[idx])
        if self.kp_feat is not None:
            sample["cond"]["kp_feat"] = torch.from_numpy(self.kp_feat[idx])
        if self.kp_mask_levels is not None:
            if self.kp_mask_levels_per_sample:
                sample["cond"]["kp_mask_levels"] = torch.from_numpy(self.kp_mask_levels[idx])
            else:
                sample["cond"]["kp_mask_levels"] = torch.from_numpy(self.kp_mask_levels)
        if self.difficulty is not None:
            sample["difficulty"] = torch.tensor(int(self.difficulty[idx]), dtype=torch.int64)
        if sdf is not None:
            sample["cond"]["sdf"] = torch.from_numpy(sdf)
        return sample
