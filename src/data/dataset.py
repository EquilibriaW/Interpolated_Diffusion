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


def _extract_maze_map(env):
    candidates = [env, getattr(env, "unwrapped", env)]
    for obj in candidates:
        if hasattr(obj, "get_maze_map"):
            maze_map = obj.get_maze_map()
            if maze_map is not None:
                return maze_map
        for attr in ["maze_map", "maze"]:
            if hasattr(obj, attr):
                maze_map = getattr(obj, attr)
                if hasattr(maze_map, "maze_map"):
                    maze_map = maze_map.maze_map
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
        if any(k in name_l for k in ["wall", "block", "maze"]) and not any(k in name_l for k in ["floor", "ground"]):
            wall_ids.append(i)
    if len(wall_ids) == 0:
        for i in range(model.ngeom):
            name = _get_geom_name(model, i) or ""
            name_l = name.lower()
            if any(k in name_l for k in ["floor", "ground"]):
                continue
            if int(geom_type[i]) == box_type:
                wall_ids.append(i)

    walls: List[np.ndarray] = []
    for i in wall_ids:
        if int(geom_type[i]) != box_type:
            continue
        size = np.array(geom_size[i], dtype=np.float32)
        pos = np.array(geom_pos[i], dtype=np.float32)
        quat = np.array(geom_quat[i], dtype=np.float32)
        sx, sy = float(size[0]), float(size[1])
        if sx <= 0 or sy <= 0:
            continue
        corners = np.array(
            [[sx, sy, 0.0], [sx, -sy, 0.0], [-sx, -sy, 0.0], [-sx, sy, 0.0]], dtype=np.float32
        )
        rot = _quat_to_rotmat(quat)
        world = corners @ rot.T + pos[None, :]
        walls.append(world[:, :2])
    return walls if len(walls) > 0 else None


def _maze_map_to_occ(maze_map) -> np.ndarray:
    if isinstance(maze_map, np.ndarray):
        occ = (maze_map > 0).astype(np.float32)
        return occ
    if isinstance(maze_map, (list, tuple)) and len(maze_map) > 0:
        if isinstance(maze_map[0], (list, tuple, np.ndarray)):
            occ = (np.array(maze_map) > 0).astype(np.float32)
            return occ
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


def _resample_sequence(seq: torch.Tensor, T: int) -> torch.Tensor:
    L = seq.shape[0]
    if L == T:
        return seq
    if L == 1:
        return seq.repeat(T, 1)
    idx = torch.linspace(0, L - 1, T, device=seq.device)
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
        self.goal_arr = self._extract_goal_array(dataset)
        self.pos_low, self.pos_high = self._infer_pos_bounds(env, occ)
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

    def _infer_pos_bounds(self, env, occ: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        low = None
        high = None
        try:
            obs_space = env.observation_space
            low = obs_space.low[:2]
            high = obs_space.high[:2]
            if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
                low, high = None, None
        except Exception:
            low, high = None, None
        if low is None or high is None:
            scale = getattr(getattr(env, "unwrapped", env), "maze_size_scaling", None)
            if scale is None:
                scale = getattr(getattr(env, "unwrapped", env), "maze_size_scale", None)
            if scale is not None and occ is not None:
                h, w = occ.shape
                low = np.array([0.0, 0.0], dtype=np.float32)
                high = np.array([w * float(scale), h * float(scale)], dtype=np.float32)
            else:
                low = self.obs[:, :2].min(axis=0)
                high = self.obs[:, :2].max(axis=0)
        low_t = torch.from_numpy(low.astype(np.float32))
        high_t = torch.from_numpy(high.astype(np.float32))
        return low_t, high_t

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        pos = obs[:, :2]
        pos = (pos - self.pos_low) / self.pos_scale
        if self.flip_y:
            pos[:, 1] = 1.0 - pos[:, 1]
        if not self.with_velocity:
            return pos
        vel = obs[:, 2:4] if obs.shape[1] >= 4 else torch.zeros_like(pos)
        vel = vel / self.pos_scale
        if self.flip_y:
            vel[:, 1] = -vel[:, 1]
        return torch.cat([pos, vel], dim=-1)

    def __getitem__(self, idx: int):
        gen = torch.Generator()
        gen.manual_seed(self.seed + idx)
        ep_idx = int(torch.randint(0, len(self.episodes), (1,), generator=gen).item())
        start, end = self.episodes[ep_idx]
        length = end - start
        base_idx = start
        if length >= self.T:
            offset = int(torch.randint(0, length - self.T + 1, (1,), generator=gen).item())
            base_idx = start + offset
            obs = self.obs[base_idx : base_idx + self.T]
        else:
            obs = self.obs[start:end]
        obs_t = torch.from_numpy(obs)
        if obs_t.shape[0] != self.T:
            obs_t = _resample_sequence(obs_t, self.T)
        x = self._normalize_obs(obs_t)

        start_pos = x[0, :2]
        if self.goal_arr is not None:
            goal_raw = self.goal_arr[base_idx]
            goal_t = torch.from_numpy(goal_raw)
            goal_t = (goal_t - self.pos_low) / self.pos_scale
            if self.flip_y:
                goal_t[1] = 1.0 - goal_t[1]
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
