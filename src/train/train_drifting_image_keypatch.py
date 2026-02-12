from __future__ import annotations

import argparse
import collections
import os
import random
import time
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from src.losses.drifting import (
    drifting_loss_feature_group_class_batched,
)
from src.models.drifting_feature_encoder import FrozenResNetMultiFeature
from src.models.frame_vae import FrameAutoencoderKL
from src.models.image_patch_refiner import ImagePatchRefiner
from src.models.latent_mae_feature_encoder import FrozenLatentMAEMultiFeature
from src.optim import build_optimizer
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.image_patches import (
    build_nested_patch_masks_batch,
    interpolate_patch_tokens_idw,
    interpolate_patch_tokens_laplacian,
    patchify_images,
    unpatchify_images,
)
from src.utils.logging import create_writer
from src.utils.seed import set_seed


def _parse_temps(s: str) -> Tuple[float, ...]:
    vals = [float(x.strip()) for x in str(s).split(",") if x.strip()]
    if not vals:
        raise ValueError("temperatures cannot be empty")
    return tuple(vals)


def _parse_floats(s: str) -> Tuple[float, ...]:
    vals = [float(x.strip()) for x in str(s).split(",") if x.strip()]
    if not vals:
        raise ValueError("value list cannot be empty")
    return tuple(vals)


class _NullWriter:
    def add_scalar(self, *args, **kwargs) -> None:
        return None

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


def _init_distributed() -> Tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_ddp = world_size > 1
    if use_ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return use_ddp, rank, local_rank, world_size, device


class ClassBalancedBatchSampler(Sampler[List[int]]):
    """Yield class-balanced batches: n_classes * n_samples_per_class."""

    def __init__(
        self,
        labels: Sequence[int],
        n_classes: int,
        n_samples_per_class: int,
        num_steps: int,
        seed: int = 0,
    ) -> None:
        self.labels = list(int(x) for x in labels)
        self.n_classes = int(n_classes)
        self.n_samples_per_class = int(n_samples_per_class)
        self.num_steps = int(num_steps)
        if self.n_classes < 1 or self.n_samples_per_class < 1:
            raise ValueError("n_classes and n_samples_per_class must be >= 1")
        self.class_to_idx: Dict[int, List[int]] = collections.defaultdict(list)
        for i, y in enumerate(self.labels):
            self.class_to_idx[y].append(i)
        self.classes = sorted(self.class_to_idx.keys())
        if not self.classes:
            raise ValueError("empty labels")
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.num_steps

    def __iter__(self) -> Iterable[List[int]]:
        rng = random.Random(self.seed)
        for _ in range(self.num_steps):
            chosen_classes = [self.classes[rng.randrange(len(self.classes))] for _ in range(self.n_classes)]
            batch: List[int] = []
            for c in chosen_classes:
                idxs = self.class_to_idx[c]
                if len(idxs) >= self.n_samples_per_class:
                    picks = rng.sample(idxs, self.n_samples_per_class)
                else:
                    picks = [idxs[rng.randrange(len(idxs))] for _ in range(self.n_samples_per_class)]
                batch.extend(picks)
            rng.shuffle(batch)
            yield batch


@dataclass
class QueueStats:
    non_empty: int
    mean_occupancy: float
    median_occupancy: float
    max_fraction: float
    entropy: float


class PerClassFeatureQueue:
    """CPU-side per-class queue of feature vectors for one feature-space family."""

    def __init__(self, num_classes: int, n_features: int, maxlen: int = 128) -> None:
        self.num_classes = int(num_classes)
        self.n_features = int(n_features)
        self.maxlen = int(maxlen)
        self._q: List[List[Deque[torch.Tensor]]] = [
            [collections.deque(maxlen=self.maxlen) for _ in range(self.n_features)] for _ in range(self.num_classes)
        ]

    def push(self, class_ids: torch.Tensor, feats: Sequence[torch.Tensor]) -> None:
        if len(feats) != self.n_features:
            raise ValueError("feature count mismatch")
        cid = class_ids.detach().cpu().long().tolist()
        for b, c in enumerate(cid):
            if c < 0 or c >= self.num_classes:
                continue
            for j in range(self.n_features):
                self._q[c][j].append(feats[j][b].detach().cpu())

    def sample(self, class_id: int, feat_idx: int, k: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        q = self._q[int(class_id)][int(feat_idx)]
        if len(q) == 0 or k <= 0:
            return torch.empty((0, 0), device=device, dtype=dtype)
        m = len(q)
        picks = random.sample(range(m), min(m, k))
        vals = [q[i] for i in picks]
        out = torch.stack(vals, dim=0).to(device=device, dtype=dtype, non_blocking=True)
        return out

    def class_occupancy(self, feat_idx: int = 0) -> List[int]:
        return [len(self._q[c][feat_idx]) for c in range(self.num_classes)]

    def stats(self, feat_idx: int = 0) -> QueueStats:
        occ = np.array(self.class_occupancy(feat_idx=feat_idx), dtype=np.float64)
        non_empty = int((occ > 0).sum())
        mean_occ = float(occ.mean()) if occ.size else 0.0
        median_occ = float(np.median(occ)) if occ.size else 0.0
        total = float(occ.sum())
        if total > 0:
            p = occ / total
            nz = p[p > 0]
            entropy = float(-(nz * np.log(nz)).sum())
            max_fraction = float(p.max())
        else:
            entropy = 0.0
            max_fraction = 0.0
        return QueueStats(
            non_empty=non_empty,
            mean_occupancy=mean_occ,
            median_occupancy=median_occ,
            max_fraction=max_fraction,
            entropy=entropy,
        )


def _safe_import_sklearn():
    try:
        from sklearn.metrics import davies_bouldin_score, silhouette_score
    except Exception:
        return None, None
    return silhouette_score, davies_bouldin_score


def separation_metrics(feat: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Simple class-separation diagnostics on one feature tensor."""
    out: Dict[str, float] = {}
    if feat.numel() == 0 or feat.shape[0] < 3:
        return out
    x = feat.detach().float()
    y = labels.detach()
    d = torch.cdist(x, x)
    same = y[:, None].eq(y[None, :])
    eye = torch.eye(x.shape[0], device=x.device, dtype=torch.bool)
    same = same & (~eye)
    diff = ~same & (~eye)
    if same.any():
        intra = d[same].mean().item()
    else:
        intra = float("nan")
    if diff.any():
        inter = d[diff].mean().item()
    else:
        inter = float("nan")
    ratio = float(intra / max(inter, 1e-8)) if (np.isfinite(intra) and np.isfinite(inter)) else float("nan")
    out["sep/intra_mean"] = float(intra) if np.isfinite(intra) else 0.0
    out["sep/inter_mean"] = float(inter) if np.isfinite(inter) else 0.0
    out["sep/intra_inter_ratio"] = float(ratio) if np.isfinite(ratio) else 0.0

    sil_fn, db_fn = _safe_import_sklearn()
    if sil_fn is not None and db_fn is not None:
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        uniq = np.unique(y_np)
        if uniq.size >= 2 and x_np.shape[0] > uniq.size:
            try:
                out["sep/silhouette"] = float(sil_fn(x_np, y_np))
            except Exception:
                pass
            try:
                out["sep/davies_bouldin"] = float(db_fn(x_np, y_np))
            except Exception:
                pass
    return out


def _feature_group_key(name: str) -> str:
    loc_tag = "_loc_"
    if loc_tag in name:
        return name.split(loc_tag, 1)[0]
    return name


def _sample_alpha_per_class(
    class_ids: Sequence[int],
    alpha_values: Sequence[float],
    alpha_probs: Sequence[float] | None = None,
) -> List[float]:
    if not class_ids:
        return []
    if alpha_probs is None:
        return [float(random.choice(alpha_values)) for _ in class_ids]
    return [float(random.choices(alpha_values, weights=alpha_probs, k=1)[0]) for _ in class_ids]


def _pack_feature_class_batch(
    x_feat: torch.Tensor,
    x_feat_det: torch.Tensor,
    feats_real_j: torch.Tensor,
    class_ids: Sequence[int],
    class_sel: Sequence[torch.Tensor],
    pos_queue: PerClassFeatureQueue,
    neg_queue: PerClassFeatureQueue,
    unc_queue: PerClassFeatureQueue | None,
    feat_idx: int,
    n_pos: int,
    n_neg: int,
    n_unc: int,
    class_alpha: Sequence[float] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack ragged per-class tensors into padded batched tensors."""
    if len(class_ids) != len(class_sel):
        raise ValueError("class_ids/class_sel length mismatch")
    if class_alpha is not None and len(class_alpha) != len(class_ids):
        raise ValueError("class_alpha length must match class_ids")
    x_list: List[torch.Tensor] = []
    yp_list: List[torch.Tensor] = []
    yn_list: List[torch.Tensor] = []
    ynw_list: List[torch.Tensor] = []
    self_neg_k_list: List[int] = []
    for c_id, sel in zip(class_ids, class_sel):
        xf = x_feat[sel]
        xf_det = x_feat_det[sel].detach()
        yp = pos_queue.sample(
            class_id=int(c_id),
            feat_idx=feat_idx,
            k=int(n_pos),
            device=x_feat.device,
            dtype=xf.dtype,
        )
        yn = neg_queue.sample(
            class_id=int(c_id),
            feat_idx=feat_idx,
            k=int(n_neg),
            device=x_feat.device,
            dtype=xf.dtype,
        )
        if unc_queue is not None:
            yn_unc = unc_queue.sample(
                class_id=0,
                feat_idx=feat_idx,
                k=int(n_unc),
                device=x_feat.device,
                dtype=xf.dtype,
            )
        else:
            yn_unc = torch.empty((0, xf.shape[-1]), device=x_feat.device, dtype=xf.dtype)
        if yp.numel() == 0:
            yp = feats_real_j[sel].detach()
        if yn.numel() == 0:
            yn = xf_det.new_empty((0, xf_det.shape[-1]))
        cond_neg = torch.cat([xf_det, yn], dim=0)
        alpha_c = 1.0
        if class_alpha is not None:
            idx = len(x_list)
            alpha_c = float(class_alpha[idx])
        n_cond = max(1, int(cond_neg.shape[0]) - 1)
        n_unc_eff = int(yn_unc.shape[0])
        if n_unc_eff > 0 and alpha_c > 1.0:
            w_unc = (alpha_c - 1.0) * float(n_cond) / float(n_unc_eff)
        else:
            w_unc = 0.0
        if n_unc_eff > 0:
            y_neg = torch.cat([cond_neg, yn_unc], dim=0)
            y_neg_w = torch.cat(
                [
                    torch.ones((cond_neg.shape[0],), device=x_feat.device, dtype=xf.dtype),
                    torch.full((yn_unc.shape[0],), float(w_unc), device=x_feat.device, dtype=xf.dtype),
                ],
                dim=0,
            )
        else:
            y_neg = cond_neg
            y_neg_w = torch.ones((cond_neg.shape[0],), device=x_feat.device, dtype=xf.dtype)
        x_list.append(xf)
        yp_list.append(yp)
        yn_list.append(y_neg)
        ynw_list.append(y_neg_w)
        self_neg_k_list.append(int(xf.shape[0]))

    c = len(x_list)
    if c == 0:
        z = torch.empty((0, 0, 0), device=x_feat.device, dtype=x_feat.dtype)
        m = torch.empty((0, 0), device=x_feat.device, dtype=torch.bool)
        w = torch.empty((0, 0), device=x_feat.device, dtype=x_feat.dtype)
        k = torch.empty((0,), device=x_feat.device, dtype=torch.long)
        return z, m, z, m, z, m, w, k

    d = int(x_feat.shape[-1])
    n_max = max(int(v.shape[0]) for v in x_list)
    p_max = max(int(v.shape[0]) for v in yp_list)
    q_max = max(int(v.shape[0]) for v in yn_list)

    x_pad = torch.zeros((c, n_max, d), device=x_feat.device, dtype=x_feat.dtype)
    x_mask = torch.zeros((c, n_max), device=x_feat.device, dtype=torch.bool)
    yp_pad = torch.zeros((c, p_max, d), device=x_feat.device, dtype=x_feat.dtype)
    yp_mask = torch.zeros((c, p_max), device=x_feat.device, dtype=torch.bool)
    yn_pad = torch.zeros((c, q_max, d), device=x_feat.device, dtype=x_feat.dtype)
    yn_mask = torch.zeros((c, q_max), device=x_feat.device, dtype=torch.bool)
    ynw_pad = torch.zeros((c, q_max), device=x_feat.device, dtype=x_feat.dtype)

    for i, (xf, yp, yn, ynw) in enumerate(zip(x_list, yp_list, yn_list, ynw_list)):
        n = int(xf.shape[0])
        p = int(yp.shape[0])
        q = int(yn.shape[0])
        x_pad[i, :n] = xf
        x_mask[i, :n] = True
        yp_pad[i, :p] = yp
        yp_mask[i, :p] = True
        yn_pad[i, :q] = yn
        yn_mask[i, :q] = True
        ynw_pad[i, :q] = ynw
    self_neg_k = torch.tensor(self_neg_k_list, device=x_feat.device, dtype=torch.long)
    return x_pad, x_mask, yp_pad, yp_mask, yn_pad, yn_mask, ynw_pad, self_neg_k


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="imagefolder", choices=["imagefolder", "fake"])
    p.add_argument("--data_root", type=str, default="")
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--num_classes", type=int, default=1000)
    p.add_argument("--fake_num_samples", type=int, default=200000)
    p.add_argument("--input_space", type=str, default="pixel", choices=["pixel", "latent"])
    p.add_argument("--feature_space", type=str, default="resnet", choices=["resnet", "latent_mae"])
    p.add_argument("--latent_feature_ckpt", type=str, default="")
    p.add_argument("--latent_feature_base_width", type=int, default=256)
    p.add_argument("--vae_model", type=str, default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--vae_scale", type=float, default=0.18215)
    p.add_argument("--vae_use_mean", type=int, default=1)

    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon"])
    p.add_argument("--muon_lr", type=float, default=0.02)
    p.add_argument("--muon_momentum", type=float, default=0.95)
    p.add_argument("--muon_weight_decay", type=float, default=0.01)
    p.add_argument("--muon_adam_beta1", type=float, default=0.9)
    p.add_argument("--muon_adam_beta2", type=float, default=0.95)
    p.add_argument("--muon_adam_eps", type=float, default=1e-10)
    p.add_argument("--ddp_find_unused_params", type=int, default=0)

    p.add_argument("--n_classes_step", type=int, default=16)
    p.add_argument("--n_gen_per_class", type=int, default=4)

    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--K_min", type=int, default=32)
    p.add_argument("--levels", type=int, default=4)
    p.add_argument("--k_schedule", type=str, default="geom", choices=["doubling", "linear", "geom"])
    p.add_argument("--k_geom_gamma", type=float, default=None)
    p.add_argument("--interp_power", type=float, default=2.0)
    p.add_argument("--interp_mode", type=str, default="laplacian", choices=["idw", "laplacian"])
    p.add_argument("--interp_lap_sigma_feature", type=float, default=0.20)
    p.add_argument("--interp_lap_sigma_space", type=float, default=1.0)
    p.add_argument("--interp_lap_lambda_reg", type=float, default=1e-4)
    p.add_argument("--interp_lap_diag_neighbors", type=int, default=0)
    p.add_argument("--interp_lap_min_weight", type=float, default=1e-6)
    p.add_argument("--keep_anchors_fixed", type=int, default=0)

    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--feature_arch", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    p.add_argument("--feature_pretrained", type=int, default=1)
    p.add_argument("--feature_include_global", type=int, default=1)
    p.add_argument("--feature_include_input_sq", type=int, default=1)
    p.add_argument("--feature_include_local", type=int, default=0)
    p.add_argument("--feature_include_patch_stats", type=int, default=1)
    p.add_argument("--feature_max_locations", type=int, default=32)

    p.add_argument("--drift_n_pos", type=int, default=64)
    p.add_argument("--drift_n_neg", type=int, default=64)
    p.add_argument("--drift_n_unc", type=int, default=32)
    p.add_argument("--drift_taus", type=str, default="0.02,0.05,0.2")
    p.add_argument("--drift_norm_mode", type=str, default="both", choices=["none", "y", "both"])
    p.add_argument("--share_spatial_norm", type=int, default=1)
    p.add_argument("--cfg_alpha_values", type=str, default="1.0,2.0,4.0")
    p.add_argument("--cfg_alpha_probs", type=str, default="")
    p.add_argument("--vanilla_loss_weight", type=float, default=1.0)
    p.add_argument("--queue_pos", type=int, default=256)
    p.add_argument("--queue_neg", type=int, default=256)
    p.add_argument("--queue_unc", type=int, default=1000)
    p.add_argument("--amp_bf16", type=int, default=1)
    p.add_argument("--compile_drift", type=int, default=0)
    p.add_argument("--compile_mode", type=str, default="max-autotune-no-cudagraphs")

    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/drifting_image_keypatch")
    p.add_argument("--log_dir", type=str, default="runs/drifting_image_keypatch")
    p.add_argument("--resume", type=str, default="")
    return p


def _build_dataset(args: argparse.Namespace) -> Tuple[Dataset, int, Sequence[int]]:
    try:
        import torchvision
        import torchvision.transforms as T
    except Exception as exc:  # pragma: no cover
        raise ImportError("torchvision is required for image drifting training") from exc

    if args.dataset == "imagefolder":
        if not args.data_root:
            raise ValueError("--data_root is required for dataset=imagefolder")
        tfm = T.Compose(
            [
                T.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]
        )
        ds = torchvision.datasets.ImageFolder(args.data_root, transform=tfm)
        labels = list(int(y) for y in ds.targets)
        num_classes = len(ds.classes)
        return ds, num_classes, labels

    tfm = torchvision.transforms.ToTensor()
    ds = torchvision.datasets.FakeData(
        size=int(args.fake_num_samples),
        image_size=(3, args.image_size, args.image_size),
        num_classes=int(args.num_classes),
        transform=tfm,
        random_offset=int(args.seed),
    )
    labels = [int(ds[i][1]) for i in range(len(ds))]
    return ds, int(args.num_classes), labels


def main() -> None:
    args = build_parser().parse_args()
    use_ddp, rank, local_rank, world_size, device = _init_distributed()
    seed_rank = int(args.seed) + (rank * 100003)
    set_seed(seed_rank, deterministic=False)
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for drifting image training.")
    is_main = rank == 0
    if is_main:
        print(f"[drifting-image] device={device}")
        print(f"[drifting-image] cuda={torch.cuda.get_device_name(local_rank if use_ddp else 0)}")
        print(f"[drifting-image] ddp={int(use_ddp)} world_size={world_size}")

    ds, num_classes, labels = _build_dataset(args)
    batch_sampler = ClassBalancedBatchSampler(
        labels=labels,
        n_classes=int(args.n_classes_step),
        n_samples_per_class=int(args.n_gen_per_class),
        num_steps=int(args.steps) * int(args.grad_accum) * 2,  # allow resume offsets / exhausted iter guard
        seed=seed_rank,
    )
    loader = DataLoader(ds, batch_sampler=batch_sampler, num_workers=int(args.num_workers), pin_memory=True)
    it = iter(loader)

    batch_size_local = int(args.n_classes_step) * int(args.n_gen_per_class)
    if batch_size_local <= 0:
        raise ValueError("local batch size must be > 0")
    grad_accum = max(1, int(args.grad_accum))
    global_batch_effective = int(batch_size_local) * int(world_size) * int(grad_accum)
    if int(args.patch_size) <= 0:
        raise ValueError("patch_size must be > 0")
    input_channels = 3 if str(args.input_space) == "pixel" else 4
    if str(args.input_space) == "pixel":
        h, w = int(args.image_size), int(args.image_size)
        if (h % int(args.patch_size)) != 0 or (w % int(args.patch_size)) != 0:
            raise ValueError("image_size must be divisible by patch_size")
    data_dim = int(input_channels) * int(args.patch_size) * int(args.patch_size)

    model = ImagePatchRefiner(
        data_dim=data_dim,
        num_classes=num_classes,
        max_levels=int(args.levels),
        d_model=int(args.d_model),
        n_layers=int(args.n_layers),
        n_heads=int(args.n_heads),
        d_ff=int(args.d_ff),
        dropout=float(args.dropout),
    ).to(device)
    optimizer, opt_info = build_optimizer(
        model=model,
        optimizer_name=str(args.optimizer),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        muon_lr=float(args.muon_lr),
        muon_momentum=float(args.muon_momentum),
        muon_weight_decay=float(args.muon_weight_decay),
        muon_adam_betas=(float(args.muon_adam_beta1), float(args.muon_adam_beta2)),
        muon_adam_eps=float(args.muon_adam_eps),
    )
    if use_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=bool(args.ddp_find_unused_params),
            broadcast_buffers=False,
        )
    model_raw = model.module if isinstance(model, DDP) else model
    vae: FrameAutoencoderKL | None = None
    if str(args.input_space) == "latent":
        vae = FrameAutoencoderKL(
            model_name=str(args.vae_model),
            device=device,
            dtype=torch.float16,
            scale=float(args.vae_scale),
            use_mean=bool(args.vae_use_mean),
            freeze=True,
        )

    if str(args.feature_space) == "resnet":
        feature_encoder = FrozenResNetMultiFeature(
            arch=str(args.feature_arch),
            pretrained=bool(args.feature_pretrained),
            normalize_imagenet=True,
        ).to(device)
    else:
        if not args.latent_feature_ckpt:
            raise ValueError("--latent_feature_ckpt is required for feature_space=latent_mae")
        feature_encoder = FrozenLatentMAEMultiFeature(
            ckpt_path=str(args.latent_feature_ckpt),
            in_channels=4,
            base_width=int(args.latent_feature_base_width),
        ).to(device)
    feature_encoder.eval()

    if is_main:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        writer = create_writer(args.log_dir)
    else:
        writer = _NullWriter()
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model_raw, optimizer, ema=None, map_location=device)
    if use_ddp:
        start_t = torch.tensor([int(start_step)], device=device, dtype=torch.long)
        dist.broadcast(start_t, src=0)
        start_step = int(start_t.item())

    gen = torch.Generator(device=device)
    gen.manual_seed(seed_rank + 123)
    drift_taus = _parse_temps(args.drift_taus)
    alpha_values = _parse_floats(args.cfg_alpha_values)
    alpha_probs: Tuple[float, ...] | None = None
    if str(args.cfg_alpha_probs).strip():
        probs = _parse_floats(args.cfg_alpha_probs)
        if len(probs) != len(alpha_values):
            raise ValueError("--cfg_alpha_probs length must match --cfg_alpha_values length")
        p_sum = float(sum(probs))
        if p_sum <= 0.0:
            raise ValueError("--cfg_alpha_probs must have positive sum")
        alpha_probs = tuple(float(p / p_sum) for p in probs)

    share_spatial_norm = bool(args.share_spatial_norm)
    vanilla_loss_weight = float(args.vanilla_loss_weight)
    amp_bf16 = bool(args.amp_bf16)
    if bool(args.compile_drift) and is_main:
        print("[drifting-image] compile_drift requested, but group drift path is used; skipping compile.")
    if is_main:
        print(f"[drifting-image] amp_bf16={int(amp_bf16)}")
        print(
            f"[drifting-image] optimizer={opt_info.name} n_muon={opt_info.n_muon} n_adam={opt_info.n_adam} "
            f"local_batch={batch_size_local} grad_accum={grad_accum} effective_global_batch={global_batch_effective}"
        )

    # Queue initialization deferred until first feature extraction so n_features is known.
    pos_queue: PerClassFeatureQueue | None = None
    neg_queue: PerClassFeatureQueue | None = None
    unc_queue: PerClassFeatureQueue | None = None
    feat_names: List[str] = []
    feat_groups: Dict[str, List[int]] = {}

    model.train()
    pbar = tqdm(range(start_step, int(args.steps)), dynamic_ncols=True, disable=not is_main)
    for step in pbar:
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        need_stats = (step % int(args.log_every)) == 0
        micro_losses: List[float] = []
        drift_stats: Dict[str, float] = {}
        last_b = int(batch_size_local)
        last_mask = None
        last_alpha = None
        last_tokens = None
        last_zs = None
        last_feats_gen = None
        last_y = None

        for accum_idx in range(grad_accum):
            try:
                x0, y = next(it)
            except StopIteration:
                it = iter(loader)
                x0, y = next(it)
            x0 = x0.to(device, non_blocking=True).clamp(0.0, 1.0)
            y = y.to(device, non_blocking=True).long()
            b = x0.shape[0]
            if b != batch_size_local:
                continue

            if str(args.input_space) == "latent":
                if vae is None:
                    raise RuntimeError("VAE not initialized for latent input")
                with torch.no_grad():
                    x_base = vae.encode(x0).to(dtype=torch.float32)
            else:
                x_base = x0

            tokens, grid_hw = patchify_images(x_base, int(args.patch_size))  # [B,N,D]
            n_tokens = tokens.shape[1]
            masks_levels, _ = build_nested_patch_masks_batch(
                batch_size=b,
                n_tokens=n_tokens,
                k_min=int(args.K_min),
                levels=int(args.levels),
                generator=gen,
                device=device,
                k_schedule=str(args.k_schedule),
                k_geom_gamma=args.k_geom_gamma,
            )
            level = torch.randint(1, int(args.levels) + 1, (b,), generator=gen, device=device, dtype=torch.long)
            mask = masks_levels[torch.arange(b, device=device), level]  # [B,N]
            if args.interp_mode == "idw":
                z_s = interpolate_patch_tokens_idw(
                    tokens,
                    mask,
                    grid_hw=grid_hw,
                    power=float(args.interp_power),
                )
            else:
                z_s = interpolate_patch_tokens_laplacian(
                    tokens,
                    mask,
                    grid_hw=grid_hw,
                    init_mode="idw",
                    idw_power=float(args.interp_power),
                    sigma_feature=float(args.interp_lap_sigma_feature),
                    sigma_space=float(args.interp_lap_sigma_space),
                    lambda_reg=float(args.interp_lap_lambda_reg),
                    include_diag_neighbors=bool(args.interp_lap_diag_neighbors),
                    min_edge_weight=float(args.interp_lap_min_weight),
                )

            uniq_batch = y.unique()
            class_ids_batch = [int(c) for c in uniq_batch.tolist()]
            alpha_per_class = _sample_alpha_per_class(
                class_ids=class_ids_batch,
                alpha_values=alpha_values,
                alpha_probs=alpha_probs,
            )
            alpha_lookup = {c: a for c, a in zip(class_ids_batch, alpha_per_class)}
            alpha = torch.ones((b,), device=device, dtype=tokens.dtype)
            for c_id, a in alpha_lookup.items():
                alpha[y == int(c_id)] = float(a)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_bf16):
                delta = model(z_s, level, mask, y, grid_hw, alpha=alpha)
                if bool(args.keep_anchors_fixed):
                    non_anchor = (~mask).float().unsqueeze(-1)
                    x_hat_tokens = z_s + delta * non_anchor
                else:
                    x_hat_tokens = z_s + delta
                x_hat = unpatchify_images(
                    x_hat_tokens,
                    int(args.patch_size),
                    grid_hw=grid_hw,
                    channels=int(input_channels),
                )
                if str(args.input_space) == "pixel":
                    x_hat = x_hat.clamp(0.0, 1.0)

                if str(args.feature_space) == "resnet":
                    if str(args.input_space) == "latent":
                        if vae is None:
                            raise RuntimeError("VAE not initialized for latent input")
                        with torch.no_grad():
                            x_hat_feat = vae.decode(x_hat)
                            x_real_feat = vae.decode(x_base)
                    else:
                        x_hat_feat = x_hat
                        x_real_feat = x_base
                else:
                    # latent-MAE features operate directly on latent tensors.
                    if str(args.input_space) != "latent":
                        raise ValueError("feature_space=latent_mae requires input_space=latent")
                    x_hat_feat = x_hat
                    x_real_feat = x_base

                # Feature extraction:
                feats_gen, names_now = feature_encoder.extract_feature_vectors(
                    x_hat_feat,
                    include_global=bool(args.feature_include_global),
                    include_input_sq=bool(args.feature_include_input_sq),
                    include_local=bool(args.feature_include_local),
                    include_patch_stats=bool(args.feature_include_patch_stats),
                    max_locations=int(args.feature_max_locations),
                )
                with torch.no_grad():
                    feats_real, _ = feature_encoder.extract_feature_vectors(
                        x_real_feat,
                        include_global=bool(args.feature_include_global),
                        include_input_sq=bool(args.feature_include_input_sq),
                        include_local=bool(args.feature_include_local),
                        include_patch_stats=bool(args.feature_include_patch_stats),
                        max_locations=int(args.feature_max_locations),
                    )
                feats_gen = list(feats_gen)
                feats_real = list(feats_real)
                names_now = list(names_now)
                feats_gen_det = [f.detach() for f in feats_gen]

                if vanilla_loss_weight > 0.0:
                    vanilla_gen = x_hat_tokens.mean(dim=1)
                    vanilla_real = tokens.mean(dim=1).detach()
                    feats_gen.append(vanilla_gen)
                    feats_real.append(vanilla_real)
                    feats_gen_det.append(vanilla_gen.detach())
                    names_now.append("vanilla_token_mean")

                if not feat_names:
                    feat_names = list(names_now)
                    pos_queue = PerClassFeatureQueue(
                        num_classes=num_classes,
                        n_features=len(feat_names),
                        maxlen=int(args.queue_pos),
                    )
                    neg_queue = PerClassFeatureQueue(
                        num_classes=num_classes,
                        n_features=len(feat_names),
                        maxlen=int(args.queue_neg),
                    )
                    unc_queue = PerClassFeatureQueue(
                        num_classes=1,
                        n_features=len(feat_names),
                        maxlen=int(args.queue_unc),
                    )
                    feat_groups = collections.defaultdict(list)
                    for i, name_i in enumerate(feat_names):
                        feat_groups[_feature_group_key(name_i)].append(i)
                    feat_groups = dict(feat_groups)
                assert pos_queue is not None and neg_queue is not None and unc_queue is not None

                # Update queues with current features first.
                pos_queue.push(y, feats_real)
                neg_queue.push(y, feats_gen_det)
                unc_queue.push(torch.zeros_like(y), feats_real)

                uniq = y.unique()
                class_ids = [int(c) for c in uniq.tolist()]
                class_sel = [(y == c).nonzero(as_tuple=False).squeeze(1) for c in class_ids]
                if not class_sel:
                    continue
                class_alpha = [float(alpha_lookup[c]) for c in class_ids]

                losses: List[torch.Tensor] = []
                stats_this_micro: Dict[str, float] = {}
                stats_enabled = bool(need_stats and (accum_idx == (grad_accum - 1)))
                for group_key, group_indices in feat_groups.items():
                    x_group: List[torch.Tensor] = []
                    x_mask_group: List[torch.Tensor] = []
                    yp_group: List[torch.Tensor] = []
                    yp_mask_group: List[torch.Tensor] = []
                    yn_group: List[torch.Tensor] = []
                    yn_mask_group: List[torch.Tensor] = []
                    yn_w_group: List[torch.Tensor] = []
                    self_k_group: List[torch.Tensor] = []
                    names_group: List[str] = []
                    for j in group_indices:
                        x_pad, x_mask, yp_pad, yp_mask, yn_pad, yn_mask, yn_w, self_k = _pack_feature_class_batch(
                            x_feat=feats_gen[j],
                            x_feat_det=feats_gen_det[j],
                            feats_real_j=feats_real[j],
                            class_ids=class_ids,
                            class_sel=class_sel,
                            pos_queue=pos_queue,
                            neg_queue=neg_queue,
                            unc_queue=unc_queue,
                            feat_idx=j,
                            n_pos=int(args.drift_n_pos),
                            n_neg=int(args.drift_n_neg),
                            n_unc=int(args.drift_n_unc),
                            class_alpha=class_alpha,
                        )
                        if x_pad.numel() == 0:
                            continue
                        x_group.append(x_pad)
                        x_mask_group.append(x_mask)
                        yp_group.append(yp_pad)
                        yp_mask_group.append(yp_mask)
                        yn_group.append(yn_pad)
                        yn_mask_group.append(yn_mask)
                        yn_w_group.append(yn_w)
                        self_k_group.append(self_k)
                        names_group.append(feat_names[j])
                    if not x_group:
                        continue

                    share_group_norm = bool(share_spatial_norm and len(group_indices) > 1)
                    loss_g, _, stats_g = drifting_loss_feature_group_class_batched(
                        x_feats=x_group,
                        x_valid_masks=x_mask_group,
                        y_pos_feats=yp_group,
                        y_pos_valid_masks=yp_mask_group,
                        y_neg_feats=yn_group,
                        y_neg_valid_masks=yn_mask_group,
                        temperatures=drift_taus,
                        normalize_mode=str(args.drift_norm_mode),
                        y_neg_weights=yn_w_group,
                        self_neg_k=self_k_group,
                        share_feature_scale=share_group_norm,
                        share_drift_scale=share_group_norm,
                        return_stats=stats_enabled,
                        names=names_group,
                    )
                    is_vanilla_group = all(n.startswith("vanilla_") for n in names_group)
                    if is_vanilla_group and vanilla_loss_weight != 1.0:
                        loss_g = loss_g * vanilla_loss_weight
                        if stats_enabled:
                            stats_this_micro[f"{group_key}/loss_weight"] = float(vanilla_loss_weight)
                    losses.append(loss_g)
                    if stats_enabled:
                        for k, v in stats_g.items():
                            stats_this_micro[k] = float(v)

                if not losses:
                    continue
                # Equivalent objective: mean over classes of sum over feature losses.
                loss = torch.stack(losses).sum()

            (loss / float(grad_accum)).backward()
            micro_losses.append(float(loss.detach().item()))
            last_b = int(b)
            last_mask = mask
            last_alpha = alpha
            last_tokens = tokens
            last_zs = z_s
            last_feats_gen = feats_gen
            last_y = y
            if stats_this_micro:
                drift_stats = stats_this_micro

        if not micro_losses:
            optimizer.zero_grad(set_to_none=True)
            continue

        if float(args.grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()

        loss_value = float(sum(micro_losses) / max(1, len(micro_losses)))
        step_time = time.time() - t0
        if is_main and (step % int(args.log_every) == 0):
            writer.add_scalar("train/loss", float(loss_value), step)
            writer.add_scalar("train/step_time_sec", float(step_time), step)
            writer.add_scalar("train/samples_per_sec", float((last_b * grad_accum) / max(step_time, 1e-8)), step)
            writer.add_scalar("train/effective_global_batch", float(global_batch_effective), step)
            if last_mask is not None and last_alpha is not None and last_tokens is not None and last_zs is not None:
                writer.add_scalar("train/anchors_mean", float(last_mask.float().mean().item()), step)
                writer.add_scalar("train/cfg_alpha_mean", float(last_alpha.mean().item()), step)
                missing = (~last_mask).float()
                diff_interp = (last_zs - last_tokens).pow(2).sum(dim=-1)
                interp_mse_missing = (diff_interp * missing).sum() / (missing.sum() * last_tokens.shape[-1] + 1e-8)
                interp_mse_all = diff_interp.mean() / max(1.0, float(last_tokens.shape[-1]))
                writer.add_scalar("interp/mse_missing", float(interp_mse_missing.item()), step)
                writer.add_scalar("interp/mse_all", float(interp_mse_all.item()), step)

            for k, v in drift_stats.items():
                writer.add_scalar(f"drift/{k}", float(v), step)

            # Representation separation diagnostics (on a cheap global feature).
            if (last_feats_gen is not None) and (last_y is not None):
                sep_feat = None
                for n, f in zip(feat_names, last_feats_gen):
                    if "global_mean" in n:
                        sep_feat = f
                        break
                if sep_feat is None and len(last_feats_gen) > 0:
                    sep_feat = last_feats_gen[0]
                if sep_feat is not None:
                    sep = separation_metrics(sep_feat, last_y)
                    for k, v in sep.items():
                        writer.add_scalar(k, float(v), step)

            if pos_queue is not None and neg_queue is not None and unc_queue is not None:
                q_pos = pos_queue.stats(feat_idx=0)
                q_neg = neg_queue.stats(feat_idx=0)
                q_unc = unc_queue.stats(feat_idx=0)
                writer.add_scalar("queue_pos/non_empty", float(q_pos.non_empty), step)
                writer.add_scalar("queue_pos/mean_occupancy", float(q_pos.mean_occupancy), step)
                writer.add_scalar("queue_pos/median_occupancy", float(q_pos.median_occupancy), step)
                writer.add_scalar("queue_pos/max_fraction", float(q_pos.max_fraction), step)
                writer.add_scalar("queue_pos/entropy", float(q_pos.entropy), step)
                writer.add_scalar("queue_neg/non_empty", float(q_neg.non_empty), step)
                writer.add_scalar("queue_neg/mean_occupancy", float(q_neg.mean_occupancy), step)
                writer.add_scalar("queue_neg/median_occupancy", float(q_neg.median_occupancy), step)
                writer.add_scalar("queue_neg/max_fraction", float(q_neg.max_fraction), step)
                writer.add_scalar("queue_neg/entropy", float(q_neg.entropy), step)
                writer.add_scalar("queue_unc/non_empty", float(q_unc.non_empty), step)
                writer.add_scalar("queue_unc/mean_occupancy", float(q_unc.mean_occupancy), step)
                writer.add_scalar("queue_unc/median_occupancy", float(q_unc.median_occupancy), step)
                writer.add_scalar("queue_unc/max_fraction", float(q_unc.max_fraction), step)
                writer.add_scalar("queue_unc/entropy", float(q_unc.entropy), step)

            pbar.set_description(
                f"loss={loss_value:.4f} step={step_time:.3f}s local_b={last_b} accum={grad_accum} feat={len(feat_names)}"
            )

        if is_main and step > 0 and (step % int(args.save_every) == 0):
            ckpt = os.path.join(args.ckpt_dir, f"ckpt_{step:07d}.pt")
            meta = {
                "stage": "drifting_image_keypatch",
                "dataset": str(args.dataset),
                "data_root": str(args.data_root),
                "image_size": int(args.image_size),
                "input_space": str(args.input_space),
                "feature_space": str(args.feature_space),
                "latent_feature_ckpt": str(args.latent_feature_ckpt),
                "latent_feature_base_width": int(args.latent_feature_base_width),
                "vae_model": str(args.vae_model),
                "vae_scale": float(args.vae_scale),
                "vae_use_mean": bool(args.vae_use_mean),
                "num_classes": int(num_classes),
                "batch_size_local": int(batch_size_local),
                "batch_size_effective_global": int(global_batch_effective),
                "world_size": int(world_size),
                "grad_accum": int(grad_accum),
                "n_classes_step": int(args.n_classes_step),
                "n_gen_per_class": int(args.n_gen_per_class),
                "optimizer": str(args.optimizer),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "muon_lr": float(args.muon_lr),
                "muon_momentum": float(args.muon_momentum),
                "muon_weight_decay": float(args.muon_weight_decay),
                "muon_adam_beta1": float(args.muon_adam_beta1),
                "muon_adam_beta2": float(args.muon_adam_beta2),
                "muon_adam_eps": float(args.muon_adam_eps),
                "patch_size": int(args.patch_size),
                "K_min": int(args.K_min),
                "levels": int(args.levels),
                "k_schedule": str(args.k_schedule),
                "k_geom_gamma": float(args.k_geom_gamma) if args.k_geom_gamma is not None else None,
                "interp_power": float(args.interp_power),
                "interp_mode": str(args.interp_mode),
                "interp_lap_sigma_feature": float(args.interp_lap_sigma_feature),
                "interp_lap_sigma_space": float(args.interp_lap_sigma_space),
                "interp_lap_lambda_reg": float(args.interp_lap_lambda_reg),
                "interp_lap_diag_neighbors": bool(args.interp_lap_diag_neighbors),
                "interp_lap_min_weight": float(args.interp_lap_min_weight),
                "keep_anchors_fixed": bool(args.keep_anchors_fixed),
                "drift_n_pos": int(args.drift_n_pos),
                "drift_n_neg": int(args.drift_n_neg),
                "drift_n_unc": int(args.drift_n_unc),
                "drift_taus": str(args.drift_taus),
                "drift_norm_mode": str(args.drift_norm_mode),
                "share_spatial_norm": bool(args.share_spatial_norm),
                "cfg_alpha_values": str(args.cfg_alpha_values),
                "cfg_alpha_probs": str(args.cfg_alpha_probs),
                "vanilla_loss_weight": float(args.vanilla_loss_weight),
                "queue_pos": int(args.queue_pos),
                "queue_neg": int(args.queue_neg),
                "queue_unc": int(args.queue_unc),
                "amp_bf16": bool(args.amp_bf16),
                "compile_drift": bool(args.compile_drift),
                "compile_mode": str(args.compile_mode),
                "feature_arch": str(args.feature_arch),
                "feature_pretrained": bool(args.feature_pretrained),
                "feature_include_global": bool(args.feature_include_global),
                "feature_include_input_sq": bool(args.feature_include_input_sq),
                "feature_include_local": bool(args.feature_include_local),
                "feature_include_patch_stats": bool(args.feature_include_patch_stats),
                "feature_max_locations": int(args.feature_max_locations),
                "n_features": len(feat_names),
                "feature_names": ",".join(feat_names[:64]),  # cap metadata size
            }
            save_checkpoint(ckpt, model_raw, optimizer, step, ema=None, meta=meta)

    if is_main:
        final = os.path.join(args.ckpt_dir, "ckpt_final.pt")
        meta = {
        "stage": "drifting_image_keypatch",
        "dataset": str(args.dataset),
        "data_root": str(args.data_root),
        "image_size": int(args.image_size),
        "input_space": str(args.input_space),
        "feature_space": str(args.feature_space),
        "latent_feature_ckpt": str(args.latent_feature_ckpt),
        "latent_feature_base_width": int(args.latent_feature_base_width),
        "vae_model": str(args.vae_model),
        "vae_scale": float(args.vae_scale),
        "vae_use_mean": bool(args.vae_use_mean),
        "num_classes": int(num_classes),
        "batch_size_local": int(batch_size_local),
        "batch_size_effective_global": int(global_batch_effective),
        "world_size": int(world_size),
        "grad_accum": int(grad_accum),
        "n_classes_step": int(args.n_classes_step),
        "n_gen_per_class": int(args.n_gen_per_class),
        "optimizer": str(args.optimizer),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "muon_lr": float(args.muon_lr),
        "muon_momentum": float(args.muon_momentum),
        "muon_weight_decay": float(args.muon_weight_decay),
        "muon_adam_beta1": float(args.muon_adam_beta1),
        "muon_adam_beta2": float(args.muon_adam_beta2),
        "muon_adam_eps": float(args.muon_adam_eps),
        "patch_size": int(args.patch_size),
        "K_min": int(args.K_min),
        "levels": int(args.levels),
        "k_schedule": str(args.k_schedule),
        "k_geom_gamma": float(args.k_geom_gamma) if args.k_geom_gamma is not None else None,
        "interp_power": float(args.interp_power),
        "interp_mode": str(args.interp_mode),
        "interp_lap_sigma_feature": float(args.interp_lap_sigma_feature),
        "interp_lap_sigma_space": float(args.interp_lap_sigma_space),
        "interp_lap_lambda_reg": float(args.interp_lap_lambda_reg),
        "interp_lap_diag_neighbors": bool(args.interp_lap_diag_neighbors),
        "interp_lap_min_weight": float(args.interp_lap_min_weight),
        "keep_anchors_fixed": bool(args.keep_anchors_fixed),
        "drift_n_pos": int(args.drift_n_pos),
        "drift_n_neg": int(args.drift_n_neg),
        "drift_n_unc": int(args.drift_n_unc),
        "drift_taus": str(args.drift_taus),
        "drift_norm_mode": str(args.drift_norm_mode),
        "share_spatial_norm": bool(args.share_spatial_norm),
        "cfg_alpha_values": str(args.cfg_alpha_values),
        "cfg_alpha_probs": str(args.cfg_alpha_probs),
        "vanilla_loss_weight": float(args.vanilla_loss_weight),
        "queue_pos": int(args.queue_pos),
        "queue_neg": int(args.queue_neg),
        "queue_unc": int(args.queue_unc),
        "amp_bf16": bool(args.amp_bf16),
        "compile_drift": bool(args.compile_drift),
        "compile_mode": str(args.compile_mode),
        "feature_arch": str(args.feature_arch),
        "feature_pretrained": bool(args.feature_pretrained),
        "feature_include_global": bool(args.feature_include_global),
        "feature_include_input_sq": bool(args.feature_include_input_sq),
        "feature_include_local": bool(args.feature_include_local),
        "feature_include_patch_stats": bool(args.feature_include_patch_stats),
        "feature_max_locations": int(args.feature_max_locations),
        "n_features": len(feat_names),
        "feature_names": ",".join(feat_names[:64]),
        }
        save_checkpoint(final, model_raw, optimizer, int(args.steps), ema=None, meta=meta)
    writer.flush()
    writer.close()
    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
