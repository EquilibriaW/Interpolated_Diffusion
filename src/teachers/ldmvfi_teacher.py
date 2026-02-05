from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import sys
import types

import torch


def _ensure_ldmvfi_import(repo_root: str | Path):
    repo_root = Path(repo_root)
    if not repo_root.exists():
        raise FileNotFoundError(f"LDMVFI repo not found: {repo_root}")
    repo_str = str(repo_root.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    # Patch Lightning v2+ to provide legacy rank_zero_only import path.
    try:
        from pytorch_lightning.utilities.rank_zero import rank_zero_only  # type: ignore

        if "pytorch_lightning.utilities.distributed" not in sys.modules:
            mod = types.ModuleType("pytorch_lightning.utilities.distributed")
            mod.rank_zero_only = rank_zero_only
            sys.modules["pytorch_lightning.utilities.distributed"] = mod
    except Exception:
        pass
    # Ensure taming-transformers is discoverable (editable installs can miss it).
    try:
        import taming  # noqa: F401
    except Exception:
        venv_root = Path(sys.prefix)
        candidate = venv_root / "src" / "taming-transformers"
        if candidate.exists():
            sys.path.insert(0, str(candidate))
    try:
        from omegaconf import OmegaConf  # type: ignore
        from ldm.util import instantiate_from_config  # type: ignore
        from ldm.models.diffusion.ddim import DDIMSampler  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Failed to import LDMVFI dependencies. Install requirements from tmp_repos/LDMVFI/environment.yaml "
            "(omegaconf, pytorch-lightning, timm, taming-transformers, clip, etc.)."
        ) from exc
    return OmegaConf, instantiate_from_config, DDIMSampler


class LDMVFITEACHER:
    def __init__(
        self,
        *,
        repo_root: str | Path,
        config_path: str | Path,
        ckpt_path: str | Path,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        use_ddim: bool = True,
        ddim_steps: int = 50,
        ddim_eta: float = 1.0,
        autocast_dtype: Optional[torch.dtype] = None,
    ) -> None:
        OmegaConf, instantiate_from_config, DDIMSampler = _ensure_ldmvfi_import(repo_root)
        config = OmegaConf.load(str(config_path))
        model = instantiate_from_config(config.model)
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            # LDMVFI checkpoints often contain extra keys; keep best-effort load.
            pass
        model.to(device=device, dtype=dtype)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        self.model = model
        self.device = device
        self.dtype = dtype
        self.use_ddim = bool(use_ddim)
        self.ddim_steps = int(ddim_steps)
        self.ddim_eta = float(ddim_eta)
        self.autocast_dtype = autocast_dtype
        self.ddim_sampler = DDIMSampler(model) if self.use_ddim else None

    @torch.no_grad()
    def interpolate(self, prev: torch.Tensor, next: torch.Tensor) -> torch.Tensor:
        prev = prev.to(device=self.device, dtype=self.dtype)
        next = next.to(device=self.device, dtype=self.dtype)
        autocast_ctx = (
            torch.cuda.amp.autocast(dtype=self.autocast_dtype)
            if self.autocast_dtype is not None
            else torch.autocast("cuda", enabled=False)
        )
        def _run_infer():
            xc = {"prev_frame": prev, "next_frame": next}
            c, phi_prev_list, phi_next_list = self.model.get_learned_conditioning(xc)
            shape = (self.model.channels, c.shape[2], c.shape[3])
            if self.use_ddim and self.ddim_sampler is not None:
                out = self.ddim_sampler.sample(
                    S=self.ddim_steps,
                    conditioning=c,
                    batch_size=c.shape[0],
                    shape=shape,
                    eta=self.ddim_eta,
                    verbose=False,
                )
            else:
                out = self.model.sample_ddpm(conditioning=c, batch_size=c.shape[0], shape=shape, x_T=None)
            if isinstance(out, tuple):
                out = out[0]
            out = self.model.decode_first_stage(out, xc, phi_prev_list, phi_next_list)
            return out

        with autocast_ctx:
            if hasattr(self.model, "ema_scope"):
                try:
                    with self.model.ema_scope():
                        out = _run_infer()
                except AssertionError:
                    out = _run_infer()
            else:
                out = _run_infer()
        return torch.clamp(out, min=-1.0, max=1.0)


__all__ = ["LDMVFITEACHER"]
