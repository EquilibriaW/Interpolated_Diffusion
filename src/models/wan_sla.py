from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from SLA import SparseLinearAttention, SageSparseLinearAttention
except Exception:  # pragma: no cover - optional dependency
    SparseLinearAttention = None
    SageSparseLinearAttention = None


def _apply_rotary_emb(hidden_states: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


class WanSLAProcessor(nn.Module):
    def __init__(
        self,
        *,
        head_dim: int,
        topk: float,
        attention_type: str = "sagesla",
        use_bf16: bool = True,
        feature_map: str = "softmax",
    ) -> None:
        super().__init__()
        attention_type = attention_type.lower()
        if attention_type not in {"sla", "sagesla"}:
            raise ValueError(f"Unsupported SLA attention_type: {attention_type}")

        if attention_type == "sagesla":
            if SageSparseLinearAttention is None:
                raise ImportError(
                    "SageSLA is not available. Install SpargeAttn/SageSLA dependencies to enable it."
                )
            self.sla = SageSparseLinearAttention(head_dim=head_dim, topk=topk, feature_map=feature_map, use_bf16=use_bf16)
        else:
            if SparseLinearAttention is None:
                raise ImportError("SLA is not available. Install SLA dependencies to enable it.")
            # Match TurboDiffusion default block sizes for SLA.
            self.sla = SparseLinearAttention(
                head_dim=head_dim,
                topk=topk,
                feature_map=feature_map,
                BLKQ=128,
                BLKK=64,
                use_bf16=use_bf16,
            )

        # For compatibility with diffusers' attention backend API.
        self._attention_backend: Optional[str] = None

    def _run_sla(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        # SLA kernels expect inputs in (B, L, H, D) and return (B, L, H, D).
        # The Wan QKV tensors are already (B, L, H, D), so avoid permuting.
        return self.sla(query, key, value)

    def forward(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Import locally to avoid hard dependency in module import time.
        from diffusers.models.transformers.transformer_wan import (
            _get_added_kv_projections,
            _get_qkv_projections,
        )
        from diffusers.models.attention_dispatch import dispatch_attention_fn

        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded in Wan.
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))
            # Added KV often has different sequence length; use dense attention for safety.
            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                parallel_config=getattr(self, "_parallel_config", None),
            )
            hidden_states_img = hidden_states_img.flatten(2, 3).type_as(query)

        if attn.is_cross_attention:
            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                parallel_config=getattr(self, "_parallel_config", None),
            )
            hidden_states = hidden_states.flatten(2, 3).type_as(query)
        else:
            hidden_states = self._run_sla(query, key, value)
            hidden_states = hidden_states.flatten(2, 3).type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def apply_wan_sla(
    model: torch.nn.Module,
    *,
    topk: float = 0.1,
    attention_type: str = "sagesla",
    use_bf16: bool = True,
    feature_map: str = "softmax",
) -> int:
    replaced = 0
    for module in model.modules():
        if module.__class__.__name__ == "WanAttention":
            head_dim = module.inner_dim // module.heads
            processor = WanSLAProcessor(
                head_dim=head_dim,
                topk=topk,
                attention_type=attention_type,
                use_bf16=use_bf16,
                feature_map=feature_map,
            )
            processor = processor.to(device=module.to_q.weight.device)
            module.set_processor(processor)
            replaced += 1
    if replaced == 0:
        raise ValueError("No WanAttention modules found to replace with SLA.")
    return replaced


__all__ = ["WanSLAProcessor", "apply_wan_sla"]
