from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

try:  # optional dependency
    from transformers import CLIPTextModel, CLIPTokenizer

    _TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    CLIPTextModel = None
    CLIPTokenizer = None
    _TRANSFORMERS_AVAILABLE = False


class CLIPTextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
        max_length: int = 77,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required for CLIPTextEncoder")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_model = CLIPTextModel.from_pretrained(model_name, use_safetensors=True)
        self.max_length = int(max_length)
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.text_model.to(self.device, dtype=self.dtype)
        if freeze:
            for param in self.text_model.parameters():
                param.requires_grad = False
            self.text_model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        outputs = self.text_model(**tokens)
        return outputs.pooler_output

    def forward(self, texts: List[str]) -> torch.Tensor:
        return self.encode(texts)


__all__ = ["CLIPTextEncoder"]
