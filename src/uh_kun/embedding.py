from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class ClipConfig:
    model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    device: str | None = None


@lru_cache(maxsize=2)
def _load_openclip(model_name: str, pretrained: str, device: str):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval()
    model.to(device)
    return model, preprocess


def _default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ClipEmbedder:
    def __init__(self, config: ClipConfig | None = None):
        self.config = config or ClipConfig()
        self.device = self.config.device or _default_device()
        self._model, self._preprocess = _load_openclip(
            self.config.model_name, self.config.pretrained, self.device
        )

    @torch.inference_mode()
    def embed_pil(self, images: Iterable["PIL.Image.Image"]) -> np.ndarray:
        # Returned shape: (N, D), normalized.
        import PIL.Image

        tensors: list[torch.Tensor] = []
        for img in images:
            if not isinstance(img, PIL.Image.Image):
                raise TypeError("Expected PIL.Image.Image")
            tensors.append(self._preprocess(img))

        if not tensors:
            return np.zeros((0, 0), dtype=np.float32)

        batch = torch.stack(tensors).to(self.device)
        feats = self._model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.detach().cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def embed_one_pil(self, image: "PIL.Image.Image") -> np.ndarray:
        vec = self.embed_pil([image])
        return vec[0]
