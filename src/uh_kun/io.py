from __future__ import annotations

from pathlib import Path

from PIL import Image


def load_image(path: str | Path) -> Image.Image:
    p = Path(path)
    with Image.open(p) as img:
        return img.convert("RGB")
