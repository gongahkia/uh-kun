from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .labels import infer_label_from_path
from .types import Label


@dataclass(frozen=True)
class ImageItem:
    id: str
    path: Path
    label: Label


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def scan_image_folder(root: str | Path) -> list[ImageItem]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(root_path)

    items: list[ImageItem] = []
    for p in sorted(root_path.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in _IMAGE_EXTS:
            continue
        label = infer_label_from_path(p)
        # Use relative path as stable ID
        _id = str(p.relative_to(root_path))
        items.append(ImageItem(id=_id, path=p, label=label))
    return items
