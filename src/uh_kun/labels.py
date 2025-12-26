from __future__ import annotations

from pathlib import Path

from .types import Label


_CANONICAL: dict[str, Label] = {
    "ya-kun": "Ya-kun",
    "yakun": "Ya-kun",
    "ya": "Ya-kun",
    "no-kun": "No-kun",
    "nokun": "No-kun",
    "no": "No-kun",
    "maybe-kun": "Maybe-kun",
    "maybekun": "Maybe-kun",
    "maybe": "Maybe-kun",
}


def normalize_label(value: str) -> Label:
    key = value.strip().lower()
    if key not in _CANONICAL:
        raise ValueError(f"Unknown class label '{value}'. Expected one of: {sorted(set(_CANONICAL))}")
    return _CANONICAL[key]


def infer_label_from_path(path: Path) -> Label:
    # Assumes dataset layout: <root>/<split>/<class>/<image>
    for part in reversed(path.parts):
        if part.strip().lower() in _CANONICAL:
            return normalize_label(part)
    raise ValueError(f"Could not infer label from path: {path}")
