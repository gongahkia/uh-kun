from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Label = Literal["Ya-kun", "No-kun", "Maybe-kun"]


@dataclass(frozen=True)
class Prediction:
    label: Label
    scores: dict[Label, float]
    neighbors: list[tuple[str, Label, float]]
