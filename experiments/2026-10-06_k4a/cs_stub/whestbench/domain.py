from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List

@dataclass
class MLP:
    width: int
    depth: int
    weights: List[Any]
    seed: int = 0
    name: str = ""
    def validate(self) -> None:
        if len(self.weights) != self.depth:
            raise ValueError("depth mismatch")
        for w in self.weights:
            if tuple(w.shape) != (self.width, self.width):
                raise ValueError(f"bad weight shape {w.shape}")
