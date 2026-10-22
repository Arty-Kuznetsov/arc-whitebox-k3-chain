from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import flopscope.numpy as fnp
from .domain import MLP

@dataclass(frozen=True)
class SetupContext:
    width: int
    depth: int
    flop_budget: int
    api_version: str
    scratch_dir: Optional[str] = None
    submission_dir: Optional[str] = None
    seed: int = 0

class BaseEstimator(ABC):
    @abstractmethod
    def predict(self, mlp: MLP, budget: int):
        raise NotImplementedError
    def setup(self, context: SetupContext) -> None:
        return None
    def teardown(self) -> None:
        return None
