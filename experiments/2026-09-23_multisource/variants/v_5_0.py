"""Ablation variant: n_sources=5, strassen_levels=0. Local sweep only."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "src"))
from estimator import Estimator as _B  # noqa: E402


class Estimator(_B):
    def __init__(self) -> None:
        super().__init__(n_sources=5, strassen_levels=0)
