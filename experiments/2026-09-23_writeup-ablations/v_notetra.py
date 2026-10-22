"""Ablation variant for the write-up: the shipped configuration with the tetrachoric term switched off."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from estimator import Estimator as _B  # noqa: E402


class Estimator(_B):
    def __init__(self) -> None:
        super().__init__(n_corr_terms=0)
