"""Ablation variant for the write-up: our September k=2 baseline (tetrachoric on, no Edgeworth, no sources).
Self-check: must reproduce 4.058e-6 final_layer_mse on the 100-MLP mini split."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from estimator import Estimator as _B  # noqa: E402


class Estimator(_B):
    def __init__(self) -> None:
        super().__init__(edgeworth=False, n_sources=0, cov_correction=False)
