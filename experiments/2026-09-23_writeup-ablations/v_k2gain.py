"""Ablation variant for the write-up: gain-only covariance propagation (no tetrachoric, no Edgeworth, no sources).
Should match the starter kit's covariance-propagation example (docs: final_layer_mse 4.050e-6)."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from estimator import Estimator as _B  # noqa: E402


class Estimator(_B):
    def __init__(self) -> None:
        super().__init__(n_corr_terms=0, edgeworth=False, n_sources=0, cov_correction=False)
