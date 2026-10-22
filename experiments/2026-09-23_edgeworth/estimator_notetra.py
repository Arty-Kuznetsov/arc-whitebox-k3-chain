"""Ablation: the shipped estimator with the tetrachoric covariance term switched OFF.

RESULTS-layer-blame.md found the covariance carries none of the k=2 final-layer error, so the
tetrachoric term should be neutral. This checks that directly, now that the Edgeworth mean
correction is in place. Local ablation only -- not a submission artifact.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from estimator import Estimator as _Base  # noqa: E402


class Estimator(_Base):
    def __init__(self) -> None:
        super().__init__(n_corr_terms=0, edgeworth=True)
