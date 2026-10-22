"""k3q + marg2 = -2.0: the k=2 marginal Edgeworth pair coefficient is Gaussian for alpha > -2.0 (cloud #9A, best arm)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from estimator_k3r import Estimator as _K3R


class Estimator(_K3R):
    def __init__(self, **kw):
        kw.setdefault("marg2", -2.0)
        super().__init__(**kw)
