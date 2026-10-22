"""k3q + marg2 everywhere: coef(2,p) = w(2,p) for every neuron (the threshold-free variant, cloud #9A x0.9867)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from estimator_k3r import Estimator as _K3R


class Estimator(_K3R):
    def __init__(self, **kw):
        kw.setdefault("marg2", -1e9)
        super().__init__(**kw)
