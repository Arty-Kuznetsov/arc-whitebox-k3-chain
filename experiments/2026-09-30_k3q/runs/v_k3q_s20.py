"""k3q = k3p (matched cut -2.5, band (5,448,rr)) + E2 saturated-row cut at alpha > 2.0 with the kappa4 level restored."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from estimator_k3q import Estimator as _K3Q


class Estimator(_K3Q):
    def __init__(self, **kw):
        kw.setdefault("satrow", 2.0)
        super().__init__(**kw)
