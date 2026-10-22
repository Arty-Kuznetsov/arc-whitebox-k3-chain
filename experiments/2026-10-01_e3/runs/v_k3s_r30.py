"""k3r + E3 arm "zero" at recv = -3.0 (receiver columns of the leg transport not computed; t = 0 there)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from estimator_k3s import Estimator as _K3S


class Estimator(_K3S):
    def __init__(self, **kw):
        kw.setdefault("recv", -3.0)
        super().__init__(**kw)
