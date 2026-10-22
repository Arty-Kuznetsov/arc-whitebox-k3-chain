"""k3q with the Strassen recursion allowed down to a 32x32 leaf (level 5): FLOPs x7/8 vs residual + wall + RSS."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from estimator_k3r import Estimator as _K3R


class Estimator(_K3R):
    def __init__(self, **kw):
        kw.setdefault("strassen_levels", 5)
        kw.setdefault("strassen_min", 64)
        super().__init__(**kw)
