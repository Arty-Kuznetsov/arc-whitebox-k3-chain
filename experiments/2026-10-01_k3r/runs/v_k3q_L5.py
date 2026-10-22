"""k3q with Strassen-Winograd level 5 (the L5 wall-time test; the cloud projects raw-neutral util x0.921)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from estimator_k3q import Estimator as _K3Q


class Estimator(_K3Q):
    def __init__(self, **kw):
        kw.setdefault("strassen_levels", 5)
        super().__init__(**kw)
