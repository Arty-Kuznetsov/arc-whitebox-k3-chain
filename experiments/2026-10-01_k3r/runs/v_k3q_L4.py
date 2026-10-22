"""k3q with its shipping Strassen level 4 (the reference arm of the L5 test, same runner, same day)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from estimator_k3q import Estimator as _K3Q


class Estimator(_K3Q):
    def __init__(self, **kw):
        kw.setdefault("strassen_levels", 4)
        super().__init__(**kw)
