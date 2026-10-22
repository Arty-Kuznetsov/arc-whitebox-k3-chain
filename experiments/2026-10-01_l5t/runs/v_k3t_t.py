"""k3r + Strassen level 5 on the leg transport only (every other contraction stays at level 4)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from estimator_k3t import Estimator as _K3T


class Estimator(_K3T):
    def __init__(self, **kw):
        kw.setdefault("transport_l5", True)
        kw.setdefault("l5_sites", ("transport",))
        super().__init__(**kw)
