"""k3r + Strassen level 5 on the leg transport, the C/path transport and C_pre (the three most wall-efficient sites)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from estimator_k3t import Estimator as _K3T


class Estimator(_K3T):
    def __init__(self, **kw):
        kw.setdefault("transport_l5", True)
        kw.setdefault("l5_sites", ("transport", "ctransport", "cpre"))
        super().__init__(**kw)
