"""k3p = k3o (matched cut -2.5) + the transported band (a0=6, r=512, init=rr). whest run takes a file, not kwargs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from estimator_k3p import Estimator as _K3P


class Estimator(_K3P):
    def __init__(self, **kw):
        kw.setdefault("band", (6, 512, "rr"))
        super().__init__(**kw)
