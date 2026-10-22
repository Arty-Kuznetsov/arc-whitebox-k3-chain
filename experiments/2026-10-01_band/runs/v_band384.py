"""k3r with the band rank set to 384 (shipped: 448). The band costs 23.5% of the budget (members 16.71% + entry 6.81%)
and its rank has never been swept on the real meter -- the cloud chose 448 with the cost model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from estimator_k3r import Estimator as _K3R


class Estimator(_K3R):
    def __init__(self, **kw):
        kw.setdefault("band", (5, 384, "rr"))
        super().__init__(**kw)
