"""k3r with the band entry age a0 = 6 instead of 5, rank unchanged. a0 trades band cost (16.7% members + 6.8% entry
+ ~7% compressed pairing) against transport cost (24.0%): a higher a0 keeps sources dense longer. Never measured --
the cloud's grid tested (6, 384), i.e. a0 and rank moved together."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from estimator_k3r import Estimator as _K3R


class Estimator(_K3R):
    def __init__(self, **kw):
        kw.setdefault("band", (6, 448, "rr"))
        super().__init__(**kw)
