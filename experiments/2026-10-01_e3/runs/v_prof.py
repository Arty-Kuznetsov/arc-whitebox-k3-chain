"""The profiling copy in the shipping configuration (recv=None), for the per-call-site FLOP breakdown."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from estimator_prof import Estimator as _P


class Estimator(_P):
    def __init__(self, **kw):
        kw.setdefault("recv", None)
        super().__init__(**kw)
