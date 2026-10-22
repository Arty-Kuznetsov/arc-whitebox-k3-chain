"""The profiling copy at full Strassen L5 (strassen_min=64), to read per-namespace FLOPs AND backend time and so
measure what L5 on the TRANSPORT ALONE would buy and cost -- without building a two-depth Strassen first."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from estimator_prof import Estimator as _P


class Estimator(_P):
    def __init__(self, **kw):
        kw.setdefault("recv", None)
        kw.setdefault("strassen_levels", 5)
        kw.setdefault("strassen_min", 64)
        super().__init__(**kw)
