"""k3r with the band entry age a0 = 4 (more members, less transport). NEGATIVE 67 killed "bands with a0 = 4" on the
cost model; this re-measures it on the meter, since the model has been wrong 3.5x and 9x today."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from estimator_k3r import Estimator as _K3R


class Estimator(_K3R):
    def __init__(self, **kw):
        kw.setdefault("band", (4, 448, "rr"))
        super().__init__(**kw)
