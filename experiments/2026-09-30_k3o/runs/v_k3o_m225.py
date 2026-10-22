"""k3o with the MATCHED dead cut at -2.25: dead-row cut (deadcut) and dead-side birth-column cut (colcut) at the same
threshold. `whest run` takes a file, not kwargs, so the variant is a subclass; src/estimator_k3o.py is untouched."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from estimator_k3o import Estimator as _K3O


class Estimator(_K3O):
    def __init__(self, **kw):
        kw.setdefault("deadcut", -2.25)
        kw.setdefault("colcut", -2.25)
        super().__init__(**kw)
