"""k3l with the dead-row threshold at -2.5 instead of -3.0 (the cloud sweep's single best lever).
`whest run` takes a file, not kwargs, so the variant is a two-line subclass — the frozen
src/estimator_k3l.py is untouched. Expected from the sweep: util 26.27% -> 24.78%, raw +1.4%,
adjusted ~5.49e-9 local. Residual time is measured HERE because only this machine can decide it."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from estimator_k3l import Estimator as _K3L


class Estimator(_K3L):
    def __init__(self, **kw):
        kw.setdefault("deadcut", -2.5)
        super().__init__(**kw)
