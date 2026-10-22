"""k3v with its shipped defaults (wrapper for a screen)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from estimator_k3v import Estimator as _K3V


class Estimator(_K3V):
    pass
