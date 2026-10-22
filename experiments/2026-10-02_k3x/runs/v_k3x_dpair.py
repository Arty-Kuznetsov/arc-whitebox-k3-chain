"""k3x with "dpair" in the level-5 site list (the candidate default), for the 100-MLP run before the default is flipped."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from estimator_k3x import Estimator as _E


class Estimator(_E):
    def __init__(self):
        super().__init__(l5_sites=("transport", "ctransport", "cpre", "band_members", "dpair"))
