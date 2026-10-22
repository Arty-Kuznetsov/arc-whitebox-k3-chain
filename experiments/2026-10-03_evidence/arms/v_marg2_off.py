"""k3y ablation arm marg2_off: marg2=None"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from estimator_k3y import Estimator as _E


class Estimator(_E):
    def __init__(self):
        super().__init__(marg2=None)
