"""k3y ablation arm csym_binv_off: csym=False, binv=False"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from estimator_k3y import Estimator as _E


class Estimator(_E):
    def __init__(self):
        super().__init__(csym=False, binv=False)
