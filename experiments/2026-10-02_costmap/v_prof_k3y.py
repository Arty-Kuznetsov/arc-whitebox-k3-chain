"""The namespaced k3y for `whest run` (the runner's breakdowns.estimator.by_namespace is the cost map)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimator_k3y_prof import Estimator as _P


class Estimator(_P):
    pass
