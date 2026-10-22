import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "src"))
from estimator_k3b import Estimator as _B
class Estimator(_B):
    def __init__(self) -> None:
        super().__init__(strassen_levels=4, chunk=8)
