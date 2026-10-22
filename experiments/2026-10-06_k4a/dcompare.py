"""Per-layer D of k4a with the restructure off (= k3z bitwise) vs on, synthetic 1024x16 (debug)."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
import estimator_k4a as A
from gate import mlp, run
m = mlp(1024, 16, 11)
e0 = A.Estimator(restructure=False, young=None); e0._drec = []
e1 = A.Estimator(young=None); e1._drec = []
run(e0, m); run(e1, m)
for (l0, D0, c0, cp0), (l1, D1, c1, cp1) in zip(e0._drec, e1._drec):
    D0 = np.asarray(D0, dtype=np.float64); D1 = np.asarray(D1, dtype=np.float64)
    print(f"layer {l0:2d}: cut_cur None={c0!s:5} cut_prev None={cp0!s:5}  ||D1-D0||/||D0|| = {np.linalg.norm(D1 - D0) / np.linalg.norm(D0):.2e}"
          f"  max|dD|/max|D| = {np.abs(D1 - D0).max() / np.abs(D0).max():.2e}  nnz rows differing: {(np.abs(D1 - D0).max(axis=1) > 1e-6 * np.abs(D0).max()).sum()}")
