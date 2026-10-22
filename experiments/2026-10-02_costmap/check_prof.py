"""The namespaced copy must be BITWISE k3y with equal FLOPs (namespaces only label the bill)."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src")); sys.path.insert(0, str(HERE))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
import importlib
NAME = sys.argv[1] if len(sys.argv) > 1 else "k3y"
Y = importlib.import_module(f"estimator_{NAME}"); P = importlib.import_module(f"estimator_{NAME}_prof")
rng = np.random.default_rng(11)
Ws = [(rng.standard_normal((1024, 1024)) * np.sqrt(2.0 / 1024)).astype(np.float32) for _ in range(16)]
class Mo:
    width = 1024
Mo.depth, Mo.seed, Mo.weights = 16, 11, [fnp.asarray(W) for W in Ws]
def run(est):
    with flops.BudgetContext(flop_budget=2 ** 50, wall_time_limit_s=3600.0, quiet=True) as ctx:
        return np.asarray(est.predict(Mo, 2 ** 41), dtype=np.float64), ctx.flops_used
o0, f0 = run(Y.Estimator()); o1, f1 = run(P.Estimator())
print(f"prof copy bitwise {NAME}: {np.array_equal(o0, o1)}  flops equal {f0 == f1}  ({f0:.6e})")
