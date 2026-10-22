"""Peak RSS of one k3y predict on a synthetic 1024x16 network (run under /usr/bin/time -l)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
import estimator_k3y as Y
rng = np.random.default_rng(11)
Ws = [(rng.standard_normal((1024, 1024)) * np.sqrt(2.0 / 1024)).astype(np.float32) for _ in range(16)]
class Mo:
    width = 1024
Mo.depth, Mo.seed, Mo.weights = 16, 11, [fnp.asarray(W) for W in Ws]
with flops.BudgetContext(flop_budget=2 ** 50, wall_time_limit_s=3600.0, quiet=True):
    Y.Estimator().predict(Mo, 2 ** 41)
print("done")
