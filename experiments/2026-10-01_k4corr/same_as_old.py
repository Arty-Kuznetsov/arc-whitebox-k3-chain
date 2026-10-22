"""The numpy-free k3v must give BITWISE the output of the rejected k3v (333367) -- only the recording branch changed."""
import sys, importlib.util
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
import estimator_k3v as NEW
spec = importlib.util.spec_from_file_location("old_k3v", sys.argv[1])
OLD = importlib.util.module_from_spec(spec); spec.loader.exec_module(OLD)
rng = np.random.default_rng(5)
n, depth = 1024, 16
Ws = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(depth)]
class Mo:
    width = n
Mo.depth, Mo.seed, Mo.weights = depth, 5, [fnp.asarray(W) for W in Ws]
outs = []
for M in (OLD, NEW):
    with flops.BudgetContext(flop_budget=2 ** 50, wall_time_limit_s=3600.0, quiet=True) as ctx:
        outs.append((np.asarray(M.Estimator().predict(Mo, 2 ** 41), dtype=np.float64), ctx.flops_used))
print(f"  corrector ON, 1024x16: bitwise {np.array_equal(outs[0][0], outs[1][0])}  flops equal {outs[0][1] == outs[1][1]}")
