"""Gate of src/estimator_k3v.py: with no coefficients (or k4corr=False) it must be BITWISE k3t with equal FLOPs."""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
import estimator_k3t as T, estimator_k3v as V


def mlp(n, depth, seed):
    rng = np.random.default_rng(seed)
    Ws = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(depth)]
    class Mo:
        width = n
    Mo.depth, Mo.seed, Mo.weights = depth, seed, [fnp.asarray(W) for W in Ws]
    return Mo


def run(est, m):
    with flops.BudgetContext(flop_budget=2 ** 50, wall_time_limit_s=3600.0, quiet=True) as ctx:
        out = np.asarray(est.predict(m, 2 ** 41), dtype=np.float64)
        return out, ctx.flops_used


# 128x8 with transport_l5=False on both: the SHIPPED k3t crashes there with its defaults (the small-width
# Strassen bug fixed in k3v -- k3t fails `whest validate`)
for (n, d, kw) in ((1024, 16, {}), (128, 8, dict(thin=128, band=None, transport_l5=False))):
    m = mlp(n, d, 11)
    o0, f0 = run(T.Estimator(**kw), m)
    o1, f1 = run(V.Estimator(k4corr=False, **kw), m)
    print(f"  {n}x{d}: bitwise {np.array_equal(o0, o1)}  flops equal {f0 == f1}")
