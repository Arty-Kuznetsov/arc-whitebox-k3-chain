"""Gate of src/estimator_k3y.py: csym=False, binv=False must be BITWISE k3x with equal FLOPs (1024x16, 128x8); csym
alone and the defaults (csym + binv) vs k3x: FLOPs ratio, max relative move, finiteness."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
import estimator_k3x as X, estimator_k3y as Y


def mlp(n, depth, seed):
    rng = np.random.default_rng(seed)
    Ws = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(depth)]
    class Mo:
        width = n
    Mo.depth, Mo.seed, Mo.weights = depth, seed, [fnp.asarray(Wm) for Wm in Ws]
    return Mo


def run(est, m):
    with flops.BudgetContext(flop_budget=2 ** 50, wall_time_limit_s=3600.0, quiet=True) as ctx:
        out = np.asarray(est.predict(m, 2 ** 41), dtype=np.float64)
        return out, ctx.flops_used


small = dict(thin=128, band=None, transport_l5=False)
for (n, d, kw) in ((1024, 16, {}), (128, 8, small)):
    m = mlp(n, d, 11)
    o0, f0 = run(X.Estimator(**kw), m)
    o1, f1 = run(Y.Estimator(csym=False, binv=False, **kw), m)
    print(f"{n}x{d}: csym=False, binv=False bitwise k3x {np.array_equal(o0, o1)}  flops equal {f0 == f1}", flush=True)
    for lab, kk in (("csym only", dict(binv=False)), ("defaults (csym + binv)", {})):
        o2, f2 = run(Y.Estimator(**kk, **kw), m)
        print(f"   {lab:24s} FLOPs x{f2 / f0:.5f} util {f2 / 2 ** 41:.4%}  max rel move "
              f"{np.abs(o2 - o0).max() / np.abs(o0).max():.2e}  finite {np.isfinite(o2).all()}", flush=True)
