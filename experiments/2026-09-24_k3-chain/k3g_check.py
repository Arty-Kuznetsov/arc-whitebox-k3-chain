"""k3g (single-view Strassen combos) must be BITWISE equal to k3f at the same Strassen settings; report residual/FLOPs."""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
import estimator_k3f as F, estimator_k3g as G
n, depth = 1024, 16
rng = np.random.default_rng(31)
Ws = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(depth)]
class M:
    width, depth, seed = n, depth, 31
    weights = [fnp.asarray(W) for W in Ws]
def run(est):
    with flops.BudgetContext(flop_budget=2**50, wall_time_limit_s=3600.0, quiet=True) as ctx:
        t0 = time.perf_counter(); out = np.asarray(est.predict(M, 2**41), dtype=np.float64)
        return out, ctx.flops_used, ctx.residual_wall_time_s, time.perf_counter() - t0
for kw in (dict(strassen_levels=3, chunk=16), dict(strassen_levels=4, chunk=8), dict(strassen_levels=5, chunk=4)):
    fe, ge = F.Estimator(**kw), G.Estimator(**kw)
    for rep in range(2):
        o0, f0, r0, w0 = run(fe); o1, f1, r1, w1 = run(ge)
        print(f"{kw} rep {rep}: bitwise {np.array_equal(o0, o1)}, flops equal {f0 == f1} ({100*f1/2**41:.2f}% of B); "
              f"residual k3f {r0:.3f} -> k3g {r1:.3f} s; wall {w0:.1f} -> {w1:.1f} s")
