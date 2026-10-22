"""Self-checks of src/estimator_k3t.py (= k3r + Strassen level 5 on the transport only).
  off  -> k3t(transport_l5=False) == k3r() BITWISE with equal FLOPs (1024x16 and 128x8)
  on   -> k3t(transport_l5=True) vs k3r at 1024x16: FLOPs ratio, max rel move, residual, wall, peak RSS
"""
import sys, time, resource
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
import estimator_k3r as Rr, estimator_k3t as Rt


def mlp(n, depth, seed):
    rng = np.random.default_rng(seed)
    Ws = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(depth)]
    class Mo:
        width = n
    Mo.depth, Mo.seed, Mo.weights = depth, seed, [fnp.asarray(W) for W in Ws]
    return Mo


def run(est, m):
    with flops.BudgetContext(flop_budget=2 ** 50, wall_time_limit_s=3600.0, quiet=True) as ctx:
        t0 = time.perf_counter()
        out = np.asarray(est.predict(m, 2 ** 41), dtype=np.float64)
        return out, ctx.flops_used, ctx.residual_wall_time_s, time.perf_counter() - t0


mode = sys.argv[1]
if mode == "off":
    for (n, d, kw) in ((1024, 16, {}), (128, 8, dict(thin=128, band=None))):
        m = mlp(n, d, 11)
        o0, f0, r0, w0 = run(Rr.Estimator(**kw), m)
        o1, f1, r1, w1 = run(Rt.Estimator(transport_l5=False, **kw), m)
        print(f"  {n}x{d}: bitwise {np.array_equal(o0, o1)}  flops equal {f0 == f1}  max|d| {np.abs(o0 - o1).max():.3e}")
else:
    m = mlp(1024, 16, 11)
    o0, f0, r0, w0 = run(Rr.Estimator(), m)
    o1, f1, r1, w1 = run(Rt.Estimator(transport_l5=True), m)
    rel = np.abs(o0 - o1).max() / max(np.abs(o0).max(), 1e-30)
    print(f"  1024x16 transport_l5: FLOPs {f0:.5e} -> {f1:.5e} x{f1/f0:.5f}   max rel move {rel:.3e}   "
          f"resid {r0:.3f} -> {r1:.3f}   wall {w0:.1f} -> {w1:.1f}")
    print(f"  peak RSS of this process (both estimators): {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9:.2f} GB")
