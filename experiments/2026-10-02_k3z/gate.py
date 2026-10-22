"""Gate of src/estimator_k3z.py: winograd=False, csym=1 must be BITWISE k3y with equal FLOPs (1024x16, 128x8); each
saving alone and both together (the defaults) vs k3y: FLOPs ratio, max relative move, wall of one warm predict."""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
import estimator_k3y as Y, estimator_k3z as Z


def mlp(n, depth, seed):
    rng = np.random.default_rng(seed)
    Ws = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(depth)]
    class Mo:
        width = n
    Mo.depth, Mo.seed, Mo.weights = depth, seed, [fnp.asarray(W) for W in Ws]
    return Mo


def run(est, m, reps=1):
    out = None
    for _ in range(reps):
        with flops.BudgetContext(flop_budget=2 ** 50, wall_time_limit_s=3600.0, quiet=True) as ctx:
            t0 = time.perf_counter()
            out = np.asarray(est.predict(m, 2 ** 41), dtype=np.float64)
            w = time.perf_counter() - t0
            f = ctx.flops_used
    return out, f, w


if __name__ == "__main__":
    small = dict(thin=128, band=None, transport_l5=False)
    for (n, d, kw) in ((1024, 16, {}), (128, 8, small)):
        m = mlp(n, d, 11)
        ey = Y.Estimator(**kw)
        o0, f0, w0 = run(ey, m, 2)
        o1, f1, _ = run(Z.Estimator(winograd=False, csym=1, **kw), m)
        print(f"{n}x{d}: winograd=False, csym=1 bitwise k3y {np.array_equal(o0, o1)}  flops equal {f0 == f1}", flush=True)
        for lab, kk in (("winograd only", dict(csym=1)), ("csym=2 only", dict(winograd=False)), ("defaults", {})):
            ez = Z.Estimator(**kk, **kw)
            o2, f2, w2 = run(ez, m, 2 if n == 1024 else 1)
            print(f"   {lab:14s} FLOPs x{f2 / f0:.5f}  max rel move {np.abs(o2 - o0).max() / np.abs(o0).max():.2e}  "
                  f"finite {np.isfinite(o2).all()}  wall {w0:.1f} -> {w2:.1f} s", flush=True)
