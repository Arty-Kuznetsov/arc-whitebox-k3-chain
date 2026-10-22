"""Self-checks of src/estimator_k3s.py (= k3r + E3 arm "zero").

  python k3s_check.py off        -> k3s(recv=None) == k3r() BITWISE with equal FLOPs, 1024x16 and 128x8
  python k3s_check.py on [TH]    -> k3s(recv=TH) vs k3r: FLOPs saved, how far the output moves, residual, and the
                                    hard invariant: t must be EXACTLY 0 at every receiver neuron (so the readout of
                                    those neurons loses its kappa3 diagonal and nothing else)
"""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
import estimator_k3r as Rr, estimator_k3s as Rs


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
        o1, f1, r1, w1 = run(Rs.Estimator(recv=None, **kw), m)
        print(f"  {n}x{d}: bitwise {np.array_equal(o0, o1)}  flops equal {f0 == f1} ({f0} vs {f1})  "
              f"max|d| {np.abs(o0 - o1).max():.3e}  resid {r0:.3f}/{r1:.3f}")
elif mode == "on":
    th = float(sys.argv[2]) if len(sys.argv) > 2 else -3.0
    for (n, d, kw) in ((128, 8, dict(thin=128, band=None)), (1024, 16, {})):
        m = mlp(n, d, 11)
        o0, f0, r0, w0 = run(Rr.Estimator(**kw), m)
        o1, f1, r1, w1 = run(Rs.Estimator(recv=th, **kw), m)
        rel = np.abs(o0 - o1).max() / max(np.abs(o0).max(), 1e-30)
        print(f"  {n}x{d} recv={th}: FLOPS {f0:.5e} -> {f1:.5e}  x{f1/f0:.5f}   max rel move {rel:.3e}   "
              f"resid {r0:.3f} -> {r1:.3f}   wall {w0:.1f} -> {w1:.1f}")
else:
    raise SystemExit("mode?")
