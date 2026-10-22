"""Self-checks of src/estimator_k3r.py (= k3q + the marg2 switch + a settable Strassen floor).

  python k3r_check.py off        -> k3r() with defaults == k3q() BITWISE (and equal FLOPs), 1024x16 and 128x8
  python k3r_check.py marg2 TH   -> k3r(marg2=TH) vs k3q: how far the output moves, FLOPs delta, residual
  python k3r_check.py l5         -> k3r(strassen_levels=5, strassen_min=64) vs k3q: FLOPs, residual, wall, output move
"""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
import estimator_k3q as Q, estimator_k3r as R


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
        o0, f0, r0, w0 = run(Q.Estimator(**kw), m)
        o1, f1, r1, w1 = run(R.Estimator(marg2=None, strassen_min=128, **kw), m)
        print(f"  {n}x{d}: bitwise {np.array_equal(o0, o1)}  flops equal {f0 == f1} ({f0} vs {f1})  "
              f"max|d| {np.abs(o0 - o1).max():.3e}  resid {r0:.3f}/{r1:.3f}  wall {w0:.1f}/{w1:.1f}")
elif mode == "marg2":
    th = float(sys.argv[2])
    for (n, d, kw) in ((128, 8, dict(thin=128, band=None)), (1024, 16, {})):
        m = mlp(n, d, 11)
        o0, f0, r0, w0 = run(Q.Estimator(**kw), m)
        o1, f1, r1, w1 = run(R.Estimator(marg2=th, **kw), m)
        rel = np.abs(o0 - o1).max() / max(np.abs(o0).max(), 1e-30)
        print(f"  {n}x{d} marg2={th}: max rel move {rel:.3e}  flops {f0} -> {f1} (x{f1/f0:.6f})  "
              f"resid {r0:.3f} -> {r1:.3f}  wall {w0:.1f} -> {w1:.1f}")
elif mode == "l5":
    m = mlp(1024, 16, 11)
    o0, f0, r0, w0 = run(Q.Estimator(), m)
    print(f"  L4 (k3q): flops {f0:.6e}  resid {r0:.3f}  wall {w0:.1f}")
    o1, f1, r1, w1 = run(R.Estimator(strassen_levels=5, strassen_min=64), m)
    rel = np.abs(o0 - o1).max() / max(np.abs(o0).max(), 1e-30)
    print(f"  L5 (k3r): flops {f1:.6e} (x{f1/f0:.5f})  resid {r1:.3f} (x{r1/max(r0,1e-9):.2f})  "
          f"wall {w1:.1f} (x{w1/w0:.2f})  max rel move vs L4 {rel:.3e}")
    import resource
    print(f"  peak RSS this process: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9:.2f} GB")
else:
    raise SystemExit("mode?")
