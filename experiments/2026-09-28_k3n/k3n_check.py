"""Self-check of src/estimator_k3n.py (= estimator_k3m.py with feedmult=0.8, k22c=0.0).

    uv run --project external/whest-starterkit python experiments/2026-09-28_k3n/k3n_check.py

Two claims on a fresh 1024x16 He-init MLP:
  (1) k3n(feedmult=1.0, k22c=1.0) == k3m() bitwise, with equal FLOPs -> the file is a faithful copy and the
      two new kwargs are the only change;
  (2) k3n() (the shipped defaults) differs, costs no MORE than k3m, and the difference is the intended size.
Both knobs are scalar levels on terms the chain already forms, so (2)'s FLOP count must be <= k3m's
(k22c=0 skips one n^2 product).
"""
import sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
import estimator_k3m as M, estimator_k3n as N


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


A = mlp(1024, 16, 11)
o0, f0, r0, w0 = run(M.Estimator(), A)
o1, f1, r1, w1 = run(N.Estimator(feedmult=1.0, k22c=1.0), A)
print(f"(1) k3n(1.0, 1.0) vs k3m(): bitwise {np.array_equal(o0, o1)}, flops equal {f0 == f1} "
      f"({100 * f1 / 2 ** 41:.3f}% of B), residual {r1:.3f}s (ref {r0:.3f}s)")
o2, f2, r2, w2 = run(N.Estimator(), A)
d = np.abs(o2 - o0)[-1]
print(f"(2) k3n() defaults: {100 * f2 / 2 ** 41:.3f}% of B (k3m {100 * f0 / 2 ** 41:.3f}%, "
      f"delta {100 * (f2 - f0) / 2 ** 41:+.4f} points), residual {r2:.3f}s, wall {w2:.1f}s; "
      f"final-layer max|k3n-k3m| {d.max():.2e}, rel {d.max() / np.abs(o0[-1]).max():.1e}")
