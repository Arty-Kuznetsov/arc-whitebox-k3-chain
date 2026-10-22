"""Prototype: C_pre symmetric split (upper block row + O22, O21 = O12^T) vs k3x defaults on a synthetic 1024x16 MLP:
FLOPs ratio, max relative move, wall of predict() (one run each, shared machine -> indicative only)."""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src")); sys.path.insert(0, str(HERE))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
import estimator_k3x as X, estimator_csym as Y


def mlp(n, depth, seed):
    rng = np.random.default_rng(seed)
    Ws = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(depth)]
    class Mo:
        width = n
    Mo.depth, Mo.seed, Mo.weights = depth, seed, [fnp.asarray(Wm) for Wm in Ws]
    return Mo


def run(est, m):
    with flops.BudgetContext(flop_budget=2 ** 50, wall_time_limit_s=3600.0, quiet=True) as ctx:
        t0 = time.time()
        out = np.asarray(est.predict(m, 2 ** 41), dtype=np.float64)
        return out, ctx.flops_used, time.time() - t0


if __name__ == "__main__":
  m = mlp(1024, 16, 11)
  o0, f0, w0 = run(X.Estimator(), m)
  o1, f1, w1 = run(Y.Estimator(), m)
  print(f"csym off == k3x: bitwise {np.array_equal(o0, o1)}  flops equal {f0 == f1}")
  o2, f2, w2 = run(Y.Estimator(csym=True), m)
  rel = np.abs(o2 - o0).max() / np.abs(o0).max()
  print(f"csym on: FLOPs x{f2 / f0:.5f} (util {f0 / 2**41:.4%} -> {f2 / 2**41:.4%})  max rel move {rel:.2e}  wall {w0:.1f} -> {w2:.1f} s")
  m8 = mlp(128, 8, 11)
  kw = dict(thin=128, band=None, transport_l5=False)
  a, fa, _ = run(X.Estimator(**kw), m8); b, fb, _ = run(Y.Estimator(csym=True, **kw), m8)
  print(f"128x8 csym on: finite {np.isfinite(b).all()}  FLOPs x{fb / fa:.5f}  max rel move {np.abs(b - a).max() / np.abs(a).max():.2e}")
