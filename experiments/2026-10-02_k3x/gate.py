"""Gate of src/estimator_k3x.py before it ships (pre-package checklist step 3):
  (a) with k3w's level-5 site list it must be BITWISE k3w with equal FLOPs (1024x16 and 128x8);
  (b) with its DEFAULTS (= k3w's sites + "dpair"): FLOPs ratio and max relative move vs k3w at 1024x16, and the
      128x8 small-width config runs (the tiny `whest validate` contract run is checked separately).
Run: uv run --project ../../external/whest-starterkit python gate.py   (from this folder)"""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
import estimator_k3w as W, estimator_k3x as X

K3W_SITES = ("transport", "ctransport", "cpre", "band_members")


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
    o0, f0 = run(W.Estimator(**kw), m)
    o1, f1 = run(X.Estimator(l5_sites=K3W_SITES, **kw), m)
    print(f"(a) {n}x{d}: k3x(k3w sites) bitwise k3w {np.array_equal(o0, o1)}  flops equal {f0 == f1}  ({f0:.6e})")
    if n == 1024:
        o2, f2 = run(X.Estimator(), m)
        rel = np.abs(o2 - o0).max() / np.abs(o0).max()
        print(f"(b) {n}x{d}: k3x defaults (+dpair) FLOPs x{f2 / f0:.5f}  util {f2 / 2 ** 41:.4%}  max rel move {rel:.2e}  "
              f"finite {np.isfinite(o2).all()}")
