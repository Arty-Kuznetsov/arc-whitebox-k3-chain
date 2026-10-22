"""Smoke test of src/estimator_k3p.py: band=None must be bitwise k3o; a band run must be finite and close to k3o."""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
import estimator_k3o as O, estimator_k3p as P


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


n, depth = int(sys.argv[1]), int(sys.argv[2])
kw = dict(thin=int(sys.argv[3])) if len(sys.argv) > 3 and int(sys.argv[3]) else {}
bands = [tuple(int(x) if x.isdigit() else x for x in b.split(",")) for b in sys.argv[4:]]
A = mlp(n, depth, 11)
o0, f0, r0, w0 = run(O.Estimator(**kw), A)
o1, f1, r1, w1 = run(P.Estimator(band=None, **kw), A)
print(f"n={n} depth={depth}: k3p(band=None) vs k3o bitwise {np.array_equal(o0, o1)}, flops equal {f0 == f1} "
      f"({100 * f1 / 2 ** 41:.3f}% of B), residual {r1:.3f} (k3o {r0:.3f}) s, wall {w1:.1f} s")
for b in bands:
    o2, f2, r2, w2 = run(P.Estimator(band=b, **kw), A)
    d = np.abs(o2 - o0).max(axis=1) / np.abs(o0).max(axis=1)
    print(f"  band {b}: flops x{f2 / f0:.4f} of k3o ({100 * f2 / 2 ** 41:.3f}%), finite {np.isfinite(o2).all()}, "
          f"max rel diff vs k3o per layer: " + " ".join(f"{x:.1e}" for x in d) + f"  residual {r2:.3f} s wall {w2:.1f} s")
