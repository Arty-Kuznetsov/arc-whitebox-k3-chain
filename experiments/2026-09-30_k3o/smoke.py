"""Smoke test of src/estimator_k3o.py against src/estimator_k3m.py (colcut=None must agree to float32 rounding)."""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
import estimator_k3m as M, estimator_k3o as O


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
kw = dict(thin=int(sys.argv[3])) if len(sys.argv) > 3 else {}
A = mlp(n, depth, 11)
o0, f0, r0, w0 = run(M.Estimator(**kw), A)
o1, f1, r1, w1 = run(O.Estimator(**kw), A)
d = np.abs(o1 - o0).max(axis=1) / np.abs(o0).max(axis=1)
print(f"n={n} depth={depth}: k3o(colcut=None) vs k3m  max rel diff per layer: " + " ".join(f"{x:.1e}" for x in d))
print(f"  flops k3m {f0:.6e}  k3o {f1:.6e}  ratio {f1 / f0:.6f};  residual {r0:.3f} / {r1:.3f} s;  wall {w0:.1f} / {w1:.1f} s")
for cc in (-2.5, -1.0):
    o2, f2, r2, w2 = run(O.Estimator(colcut=cc, deadcut=cc, **kw), A)
    o3, f3, _, _ = run(M.Estimator(deadcut=cc, **kw), A)
    e = np.abs(o2[-1] - o3[-1]).max() / np.abs(o3[-1]).max()
    print(f"  matched cut {cc}: flops {f2 / f3:.4f} x k3m(deadcut={cc}),  final-layer max rel diff vs it {e:.1e},  "
          f"residual {r2:.3f} s, wall {w2:.1f} s, finite {np.isfinite(o2).all()}")
