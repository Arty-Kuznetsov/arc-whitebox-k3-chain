"""mpair (Strassen level 5 on the members' compressed D pairing) on top of k3y: bitwise when off; FLOPs, max rel move,
and PAIRED wall timing when on (alternating on one synthetic 1024x16 MLP, warm)."""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src")); sys.path.insert(0, str(HERE))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
import estimator_k3y as Y, estimator_mpair as M
SITES = ("transport", "ctransport", "cpre", "band_members", "dpair")


def mlp(n, depth, seed):
    rng = np.random.default_rng(seed)
    Ws = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(depth)]
    class Mo:
        width = n
    Mo.depth, Mo.seed, Mo.weights = depth, seed, [fnp.asarray(Wm) for Wm in Ws]
    return Mo


def run(est, m):
    with flops.BudgetContext(flop_budget=2 ** 50, wall_time_limit_s=3600.0, quiet=True) as ctx:
        t0 = time.perf_counter()
        out = np.asarray(est.predict(m, 2 ** 41), dtype=np.float64)
        return out, ctx.flops_used, time.perf_counter() - t0


m = mlp(1024, 16, 11)
ey, eo, e5 = Y.Estimator(), M.Estimator(), M.Estimator(l5_sites=SITES + ("mpair",))
o0, f0, _ = run(ey, m); o1, f1, _ = run(eo, m)
print(f"mpair off == k3y: bitwise {np.array_equal(o0, o1)}  flops equal {f0 == f1}", flush=True)
o2, f2, _ = run(e5, m)
print(f"mpair on: FLOPs x{f2 / f0:.5f}  util {f2 / 2 ** 41:.4%}  max rel move {np.abs(o2 - o0).max() / np.abs(o0).max():.2e}  finite {np.isfinite(o2).all()}", flush=True)
ty, t5 = [], []
for rep in range(3):
    ty.append(run(ey, m)[2]); t5.append(run(e5, m)[2])
    print(rep, round(ty[-1], 1), round(t5[-1], 1), flush=True)
print(f"paired wall: k3y {np.median(ty):.2f} s, mpair {np.median(t5):.2f} s  ({np.median(t5) - np.median(ty):+.2f} s)")
