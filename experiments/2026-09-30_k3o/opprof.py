"""Per-operation profile (calls, FLOPs, flopscope overhead) of k3m vs k3o on one 1024x16 MLP: where do k3o's extra
FLOPs and residual time come from? Uses BudgetContext's op rollup.  python opprof.py [colcut]"""
import sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
import estimator_k3m as M, estimator_k3o as O


def mlp(n, depth, seed):
    rng = np.random.default_rng(seed)
    Ws = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(depth)]
    class Mo:
        width = n
    Mo.depth, Mo.seed, Mo.weights = depth, seed, [fnp.asarray(W) for W in Ws]
    return Mo


def prof(est, m):
    with flops.BudgetContext(flop_budget=2 ** 50, wall_time_limit_s=3600.0, quiet=True) as ctx:
        t0 = time.perf_counter()
        est.predict(m, 2 ** 41)
        wall = time.perf_counter() - t0
        agg = defaultdict(lambda: [0, 0, 0.0, 0.0])
        for r in ctx.op_log:
            a = agg[r.op_name]
            a[0] += 1; a[1] += r.flop_cost; a[2] += r.flopscope_overhead_duration_s; a[3] += r.flopscope_backend_duration_s
        return dict(agg), ctx.flops_used, ctx.residual_wall_time_s, wall


A = mlp(1024, 16, 11)
cc = float(sys.argv[1]) if len(sys.argv) > 1 else None
runs = [("k3m", M.Estimator()), ("k3o", O.Estimator(colcut=cc, deadcut=(cc if cc is not None else -2.5)))]
res = {name: prof(e, A) for name, e in runs}
for name, (agg, f, r, w) in res.items():
    print(f"{name}: flops {f:.6e}  residual {r:.3f} s  wall {w:.1f} s  ops {sum(v[0] for v in agg.values())}  "
          f"overhead {sum(v[2] for v in agg.values()):.3f} s")
names = sorted(set(res["k3m"][0]) | set(res["k3o"][0]), key=lambda k: -(res["k3o"][0].get(k, [0, 0])[1]))
print(f"{'op':<16}{'calls m':>9}{'calls o':>9}{'GFLOP m':>11}{'GFLOP o':>11}{'ovh m ms':>10}{'ovh o ms':>10}")
for k in names:
    a = res["k3m"][0].get(k, [0, 0, 0.0, 0.0]); b = res["k3o"][0].get(k, [0, 0, 0.0, 0.0])
    print(f"{k:<16}{a[0]:>9}{b[0]:>9}{a[1] / 1e9:>11.3f}{b[1] / 1e9:>11.3f}{1e3 * a[2]:>10.1f}{1e3 * b[2]:>10.1f}")
