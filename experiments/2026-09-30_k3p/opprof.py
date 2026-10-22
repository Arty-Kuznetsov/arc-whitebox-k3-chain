"""Per-operation profile of k3o vs k3p(band): calls, FLOPs, flopscope overhead, and where the residual time goes
(residual = wall - backend - overhead, per the BudgetContext). python opprof.py a0 r init"""
import sys, time
from collections import defaultdict
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


def prof(est, m):
    with flops.BudgetContext(flop_budget=2 ** 50, wall_time_limit_s=3600.0, quiet=True) as ctx:
        t0 = time.perf_counter()
        est.predict(m, 2 ** 41)
        wall = time.perf_counter() - t0
        agg = defaultdict(lambda: [0, 0, 0.0, 0.0])
        gaps = []
        prev_end = None
        for r in ctx.op_log:
            a = agg[r.op_name]
            a[0] += 1; a[1] += r.flop_cost; a[2] += r.flopscope_overhead_duration_s; a[3] += r.flopscope_backend_duration_s
        return dict(agg), ctx.flops_used, ctx.residual_wall_time_s, wall, ctx.flopscope_backend_time_s, ctx.flopscope_overhead_time_s


A = mlp(1024, 16, 11)
band = (int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
res = {"k3o": prof(O.Estimator(), A), "k3p": prof(P.Estimator(band=band), A)}
for name, (agg, f, r, w, be, ov) in res.items():
    print(f"{name}: flops {f:.4e} residual {r:.3f} s wall {w:.1f} s backend {be:.2f} s overhead {ov:.3f} s "
          f"ops {sum(v[0] for v in agg.values())}")
names = sorted(set(res["k3o"][0]) | set(res["k3p"][0]), key=lambda k: -(res["k3p"][0].get(k, [0, 0, 0, 0])[2]))
print(f"{'op':<22}{'calls o':>8}{'calls p':>8}{'ovh o ms':>10}{'ovh p ms':>10}{'be o s':>9}{'be p s':>9}")
for k in names[:25]:
    a = res["k3o"][0].get(k, [0, 0, 0.0, 0.0]); b = res["k3p"][0].get(k, [0, 0, 0.0, 0.0])
    print(f"{k:<22}{a[0]:>8}{b[0]:>8}{1e3 * a[2]:>10.1f}{1e3 * b[2]:>10.1f}{a[3]:>9.2f}{b[3]:>9.2f}")
