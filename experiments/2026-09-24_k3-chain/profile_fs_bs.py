import sys, time, cProfile, pstats
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
from estimator_k3b import Estimator
n, depth, lv, chunk = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rng = np.random.default_rng(0)
Ws = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(depth)]
class M:
    width, depth, seed = n, depth, 0
    weights = [fnp.asarray(W) for W in Ws]
est = Estimator(strassen_levels=lv, chunk=chunk if chunk > 0 else 16)
pr = cProfile.Profile()
with flops.BudgetContext(flop_budget=2**50, wall_time_limit_s=3600.0, quiet=True) as ctx:
    t0 = time.perf_counter()
    pr.enable(); out = est.predict(M, 2**41); pr.disable()
    dt = time.perf_counter() - t0
print(f"{ctx.flops_used:.4e} FLOPs = {100*ctx.flops_used/2**41:.1f}% of B, wall {dt:.2f}s")
print([a for a in dir(ctx) if 'time' in a.lower()])
for a in dir(ctx):
    if 'time' in a.lower():
        try: print(a, getattr(ctx, a))
        except Exception as e: pass
pstats.Stats(pr).sort_stats("tottime").print_stats(12)
