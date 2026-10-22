"""Residual time per section of predict() for the instrumented copy estimator_k3c_prof.py (a generated copy of
src/estimator_k3c.py with checkpoints). 'after_return' = frame teardown (freeing the workspaces)."""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import flopscope as flops
import flopscope.numpy as fnp
import estimator_k3c_prof as mod
kw = {k: (v == "True" if v in ("True", "False") else int(v)) for k, v in (a.split("=") for a in sys.argv[1:])}
n, depth = 1024, 16
rng = np.random.default_rng(0)
Ws = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(depth)]
class M:
    width, depth, seed = n, depth, 0
    weights = [fnp.asarray(W) for W in Ws]
for rep in range(2):
    mod._ACC.clear(); mod._LAST[0] = 0.0
    est = mod.Estimator(**kw)
    with flops.BudgetContext(flop_budget=2**50, wall_time_limit_s=3600.0, quiet=True) as ctx:
        mod._CTX[0] = ctx
        out = est.predict(M, 2**41)
        mod._cp('after_return')
        res = ctx.residual_wall_time_s
    mod._CTX[0] = None
    print(f"rep {rep} {kw}: residual {res:.3f}s, {100*ctx.flops_used/2**41:.2f}% of B")
    print("   " + "  ".join(f"{k} {v*1e3:.0f}" for k, v in mod._ACC.items()) + "   (ms)")
