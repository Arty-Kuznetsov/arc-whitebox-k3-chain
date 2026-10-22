"""Addendum to k3m_check.py step (4): the util of pathrank 32 and 48, the last ranks below the measured
break-even (158) where the lever could still live.  Same probe MLP (1024x16, seed 11) and the same harness,
so the numbers append directly to that table."""
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
import estimator_k3m as M

B = 2 ** 41
rng = np.random.default_rng(11)
Ws = [(rng.standard_normal((1024, 1024)) * np.sqrt(2.0 / 1024)).astype(np.float32) for _ in range(16)]
class Mm:
    width, depth, seed = 1024, 16, 11
with flops.BudgetContext(flop_budget=2 ** 50, wall_time_limit_s=3600.0, quiet=True):
    Mm.weights = [fnp.asarray(W) for W in Ws]

def run(est):
    with flops.BudgetContext(flop_budget=2 ** 50, wall_time_limit_s=3600.0, quiet=True) as ctx:
        t0 = time.perf_counter()
        out = np.asarray(est.predict(Mm, B), dtype=np.float64)
        return out, ctx.flops_used, ctx.residual_wall_time_s, time.perf_counter() - t0

base_out, base_f, _, _ = run(M.Estimator())
print(f"    pathrank None              {100*base_f/B:.3f}% of B  (1.0000 x pathrank=None)")
recs = []
for r in (48, 32):
    for sv, tag in ((True, "svd"), (False, "rangefinder")):
        o, f, rw, wl = run(M.Estimator(pathrank=r, pathsvd=sv))
        d = np.abs(o - base_out)
        print(f"    pathrank {r} {tag:16s} {100*f/B:7.3f}% of B  ({f/base_f:6.4f} x pathrank=None)  "
              f"residual {rw:.3f}s  wall {wl:6.1f}s  final rel diff {d[-1].max()/np.abs(base_out[-1]).max():.2e}")
        recs.append(dict(tag=f"{r} {tag}", pathrank=r, pathsvd=sv, util=f / B, ratio=f / base_f,
                         residual=rw, wall=wl))
(HERE / "logs" / "cost_pathrank32.json").write_text(json.dumps(recs, indent=1))
