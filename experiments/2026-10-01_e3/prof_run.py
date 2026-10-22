"""Where does the budget actually go? Per-call-site FLOPs via flopscope.namespace() on the shipping estimator.
Nobody in this project has measured this: the cost axis was always argued from a model, not from the meter."""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
import estimator_prof as P

rng = np.random.default_rng(11)
n, depth = 1024, 16
Ws = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(depth)]


class Mo:
    width = n


Mo.depth, Mo.seed, Mo.weights = depth, 11, [fnp.asarray(W) for W in Ws]

with flops.BudgetContext(flop_budget=2 ** 50, wall_time_limit_s=3600.0, quiet=True) as ctx:
    t0 = time.perf_counter()
    _ = P.Estimator(recv=None).predict(Mo, 2 ** 41)
    wall = time.perf_counter() - t0
    summary = flops.budget_summary_dict()
tot = ctx.flops_used
print(f"total {tot:.5e}  util {tot/2.2e12*100:.3f}%  wall {wall:.1f}s  residual {ctx.residual_wall_time_s:.3f}s\n")


def walk(d, prefix=""):
    rows = []
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            f = v.get("flops_used")
            if f:
                rows.append((prefix + k, f))
            rows += walk(v.get("children") or v.get("by_namespace") or {}, prefix + k + ".")
    return rows


import json
txt = json.dumps(summary)
Path(HERE / "prof_summary.json").write_text(txt)
rows = walk(summary.get("by_namespace") or summary)
if not rows:
    print("structure:", list(summary.keys()))
    print(txt[:1500])
else:
    for name, f in sorted(rows, key=lambda t: -t[1]):
        print(f"  {name:48s} {f:14.5e}  {100*f/tot:6.2f}%")
