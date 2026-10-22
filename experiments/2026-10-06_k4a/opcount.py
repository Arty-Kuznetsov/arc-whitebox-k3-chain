"""Count flopscope operations (ctx.op_log) per predict: k3z vs k4a defaults, synthetic 1024x16; top op names by count delta."""
import sys, collections
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
import estimator_k3z as Zm, estimator_k4a as A
from gate import mlp
m = mlp(1024, 16, 11)
def ops(est):
    with flops.BudgetContext(flop_budget=2 ** 50, wall_time_limit_s=3600.0, quiet=True) as ctx:
        est.predict(m, 2 ** 41)
        log = ctx.op_log
        names = collections.Counter()
        for e in log:
            nm = getattr(e, "name", None) or (e.get("name") if isinstance(e, dict) else None) or (e[0] if isinstance(e, (tuple, list)) else str(type(e)))
            names[str(nm)] += 1
        return len(log), names
n0, c0 = ops(Zm.Estimator()); print(f"k3z ops: {n0}")
n1, c1 = ops(A.Estimator()); print(f"k4a ops: {n1}  (+{n1 - n0})")
delta = collections.Counter({k: c1[k] - c0.get(k, 0) for k in set(c0) | set(c1)})
for k, v in delta.most_common(14): print(f"  {v:+6d}  {k}  (k3z {c0.get(k,0)} -> k4a {c1.get(k,0)})")
