"""Flopscope calls per code region of one k3p predict: which parts of the estimator issue the calls (residual time is
~20 us per call). python opcount.py [module] [band|none]"""
import sys, time
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
mod = __import__(f"estimator_{sys.argv[1] if len(sys.argv) > 1 else 'k3p'}")
counts = Counter()
stack = []
def wrap(obj, name, label):
    f = getattr(obj, name)
    def g(*a, **k):
        stack.append(label)
        n0 = len(ctx.op_log)
        try:
            return f(*a, **k)
        finally:
            stack.pop()
            if not stack:
                counts[label] += len(ctx.op_log) - n0
    setattr(obj, name, g)
S = mod._Strassen
for nm in ("shared", "rshared", "paired", "paired_sum"):
    wrap(S, nm, "strassen." + nm)
mod._nonlin = (lambda f: (lambda *a, **k: (stack.append("nonlin"), (n0 := len(ctx.op_log)), (r := f(*a, **k)), stack.pop(), counts.__setitem__("nonlin", counts["nonlin"] + len(ctx.op_log) - n0), r)[-1]))(mod._nonlin)
rng = np.random.default_rng(11)
Ws = [(rng.standard_normal((1024, 1024)) * np.sqrt(2.0 / 1024)).astype(np.float32) for _ in range(16)]
class Mo:
    width = 1024
Mo.depth, Mo.seed, Mo.weights = 16, 11, [fnp.asarray(W) for W in Ws]
kw = {} if len(sys.argv) < 3 or sys.argv[2] == "band" else {"band": None}
est = mod.Estimator(**kw)
with flops.BudgetContext(flop_budget=2 ** 50, wall_time_limit_s=3600.0, quiet=True) as ctx:
    est.predict(Mo, 2 ** 41)
    tot = len(ctx.op_log)
    print(f"total calls {tot}, residual {ctx.residual_wall_time_s:.3f} s")
    for k, v in counts.most_common():
        print(f"  {k:<22} {v:>6}  ({100 * v / tot:.1f}%)")
    print(f"  {'rest':<22} {tot - sum(counts.values()):>6}  ({100 * (tot - sum(counts.values())) / tot:.1f}%)")
