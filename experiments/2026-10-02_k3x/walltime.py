"""Paired wall timing, k3w vs k3x (and k3y), alternating on the same synthetic 1024x16 MLP, to get the dpair
increment without the machine's drift (the 100-MLP local run's +6 s was measured under heavy contention)."""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
import estimator_k3w as W, estimator_k3x as X, estimator_k3y as Y
from gate import mlp


def timed(est, m):
    with flops.BudgetContext(flop_budget=2 ** 50, wall_time_limit_s=3600.0, quiet=True) as ctx:
        t0 = time.perf_counter()
        est.predict(m, 2 ** 41)
        return time.perf_counter() - t0


m = mlp(1024, 16, 11)
ests = {"k3w": W.Estimator(), "k3x": X.Estimator(), "k3y": Y.Estimator()}
for e in ests.values():
    timed(e, m)            # warm: workspace + caches
res = {k: [] for k in ests}
for rep in range(3):
    for k, e in ests.items():
        res[k].append(timed(e, m))
    print(rep, {k: round(v[-1], 1) for k, v in res.items()}, flush=True)
med = {k: float(np.median(v)) for k, v in res.items()}
print("median:", {k: round(v, 2) for k, v in med.items()}, " k3x-k3w %+.2f s, k3y-k3x %+.2f s" % (med["k3x"] - med["k3w"], med["k3y"] - med["k3x"]))
