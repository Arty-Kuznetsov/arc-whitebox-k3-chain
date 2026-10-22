import sys, resource
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
from estimator_k3b import Estimator
lv, ch = int(sys.argv[1]), int(sys.argv[2])
n, depth = 1024, 16
rng = np.random.default_rng(0)
Ws = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(depth)]
class M:
    width, depth, seed = n, depth, 0
    weights = [fnp.asarray(W) for W in Ws]
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
with flops.BudgetContext(flop_budget=2**50, wall_time_limit_s=3600.0, quiet=True):
    Estimator(strassen_levels=lv, chunk=ch).predict(M, 2**41)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(f"L{lv} chunk {ch}: peak RSS {peak/2**30:.2f} GB (before predict {base/2**30:.2f} GB)")
