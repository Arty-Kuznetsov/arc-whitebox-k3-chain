"""Peak RSS of ONE predict of an estimator in a fresh process (the grader's cap is 8 GB per process).
    python memprobe.py k3m|k3o|k3p|k3q [kwargs as key=value]"""
import resource, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
mod = __import__(f"estimator_{sys.argv[1]}")
kw = {}
for a in sys.argv[2:]:
    k, v = a.split("=")
    kw[k] = eval(v)
rng = np.random.default_rng(11)
Ws = [(rng.standard_normal((1024, 1024)) * np.sqrt(2.0 / 1024)).astype(np.float32) for _ in range(16)]
class Mo:
    width = 1024
Mo.depth, Mo.seed, Mo.weights = 16, 11, [fnp.asarray(W) for W in Ws]
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30
est = mod.Estimator(**kw)
for i in range(2):  # the second predict reuses the cached workspace, as on the grader's workers
    with flops.BudgetContext(flop_budget=2 ** 50, wall_time_limit_s=3600.0, quiet=True) as ctx:
        t0 = time.perf_counter()
        est.predict(Mo, 2 ** 41)
        print(f"{sys.argv[1]} {kw} predict {i}: peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30:.2f} GB "
              f"(before the estimator {base:.2f} GB), residual {ctx.residual_wall_time_s:.3f} s, wall {time.perf_counter() - t0:.1f} s")
