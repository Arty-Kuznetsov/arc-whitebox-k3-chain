"""Peak and current RSS of an estimator over 3 consecutive predict() calls on one instance (the grader keeps one
instance per worker). Usage: mem_probe2.py <module> [k=v ...]"""
import sys, resource, os, subprocess, importlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
mod = importlib.import_module(sys.argv[1])
kw = {k: (v == "True" if v in ("True", "False") else int(v)) for k, v in (a.split("=") for a in sys.argv[2:])}
n, depth = 1024, 16
cur = lambda: int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])) / 2**20
def mlp(seed):
    rng = np.random.default_rng(seed)
    Ws = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(depth)]
    class M:
        width = n
    M.depth, M.seed, M.weights = depth, seed, [fnp.asarray(W) for W in Ws]
    return M
est = mod.Estimator(**kw)
print(f"{sys.argv[1]} {kw}: before predict: current {cur():.2f} GB, peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30:.2f} GB")
for s in range(int(os.environ.get("NPRED", "3"))):
    M = mlp(s)
    with flops.BudgetContext(flop_budget=2**50, wall_time_limit_s=3600.0, quiet=True):
        out = est.predict(M, 2**41)
    del out, M
    print(f"MEM after predict {s}: current {cur():.2f} GB, peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30:.2f} GB")
