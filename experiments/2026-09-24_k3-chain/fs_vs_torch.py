"""Layer-by-layer comparison of the flopscope estimator (src/estimator_k3.py) with the torch chain (chain.py)."""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
from estimator_k3 import Estimator

n = int(sys.argv[1]) if len(sys.argv) > 1 else 128
depth = int(sys.argv[2]) if len(sys.argv) > 2 else 6
lv = int(sys.argv[3]) if len(sys.argv) > 3 else 1
rng = np.random.default_rng(0)
Ws = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(depth)]
np.save(HERE / "runs" / "fs_test_W.npy", np.stack(Ws))

class M:  # minimal MLP stand-in
    width, depth, seed = n, depth, 0
    weights = [fnp.asarray(W) for W in Ws]

est = Estimator(strassen_levels=lv)
with flops.BudgetContext(flop_budget=2**50, wall_time_limit_s=3600.0, quiet=True) as ctx:
    t0 = time.perf_counter()
    out = est.predict(M, 2**41)
    dt = time.perf_counter() - t0
np.save(HERE / "runs" / "fs_test_out.npy", np.asarray(out))
print(f"flopscope done: {ctx.flops_used:.4e} FLOPs, {dt:.2f}s wall")
