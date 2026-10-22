"""Compare the predictions of estimator_k3c under three matmul paths on one random width-1024 MLP:
L0 (no Strassen), L3nb (Strassen on the legs only), L3 (legs + base products). Sanity check that the
Strassen paths agree with plain matmul to float32 rounding."""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
from estimator_k3c import Estimator

n, depth = 1024, 16
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
Ws = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(depth)]

class M:
    width, depth, seed = n, depth, 0
    weights = [fnp.asarray(W) for W in Ws]

outs = {}
for name, kw in [("L0", dict(strassen_levels=0)), ("L3nb", dict(strassen_levels=3, base_strassen=False)),
                 ("L3", dict(strassen_levels=3))]:
    est = Estimator(**kw)
    with flops.BudgetContext(flop_budget=2**50, wall_time_limit_s=3600.0, quiet=True) as ctx:
        t0 = time.perf_counter()
        outs[name] = np.asarray(est.predict(M, 2**41), dtype=np.float64)
        dt = time.perf_counter() - t0
    print(f"{name:5s} {100*ctx.flops_used/2**41:6.2f}% of B  wall {dt:5.1f}s")
for a, b in [("L0", "L3nb"), ("L0", "L3"), ("L3nb", "L3")]:
    d = np.abs(outs[a] - outs[b])
    print(f"{a:4s} vs {b:4s}: max|diff| per layer", " ".join(f"{x:.1e}" for x in d.max(axis=1)),
          f" final-layer rms {np.sqrt((d[-1]**2).mean()):.2e}")
print("final-layer mean activation level", outs["L0"][-1].mean())
