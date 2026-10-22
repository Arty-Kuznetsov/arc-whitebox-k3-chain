"""Rule 3: measure what the 39 features cost on the meter AND in residual time, before building the corrector."""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
from k4feats import k4feats, NF

n = 1024
rng = np.random.default_rng(0)
A = lambda *s: fnp.asarray(rng.standard_normal(s).astype(np.float32))
mu, var, t, K4d = A(n), fnp.asarray(np.abs(rng.standard_normal(n)).astype(np.float32) + 0.1), A(n), A(n)
C0, D, K22, wl, wn = A(n, n), A(n, n), A(n, n), A(n, n), A(n, n)
out = fnp.empty((n, NF), dtype=fnp.float32)
with flops.BudgetContext(flop_budget=2 ** 50, wall_time_limit_s=600.0, quiet=True) as ctx:
    t0 = time.perf_counter()
    for l in range(8, 16):   # the 8 layers the corrector touches
        k4feats(mu, var, t, K4d, C0, D, K22, wl, wn, l, n, out)
        coef = fnp.asarray(rng.standard_normal(NF).astype(np.float32))
        pe = out @ coef
    wall = time.perf_counter() - t0
print(f"8 layers of features + the dot product: FLOPs {ctx.flops_used:.3e} ({ctx.flops_used/2.2e12*100:.3f}% of the budget)"
      f"   residual {ctx.residual_wall_time_s*1000:.1f} ms   wall {wall*1000:.0f} ms")
