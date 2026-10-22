"""k3f residual trims: (1) fuse_base=False must be BITWISE equal to the pre-trim k3f (scratch copy estimator_k3f_v0.py),
since trims (a) t/6 once, (b) unused Wick coefficients skipped do not change any value; (2) fuse_base=True (Wt C rides
the leg transport) vs fuse_base=False: max |diff| (float noise at most), FLOPs, residual. MLPs A, B, A on one instance."""
import sys, time, importlib.util
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
import estimator_k3f as NEW
spec = importlib.util.spec_from_file_location("k3f_v0", sys.argv[1]); OLD = importlib.util.module_from_spec(spec); spec.loader.exec_module(OLD)
n, depth = 1024, 16
def mlp(seed):
    rng = np.random.default_rng(seed)
    Ws = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(depth)]
    class M:
        width = n
    M.depth, M.seed, M.weights = depth, seed, [fnp.asarray(W) for W in Ws]
    return M
def run(est, m):
    with flops.BudgetContext(flop_budget=2**50, wall_time_limit_s=3600.0, quiet=True) as ctx:
        out = np.asarray(est.predict(m, 2**41), dtype=np.float64)
        return out, ctx.flops_used, ctx.residual_wall_time_s
A, B = mlp(21), mlp(22)
old, nof, fus = OLD.Estimator(), NEW.Estimator(fuse_base=False), NEW.Estimator()
for nm, m in (("A", A), ("B", B), ("A", A)):
    o0, f0, r0 = run(old, m); o1, f1, r1 = run(nof, m); o2, f2, r2 = run(fus, m)
    print(f"{nm}: fuse_base=False vs pre-trim: bitwise {np.array_equal(o0, o1)}, flops {f1 - f0:+d}; residual {r0:.3f} -> {r1:.3f} s | "
          f"fused vs unfused: max|diff| {np.abs(o2 - o1).max():.2e}, flops {100*(f2 - f1)/2**41:+.3f}% of B, residual {r2:.3f} s")
