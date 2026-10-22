"""k3h (sum-of-products Strassen for the D pairings) vs k3f: float-noise agreement at L3, then L4 chunk 16:
FLOPs, residual, wall, peak RSS. Two MLPs on one instance each (workspace reuse)."""
import sys, time, resource
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
import estimator_k3f as F, estimator_k3h as H
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
        t0 = time.perf_counter(); out = np.asarray(est.predict(m, 2**41), dtype=np.float64)
        return out, ctx.flops_used, ctx.residual_wall_time_s, time.perf_counter() - t0
mods = sys.argv[1] if len(sys.argv) > 1 else "all"
Ms = [mlp(41), mlp(42)]
ref = [run(F.Estimator(), m) for m in Ms]
print("k3f L3 c16:", " | ".join(f"{100*r[1]/2**41:.2f}% res {r[2]:.3f} s wall {r[3]:.1f} s" for r in ref))
for kw in (dict(strassen_levels=3, chunk=16), dict(strassen_levels=4, chunk=16)):
    est = H.Estimator(**kw)
    outs = [run(est, m) for m in Ms]
    d = [np.abs(o[0] - r[0]) for o, r in zip(outs, ref)]
    print(f"k3h {kw}: " + " | ".join(f"{100*o[1]/2**41:.2f}% res {o[2]:.3f} s wall {o[3]:.1f} s, final max|diff| {x[-1].max():.1e} "
                                    f"rms {np.sqrt((x[-1]**2).mean()):.1e}" for o, x in zip(outs, d))
          + f" | peak RSS so far {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30:.2f} GB")
