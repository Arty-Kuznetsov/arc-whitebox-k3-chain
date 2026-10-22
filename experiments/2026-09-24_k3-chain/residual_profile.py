"""Where does the residual (non-flopscope) time of estimator_k3c go? Wraps _Strassen.shared/paired and reads the live
ctx.residual_wall_time_s before/after each call. Usage: residual_profile.py <module> [kwargs as k=v ...]"""
import sys, time, importlib
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
mod = importlib.import_module(sys.argv[1])
kw = {k: (v == "True" if v in ("True", "False") else int(v)) for k, v in (a.split("=") for a in sys.argv[2:])}

n, depth = 1024, 16
rng = np.random.default_rng(0)
Ws = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(depth)]
class M:
    width, depth, seed = n, depth, 0
    weights = [fnp.asarray(W) for W in Ws]

acc = {}
CTX = [None]
def wrap(name):
    f = getattr(mod._Strassen, name)
    def g(self, a, b, out):
        tag = f"{name}(k={b.shape[0]})" if b.shape[0] <= 2 else f"{name}(legs)"
        r0 = CTX[0].residual_wall_time_s
        f(self, a, b, out)
        d = CTX[0].residual_wall_time_s - r0
        c, t = acc.get(tag, (0, 0.0)); acc[tag] = (c + 1, t + d)
    setattr(mod._Strassen, name, g)
if hasattr(mod, "_Strassen"):
    for nm in ("shared", "paired"):
        if hasattr(mod._Strassen, nm):
            wrap(nm)

for rep in range(2):
    acc.clear()
    est = mod.Estimator(**kw)
    with flops.BudgetContext(flop_budget=2**50, wall_time_limit_s=3600.0, quiet=True) as ctx:
        CTX[0] = ctx
        t0 = time.perf_counter()
        out = est.predict(M, 2**41)
        dt = time.perf_counter() - t0
        res = ctx.residual_wall_time_s
    print(f"rep {rep}: {100*ctx.flops_used/2**41:.2f}% of B  wall {dt:.1f}s  residual {res:.3f}s  "
          f"(backend {ctx.flopscope_backend_time_s:.1f}s, overhead {ctx.flopscope_overhead_time_s:.2f}s)")
    for k, (c, t) in sorted(acc.items()):
        print(f"   {k:18s} calls {c:4d}  residual {t:.3f}s  ({1e3*t/max(c,1):.2f} ms/call)")
    print(f"   {'rest':18s}             residual {res - sum(t for _, t in acc.values()):.3f}s")
np.save(HERE / "runs" / f"resprof_{sys.argv[1]}_{'_'.join(sys.argv[2:])}.npy", np.asarray(out))
