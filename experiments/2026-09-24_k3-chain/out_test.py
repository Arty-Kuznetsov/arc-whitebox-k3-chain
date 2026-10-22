import time
import flopscope as flops
import flopscope.numpy as fnp
import inspect
rng = fnp.random.default_rng(0)
n, S = 1024, 12
Wt = rng.standard_normal((n, n), dtype=fnp.float32).T
X = rng.standard_normal((S, n, n), dtype=fnp.float32)
Y = fnp.empty((S, n, n), dtype=fnp.float32)
def t(label, f, reps=3):
    try:
        with flops.BudgetContext(flop_budget=2**55, wall_time_limit_s=600.0, quiet=True) as ctx:
            t0 = time.perf_counter()
            for _ in range(reps):
                r = f()
            dt = (time.perf_counter() - t0) / reps
        print(f"{label:44s} ok  wall {dt*1e3:7.1f} ms resid {ctx.residual_wall_time_s/reps*1e3:6.2f} ms flops {ctx.flops_used/reps:.3e}")
    except Exception as e:
        print(f"{label:44s} FAILED {type(e).__name__}: {str(e)[:150]}")
t("matmul(Wt, X, out=Y)", lambda: fnp.matmul(Wt, X, out=Y))
t("matmul(Wt, X) (alloc)", lambda: fnp.matmul(Wt, X))
t("multiply(X, d[None,:,None], out=Y)", lambda: fnp.multiply(X, X[0, 0][None, :, None], out=Y))
t("sum(X, axis=0, out=Y[0])", lambda: fnp.sum(X, axis=0, out=Y[0]))
t("add(X, X, out=Y)", lambda: fnp.add(X, X, out=Y))
print("configure params:", [p for p in inspect.signature(flops.configure).parameters])
