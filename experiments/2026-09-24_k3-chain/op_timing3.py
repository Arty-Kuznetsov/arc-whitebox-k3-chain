import time
import flopscope as flops
import flopscope.numpy as fnp
rng = fnp.random.default_rng(0)
A = rng.standard_normal((1, 12, 1024, 1024), dtype=fnp.float32)
def timeit(label, f, reps=5):
    with flops.BudgetContext(flop_budget=2**55, wall_time_limit_s=600.0, quiet=True) as ctx:
        t0 = time.perf_counter()
        for _ in range(reps):
            r = f()
        dt = (time.perf_counter() - t0) / reps
        fl = ctx.flops_used / reps
    print(f"{label:46s} wall {dt*1e3:7.2f} ms  resid {ctx.residual_wall_time_s/reps*1e3:6.2f} ms  flops {fl:.3e}")
    return r
s = A[..., :512, :512]
for name in ("copy", "ascontiguousarray", "array"):
    f = getattr(fnp, name, None)
    if f is None:
        print(name, "missing"); continue
    try:
        timeit(f"fnp.{name}(slice)", lambda: f(s))
    except Exception as e:
        print(name, "failed", e)
timeit("slice + 0.0", lambda: s + 0.0)
c = fnp.copy(s)
timeit("concatenate 7 (copied)", lambda: fnp.concatenate([c] * 7, axis=0))
