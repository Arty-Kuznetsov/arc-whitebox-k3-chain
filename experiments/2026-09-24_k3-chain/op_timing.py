import time
import numpy as np
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
    print(f"{label:40s} wall {dt*1e3:8.2f} ms  resid/rep {ctx.residual_wall_time_s/reps*1e3:8.2f} ms  backend/rep {ctx.flopscope_backend_time_s/reps*1e3:8.2f} ms")
    return r
timeit("slice A[..., :512, :512]", lambda: A[..., :512, :512])
s = A[..., :512, :512]
timeit("add of two slices", lambda: A[..., :512, :512] + A[..., 512:, 512:])
timeit("concatenate 7 slices axis0", lambda: fnp.concatenate([A[..., :512, :512]] * 7, axis=0))
x = fnp.concatenate([A[..., :512, :512]] * 7, axis=0)
timeit("matmul (7,12,512,512)^2", lambda: fnp.matmul(x, x), reps=1)
timeit("stack 12 of 1024^2", lambda: fnp.stack([A[0, i] for i in range(12)], axis=0))
timeit("elementwise mul 1024^2", lambda: A[0, 0] * A[0, 1])
timeit("np slice copy (numpy)", lambda: np.ascontiguousarray(np.asarray(A)[..., :512, :512]))
