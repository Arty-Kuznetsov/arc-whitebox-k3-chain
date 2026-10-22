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
        fl = ctx.flops_used / reps
    print(f"{label:46s} wall {dt*1e3:7.2f} ms  resid {ctx.residual_wall_time_s/reps*1e3:6.2f} ms  flops {fl:.3e}")
    return r
c = [A[..., :512, :512] + A[..., 512:, 512:] for _ in range(7)]
timeit("concatenate 7 contiguous", lambda: fnp.concatenate(c, axis=0))
timeit("stack 7 contiguous", lambda: fnp.stack(c, axis=0))
def out_add():
    buf = fnp.empty((7, 12, 512, 512), dtype=fnp.float32)
    for i in range(7):
        fnp.add(A[0, :, :512, :512], A[0, :, 512:, 512:], out=buf[i])
    return buf
try:
    b = timeit("add(out=buf[i]) x7", out_add)
    print("   out= works, check", float(fnp.max(fnp.abs(b[3] - c[3][0]))))
except Exception as e:
    print("out= failed:", type(e).__name__, str(e)[:200])
def reshape_view():
    x = A.reshape(1, 12, 2, 512, 2, 512)
    return x
try:
    timeit("reshape to blocks (view)", reshape_view)
    xb = A.reshape(1, 12, 2, 512, 2, 512)
    timeit("add of block views", lambda: xb[:, :, 0, :, 0, :] + xb[:, :, 1, :, 1, :])
except Exception as e:
    print("reshape failed:", e)
# einsum-based combination with a coefficient tensor
coef = fnp.asarray(np.array([[1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],[1,1,0,0],[-1,0,1,0],[0,1,0,-1]], dtype=np.float32).reshape(7,2,2))
def ein():
    xb = A.reshape(12, 2, 512, 2, 512)
    return fnp.einsum("qxy,sxayb->qsab", coef, xb)
try:
    timeit("einsum combos (7 from 4 blocks)", ein, reps=2)
except Exception as e:
    print("einsum failed:", type(e).__name__, str(e)[:200])
