"""Billing and time of the operations a dead-row cut needs (n=1024, n'=832 ~ 81% alive):
square vs rectangular matmul, a stack of legs contracted over alive rows, and the row gathers (take / fancy index)."""
import time
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
n, na, S = 1024, 832, 8
rng = np.random.default_rng(0)
W = fnp.asarray(rng.standard_normal((n, n)).astype(np.float32))
X = fnp.asarray(rng.standard_normal((S, n, n)).astype(np.float32))
idx = fnp.asarray(np.sort(rng.choice(n, na, replace=False)).astype(np.int64))
def meas(name, f):
    with flops.BudgetContext(flop_budget=2**50, wall_time_limit_s=600.0, quiet=True) as ctx:
        t0 = time.perf_counter(); f(); w = time.perf_counter() - t0
        print(f"{name:40s} flops {ctx.flops_used:.4e}  ({ctx.flops_used / (2 * n**3):.3f} x n^3-product)  wall {w*1e3:7.1f} ms  residual {ctx.residual_wall_time_s*1e3:6.2f} ms")
meas("square  W @ X[0]", lambda: W @ X[0])
Wa = W[:, :na]; Xa = X[0, :na, :]
meas("rect    W[:, :na] @ X[0, :na, :] (views)", lambda: Wa @ Xa)
meas("take    W cols (n x na)", lambda: fnp.take(W, idx, axis=1))
meas("take    X rows (S x na x n)", lambda: fnp.take(X, idx, axis=1))
meas("stack   W[:, :na] @ X[:, :na, :]", lambda: fnp.matmul(Wa[None], X[:, :na, :]))
meas("stack   square W @ X", lambda: fnp.matmul(W[None], X))
