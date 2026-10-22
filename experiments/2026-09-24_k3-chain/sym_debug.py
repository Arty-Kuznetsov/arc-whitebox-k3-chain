import flopscope as flops
import flopscope.numpy as fnp
rng = fnp.random.default_rng(0)
n = 64
w = rng.standard_normal((n, n), dtype=fnp.float32)
C = fnp.eye(n, dtype=fnp.float32)
with flops.BudgetContext(flop_budget=2**50, wall_time_limit_s=60.0, quiet=True):
    C_pre = w.T @ (C @ w)
    print("C_pre type", type(C_pre).__name__, getattr(C_pre, "symmetry", None))
    C0 = C_pre - fnp.diag(fnp.diag(C_pre))
    print("C0 type", type(C0).__name__, getattr(C0, "symmetry", None))
    d1 = rng.standard_normal(n, dtype=fnp.float32)
    X = fnp.zeros((3, n, n), dtype=fnp.float32)
    r = d1[:, None] * C0
    print("r type", type(r).__name__, getattr(r, "symmetry", None))
    try:
        fnp.multiply(d1[:, None], C0, out=X[0]); print("out ok")
    except Exception as e:
        print("out failed:", e)
    X[1] = r
    print("setitem ok; X type", type(X).__name__)
