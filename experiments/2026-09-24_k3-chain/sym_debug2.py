import flopscope as flops
import flopscope.numpy as fnp
rng = fnp.random.default_rng(0)
n = 64
w = rng.standard_normal((n, n), dtype=fnp.float32)
with flops.BudgetContext(flop_budget=2**50, wall_time_limit_s=60.0, quiet=True) as ctx:
    C0 = w.T @ w
    d1 = rng.standard_normal(n, dtype=fnp.float32)
    for mk in ("zeros", "empty"):
        X = getattr(fnp, mk)((3, n, n), dtype=fnp.float32)
        print(mk, "symmetry of buffer:", getattr(X, "symmetry", None), "| slice:", getattr(X[0], "symmetry", None))
        for label, f in [("multiply row-scale into slice", lambda: fnp.multiply(d1[:, None], C0, out=X[0])),
                         ("copy Wt into slice via *1.0", lambda: fnp.multiply(w.T, 1.0, out=X[1])),
                         ("strided slice out", lambda: fnp.multiply(X[0:3:2], 2.0, out=X[0:3:2]))]:
            try:
                f(); print("   ", label, "ok")
            except Exception as e:
                print("   ", label, "FAILED:", str(e)[:120])
        G = getattr(fnp, mk)((4, n), dtype=fnp.float32)
        try:
            fnp.multiply(d1, 1.0, out=G[2]); print("    vector into row ok", float(fnp.max(fnp.abs(G[2] - d1))))
        except Exception as e:
            print("    vector into row FAILED", str(e)[:120])
    print("flops", ctx.flops_used)
