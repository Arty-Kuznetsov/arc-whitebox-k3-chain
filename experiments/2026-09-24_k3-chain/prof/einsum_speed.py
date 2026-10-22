import time, inspect
import numpy as np
import flopscope as flops, flopscope.numpy as fnp
print("einsum signature:", inspect.signature(fnp.einsum) if hasattr(fnp.einsum, '__call__') else None)
G, K, m = 343, 16, 128
A = fnp.asarray(np.random.default_rng(0).standard_normal((G, K, m, m)).astype(np.float32))
B = fnp.asarray(np.random.default_rng(1).standard_normal((G, K, m, m)).astype(np.float32))
out = fnp.empty((G, m, m), dtype=fnp.float32); tmp = fnp.empty((G, K, m, m), dtype=fnp.float32)
for name, f in [("einsum", lambda: fnp.einsum('gkij,gkjl->gil', A, B, out=out)),
                ("einsum optimize", lambda: fnp.einsum('gkij,gkjl->gil', A, B, out=out, optimize=True)),
                ("matmul+sum", lambda: (fnp.matmul(A, B, out=tmp), fnp.sum(tmp, axis=1, out=out))),
                ("reshape matmul", lambda: fnp.matmul(fnp.reshape(fnp.transpose(A, (0, 2, 1, 3)), (G, m, K * m)), fnp.reshape(B, (G, K * m, m)), out=out))]:
    try:
        with flops.BudgetContext(flop_budget=2**55, quiet=True) as c:
            t0 = time.perf_counter(); f(); dt = time.perf_counter() - t0
        print(f"{name:16s} {dt*1e3:8.1f} ms  flops {c.flops_used:.4e}  backend {c.flopscope_backend_time_s*1e3:.1f} ms  residual {c.residual_wall_time_s*1e3:.2f} ms")
    except Exception as e:
        print(f"{name:16s} failed: {type(e).__name__}: {str(e)[:120]}")
