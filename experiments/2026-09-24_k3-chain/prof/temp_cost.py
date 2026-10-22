"""Residual cost of an n x n temporary (allocate + free) vs writing into a preallocated buffer with out=."""
import time
import flopscope as flops, flopscope.numpy as fnp
n = 1024
a = fnp.ones((n, n), dtype=fnp.float32); b = fnp.ones((n, n), dtype=fnp.float32)
v = fnp.ones(n, dtype=fnp.float32)
buf = fnp.empty((n, n), dtype=fnp.float32)
for rep in range(2):
    for name in ("temp_mul", "out_mul", "temp_outer", "out_outer", "temp_chain3"):
        with flops.BudgetContext(flop_budget=2**50, wall_time_limit_s=3600.0, quiet=True) as ctx:
            N = 200
            if name == "temp_mul":
                for _ in range(N): x = a * b
            elif name == "out_mul":
                for _ in range(N): fnp.multiply(a, b, out=buf)
            elif name == "temp_outer":
                for _ in range(N): x = v[:, None] * v[None, :]
            elif name == "out_outer":
                for _ in range(N): fnp.multiply(v[:, None], v[None, :], out=buf)
            elif name == "temp_chain3":
                for _ in range(N): x = a * b + a * b   # 3 temporaries, 2 freed per iteration + x
            x = None
            print(f"rep {rep} {name:12s} residual {1e3*ctx.residual_wall_time_s/N:.3f} ms/op  backend {1e3*ctx.flopscope_backend_time_s/N:.3f}  overhead {1e3*ctx.flopscope_overhead_time_s/N:.3f}")
