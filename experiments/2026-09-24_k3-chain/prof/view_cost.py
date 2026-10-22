"""Residual / overhead cost of creating flopscope views (slicing) vs a flopscope op, inside a BudgetContext."""
import flopscope as flops, flopscope.numpy as fnp
X = fnp.empty((7, 16, 256, 256), dtype=fnp.float32)
v = fnp.empty(1024, dtype=fnp.float32)
N = 5000
for rep in range(2):
    for name in ("quadrant view", "leading slice", "vector op", "vector op x3 chain"):
        with flops.BudgetContext(flop_budget=2**50, wall_time_limit_s=3600.0, quiet=True) as ctx:
            if name == "quadrant view":
                for _ in range(N): a = X[:, :, :128, :128]
            elif name == "leading slice":
                for _ in range(N): a = X[2:3]
            elif name == "vector op":
                for _ in range(N): a = v * 1.5
            else:
                for _ in range(N): a = (v * 1.5 + v) * v
        print(f"rep {rep} {name:18s} residual {1e6*ctx.residual_wall_time_s/N:6.2f} us  overhead {1e6*ctx.flopscope_overhead_time_s/N:6.2f} us  backend {1e6*ctx.flopscope_backend_time_s/N:6.2f} us")
