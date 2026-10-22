"""Does flopscope bill a broadcast batched matmul (1,h,h)@(S,h,h) as S*h*h*(2h-1)?"""
import flopscope as flops
import flopscope.numpy as fnp
rng = fnp.random.default_rng(0)
h, S = 256, 7
A = rng.standard_normal((1, h, h), dtype=fnp.float32)
B = rng.standard_normal((S, h, h), dtype=fnp.float32)
with flops.BudgetContext(flop_budget=2**50, wall_time_limit_s=600.0, quiet=True) as ctx:
    C = fnp.matmul(A, B)
print("broadcast billed", ctx.flops_used, "expected", S * h * h * (2 * h - 1), "shape", C.shape)
with flops.BudgetContext(flop_budget=2**50, wall_time_limit_s=600.0, quiet=True) as ctx:
    C2 = fnp.matmul(fnp.broadcast_to(A, (S, h, h)), B)
print("explicit broadcast_to billed", ctx.flops_used)
with flops.BudgetContext(flop_budget=2**50, wall_time_limit_s=600.0, quiet=True) as ctx:
    s = fnp.sum(B, axis=0)
print("sum over batch billed", ctx.flops_used, "vs", (S - 1) * h * h)
with flops.BudgetContext(flop_budget=2**50, wall_time_limit_s=600.0, quiet=True) as ctx:
    T = fnp.swapaxes(B, 1, 2)
print("swapaxes billed", ctx.flops_used)
with flops.BudgetContext(flop_budget=2**50, wall_time_limit_s=600.0, quiet=True) as ctx:
    Q, R = fnp.linalg.qr(B[0])
print("qr billed", ctx.flops_used)
