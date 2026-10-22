"""Residual / overhead per flopscope op as a function of batch shape at (roughly) fixed total size: does an op on
2401 x 8 matrices of 64x64 cost more untracked time than an op on 7 x 16 matrices of 512x512?"""
import flopscope as flops, flopscope.numpy as fnp
shapes = [(7, 16, 512, 512), (49, 16, 256, 256), (343, 16, 128, 128), (2401, 8, 64, 64), (2401, 16, 64, 64), (16807, 4, 32, 32)]
for rep in range(2):
    for sh in shapes:
        a = fnp.empty(sh, dtype=fnp.float32); b = fnp.empty(sh, dtype=fnp.float32); o = fnp.empty(sh, dtype=fnp.float32)
        N = 20
        with flops.BudgetContext(flop_budget=2**55, wall_time_limit_s=3600.0, quiet=True) as ctx:
            for _ in range(N):
                fnp.add(a, b, out=o)
        mm = None
        with flops.BudgetContext(flop_budget=2**55, wall_time_limit_s=3600.0, quiet=True) as ctx2:
            for _ in range(N):
                fnp.matmul(a, b, out=o)
        tot = 1
        for x in sh: tot *= x
        print(f"rep {rep} {str(sh):22s} {tot/1e6:6.1f}M el | add: residual {1e3*ctx.residual_wall_time_s/N:.3f} ms, overhead {1e3*ctx.flopscope_overhead_time_s/N:.3f}, backend {1e3*ctx.flopscope_backend_time_s/N:.2f} | "
              f"matmul: residual {1e3*ctx2.residual_wall_time_s/N:.3f} ms, overhead {1e3*ctx2.flopscope_overhead_time_s/N:.3f}, backend {1e3*ctx2.flopscope_backend_time_s/N:.2f}")
        del a, b, o
