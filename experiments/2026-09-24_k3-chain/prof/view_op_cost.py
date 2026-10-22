"""Residual per op when the operands are QUADRANT VIEWS (as in _Strassen._rec) and out= is a slice of a stacked buffer."""
import flopscope as flops, flopscope.numpy as fnp
cases = [(1, 16, 1024), (7, 16, 512), (49, 16, 256), (343, 16, 128), (343, 8, 128), (2401, 8, 64)]
for rep in range(2):
    for G, K, m in cases:
        h = m // 2
        A = fnp.empty((G, K, m, m), dtype=fnp.float32)
        LA = fnp.empty((7 * G, K, h, h), dtype=fnp.float32)
        N = 20
        with flops.BudgetContext(flop_budget=2**55, wall_time_limit_s=3600.0, quiet=True) as ctx:
            for _ in range(N):
                a11, a22 = A[:, :, :h, :h], A[:, :, h:, h:]
                fnp.add(a11, a22, out=LA[0:G])
        with flops.BudgetContext(flop_budget=2**55, wall_time_limit_s=3600.0, quiet=True) as ctx2:
            for _ in range(N):
                a11 = A[:, :, :h, :h]
                fnp.multiply(a11, 1.0, out=LA[G:2 * G])
        with flops.BudgetContext(flop_budget=2**55, wall_time_limit_s=3600.0, quiet=True) as ctx3:
            for _ in range(N):
                fnp.add(LA[0:G], LA[G:2 * G], out=A[:, :, :h, :h])
        print(f"rep {rep} G={G:5d} K={K:2d} m={m:4d} | add views->buf: res {1e3*ctx.residual_wall_time_s/N:.3f} ms ovh {1e3*ctx.flopscope_overhead_time_s/N:.3f} be {1e3*ctx.flopscope_backend_time_s/N:.2f}"
              f" | copy view: res {1e3*ctx2.residual_wall_time_s/N:.3f} | add buf->view(out quadrant): res {1e3*ctx3.residual_wall_time_s/N:.3f} ovh {1e3*ctx3.flopscope_overhead_time_s/N:.3f} be {1e3*ctx3.flopscope_backend_time_s/N:.2f}")
        del A, LA
