"""Report 1 — fnp.einsum is 20-70x slower in wall time than the equivalent matmul + sum at an identical bill.
Why it matters: the grader caps wall time at 120 s per network (rules 5.5), so a slow op at equal FLOPs is a hidden cost.
Run: uv run --project external/whest-starterkit python experiments/2026-10-06_flopscope-reports/einsum_wall.py"""
import time
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

print("flopscope", flops.__version__, "| numpy", np.__version__)
rng = np.random.default_rng(0)
rows = []
with flops.BudgetContext(flop_budget=2**50, quiet=True) as ctx:
    def billed(fn):
        b = int(ctx.flops_used); t0 = time.perf_counter(); out = fn(); dt = time.perf_counter() - t0
        return int(ctx.flops_used) - b, dt, out
    for (G, K, m) in [(16, 16, 32), (64, 16, 64), (128, 8, 64), (256, 16, 32), (2401, 1, 64)]:
        P = fnp.asarray(rng.standard_normal((G, K, m, m)).astype(np.float32))
        Q = fnp.asarray(rng.standard_normal((G, K, m, m)).astype(np.float32))
        for _ in range(2):  # warm, then measure
            c1, t1, O1 = billed(lambda: fnp.einsum("gkij,gkjl->gil", P, Q))
            c2, t2, O2 = billed(lambda: fnp.sum(fnp.matmul(P, Q), axis=1))
        diff = float(np.max(np.abs(np.asarray(O1) - np.asarray(O2))))
        rows.append((G, K, m, c1, c2, t1, t2, t1 / t2, diff))
print(f"{'G':>5} {'K':>3} {'m':>3} | {'einsum bill':>13} {'matmul+sum bill':>16} | {'einsum s':>9} {'matmul+sum s':>13} {'ratio':>6} | max|diff|")
for G, K, m, c1, c2, t1, t2, r, diff in rows:
    print(f"{G:>5} {K:>3} {m:>3} | {c1:>13,} {c2:>16,} | {t1:>9.4f} {t2:>13.4f} {r:>6.1f} | {diff:.1e}")
