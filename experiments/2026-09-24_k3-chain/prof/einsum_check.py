"""Does flopscope bill einsum('gkij,gkjl->gil') like the equivalent batched matmul + sum, and does it accept out=?"""
import numpy as np
import flopscope as flops, flopscope.numpy as fnp
G, K, m = 7, 16, 64
rng = np.random.default_rng(0)
A = fnp.asarray(rng.standard_normal((G, K, m, m)).astype(np.float32)); B = fnp.asarray(rng.standard_normal((G, K, m, m)).astype(np.float32))
with flops.BudgetContext(flop_budget=2**50, quiet=True) as c1:
    P = fnp.matmul(A, B); S1 = fnp.sum(P, axis=1)
with flops.BudgetContext(flop_budget=2**50, quiet=True) as c2:
    S2 = fnp.einsum('gkij,gkjl->gil', A, B)
out = fnp.empty((G, m, m), dtype=fnp.float32)
ok_out = True
try:
    with flops.BudgetContext(flop_budget=2**50, quiet=True) as c3:
        fnp.einsum('gkij,gkjl->gil', A, B, out=out)
except Exception as e:
    ok_out = f"no: {type(e).__name__}: {e}"
print("matmul+sum flops", c1.flops_used, " einsum flops", c2.flops_used, " ratio", c2.flops_used / c1.flops_used)
print("max |diff|", float(np.abs(np.asarray(S1) - np.asarray(S2)).max()), " einsum out= supported:", ok_out,
      (float(np.abs(np.asarray(out) - np.asarray(S2)).max()) if ok_out is True else ""))
print("theory 2GKm^3 =", 2 * G * K * m ** 3)
