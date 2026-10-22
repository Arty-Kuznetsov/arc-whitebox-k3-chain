"""RectStrassen vs fnp.matmul: max relative error, billed FLOPs (vs a plain n^3 product), residual. n=1024, L4, chunk 4."""
import time
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from rect_strassen import RectStrassen
n, L, chunk = 1024, 4, 4
rng = np.random.default_rng(1)
sw = RectStrassen(n, L, chunk)
def ctx():
    return flops.BudgetContext(flop_budget=2**52, wall_time_limit_s=600.0, quiet=True)
for na in (1024, 832, 768):
    W = rng.standard_normal((n, na)).astype(np.float32)
    X = rng.standard_normal((5, na, n)).astype(np.float32)
    Wf, Xf = fnp.asarray(W), fnp.asarray(X)
    out = fnp.empty((5, n, n), dtype=fnp.float32)
    with ctx() as c:
        t0 = time.perf_counter(); sw.shared(Wf, Xf, out); w = time.perf_counter() - t0
    ref = np.einsum('ik,skj->sij', W.astype(np.float64), X.astype(np.float64))
    err = np.abs(np.asarray(out) - ref).max() / np.abs(ref).max()
    print(f"shared     n x {na} @ {na} x n, 5 legs: rel err {err:.1e}, billed {c.flops_used / (5 * 2 * n * n * na):.3f} of plain, "
          f"wall {w*1e3:.0f} ms, residual {c.residual_wall_time_s*1e3:.1f} ms")
    La = rng.standard_normal((6, na, n)).astype(np.float32)
    Rb = rng.standard_normal((6, n, na)).astype(np.float32)
    D = fnp.empty((na, na), dtype=fnp.float32); tmp = fnp.empty((na, na), dtype=fnp.float32)
    with ctx() as c:
        t0 = time.perf_counter(); sw.paired_sum(fnp.asarray(La), fnp.asarray(Rb), D, tmp); w = time.perf_counter() - t0
    ref = np.einsum('sik,skj->ij', La.astype(np.float64), Rb.astype(np.float64))
    err = np.abs(np.asarray(D) - ref).max() / np.abs(ref).max()
    print(f"paired_sum {na} x n @ n x {na}, 6 pairs: rel err {err:.1e}, billed {c.flops_used / (6 * 2 * na * n * na):.3f} of plain, "
          f"wall {w*1e3:.0f} ms, residual {c.residual_wall_time_s*1e3:.1f} ms")
