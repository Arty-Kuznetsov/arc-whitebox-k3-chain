import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
import estimator_k3 as ek
T = {}
def tick(k, t0):
    T[k] = T.get(k, 0.0) + time.perf_counter() - t0
def _bs(A, B, lv):
    m, k = A.shape[-2], A.shape[-1]; p = B.shape[-1]
    if lv <= 0 or m % 2 or k % 2 or p % 2 or min(m, k, p) < 64:
        t0 = time.perf_counter(); r = fnp.matmul(A, B); tick("leaf matmul", t0); return r
    hm, hk, hp = m // 2, k // 2, p // 2
    t0 = time.perf_counter()
    a11, a12, a21, a22 = A[..., :hm, :hk], A[..., :hm, hk:], A[..., hm:, :hk], A[..., hm:, hk:]
    b11, b12, b21, b22 = B[..., :hk, :hp], B[..., :hk, hp:], B[..., hk:, :hp], B[..., hk:, hp:]
    tick("slice", t0); t0 = time.perf_counter()
    la = [a11 + a22, a21 + a22, fnp.copy(a11), fnp.copy(a22), a11 + a12, a21 - a11, a12 - a22]
    rb = [b11 + b22, fnp.copy(b11), b12 - b22, b21 - b11, fnp.copy(b22), b11 + b12, b21 + b22]
    tick("combos", t0); t0 = time.perf_counter()
    L = fnp.concatenate(la, axis=0); R = fnp.concatenate(rb, axis=0)
    tick("concat LR", t0)
    M = _bs(L, R, lv - 1)
    t0 = time.perf_counter()
    P = A.shape[0]
    m1, m2, m3, m4, m5, m6, m7 = (M[i * P:(i + 1) * P] for i in range(7))
    c11 = m1 + m4 - m5 + m7; c12 = m3 + m5; c21 = m2 + m4; c22 = m1 - m2 + m3 + m6
    tick("recomb adds", t0); t0 = time.perf_counter()
    top = fnp.concatenate([c11, c12], axis=-1); bot = fnp.concatenate([c21, c22], axis=-1)
    r = fnp.concatenate([top, bot], axis=-2)
    tick("recomb concat", t0)
    return r
ek._bs = _bs
n, depth, lv, chunk = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rng = np.random.default_rng(0)
Ws = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(depth)]
class M:
    width, depth, seed = n, depth, 0
    weights = [fnp.asarray(W) for W in Ws]
est = ek.Estimator(strassen_levels=lv, chunk=chunk)
with flops.BudgetContext(flop_budget=2**50, wall_time_limit_s=3600.0, quiet=True) as ctx:
    out = est.predict(M, 2**41)
print(f"{ctx.flops_used/2**41*100:.1f}% of B  wall {ctx.wall_time_s:.2f}  backend {ctx.flopscope_backend_time_s:.2f}  overhead {ctx.flopscope_overhead_time_s:.2f}  RESIDUAL {ctx.residual_wall_time_s:.3f}")
for k, v in sorted(T.items(), key=lambda x: -x[1]):
    print(f"  {k:15s} {v:.3f} s")
