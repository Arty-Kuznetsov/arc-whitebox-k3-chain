"""Level-batched Strassen-Winograd: can deeper recursion fit the residual-time cap?

Recursive Strassen issues ~7x more Python-level flopscope calls per level, and the grader caps
non-flopscope time at 400 ms per MLP; that capped the shipped estimator at one level. Here every level
is executed ONCE for the whole stack of sub-problems: operands are stacked along a leading batch axis,
the 7 Strassen combinations are formed with a constant number of elementwise calls on the stacked
arrays, and the leaves are one batched matmul. The call count is then O(levels), not O(7**levels).

Checks: (1) flopscope bills a batched matmul as batch x (m*n*(2k-1)); (2) FLOP ratio vs direct products;
(3) numerical agreement; (4) wall time and flopscope call count per variant.

Run: uv run --project ../../external/whest-starterkit python batched_strassen_test.py
"""
import time

import flopscope as flops
import flopscope.numpy as fnp


def measure(fn):
    with flops.BudgetContext(flop_budget=2**55, wall_time_limit_s=600.0, quiet=True) as ctx:
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
    return out, ctx.flops_used, dt


def strassen_batched(A, B, levels):
    """A, B: (S, m, m) stacks; returns the (S, m, m) stack of products A[s] @ B[s]."""
    if levels == 0:
        return fnp.matmul(A, B)
    S, m, _ = A.shape
    h = m // 2
    a11, a12, a21, a22 = A[:, :h, :h], A[:, :h, h:], A[:, h:, :h], A[:, h:, h:]
    b11, b12, b21, b22 = B[:, :h, :h], B[:, :h, h:], B[:, h:, :h], B[:, h:, h:]
    # 7 operand pairs, stacked along the batch axis: [M1..M7] each of batch S
    Ls = fnp.concatenate([a11 + a22, a21 + a22, a11, a22, a11 + a12, a21 - a11, a12 - a22], axis=0)
    Rs = fnp.concatenate([b11 + b22, b11, b12 - b22, b21 - b11, b22, b11 + b12, b21 + b22], axis=0)
    M = strassen_batched(Ls, Rs, levels - 1)  # (7S, h, h)
    m1, m2, m3, m4, m5, m6, m7 = (M[k * S:(k + 1) * S] for k in range(7))
    top = fnp.concatenate([m1 + m4 - m5 + m7, m3 + m5], axis=2)
    bot = fnp.concatenate([m2 + m4, m1 - m2 + m3 + m6], axis=2)
    return fnp.concatenate([top, bot], axis=1)


def strassen_recursive(a, b, levels):  # the shipped scheme, for comparison
    m = a.shape[0]
    if levels <= 0:
        return a @ b
    h = m // 2
    a11, a12, a21, a22 = a[:h, :h], a[:h, h:], a[h:, :h], a[h:, h:]
    b11, b12, b21, b22 = b[:h, :h], b[:h, h:], b[h:, :h], b[h:, h:]
    r = levels - 1
    m1 = strassen_recursive(a11 + a22, b11 + b22, r)
    m2 = strassen_recursive(a21 + a22, b11, r)
    m3 = strassen_recursive(a11, b12 - b22, r)
    m4 = strassen_recursive(a22, b21 - b11, r)
    m5 = strassen_recursive(a11 + a12, b22, r)
    m6 = strassen_recursive(a21 - a11, b11 + b12, r)
    m7 = strassen_recursive(a12 - a22, b21 + b22, r)
    top = fnp.concatenate([m1 + m4 - m5 + m7, m3 + m5], axis=1)
    bot = fnp.concatenate([m2 + m4, m1 - m2 + m3 + m6], axis=1)
    return fnp.concatenate([top, bot], axis=0)


n, S = 1024, 11  # one layer of the shipped estimator at J = 4 issues 11 products
rng = fnp.random.default_rng(0)
A = rng.standard_normal((S, n, n), dtype=fnp.float32) * (2.0 / n) ** 0.5
B = rng.standard_normal((S, n, n), dtype=fnp.float32)

one = 1 * n * n * (2 * n - 1)
_, f1, _ = measure(lambda: fnp.matmul(A[:1], B[:1]))
ref, fS, tS = measure(lambda: fnp.matmul(A, B))
print(f"billing: batched matmul of {S} = {fS:.4e} FLOPs; {S} x single = {S * f1:.4e}; formula {S * one:.4e}")
print(f"direct, batched:            ratio 1.0000   wall {tS:.3f}s")

_, fr, tr = measure(lambda: [strassen_recursive(A[s], B[s], 1) for s in range(S)])
print(f"recursive Strassen L1 x{S}:  ratio {fr / fS:.4f}   wall {tr:.3f}s")
for lv in (1, 2, 3, 4, 5):
    out, f, t = measure(lambda: strassen_batched(A, B, lv))
    err = float(fnp.max(fnp.abs(out - ref))) / float(fnp.max(fnp.abs(ref)))
    print(f"batched Strassen L{lv}:        ratio {f / fS:.4f}   wall {t:.3f}s   max rel diff {err:.2e}")
