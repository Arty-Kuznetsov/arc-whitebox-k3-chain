"""Does flopscope bill a Strassen-Winograd matmul cheaper than a direct one?

Organisers confirmed on the forum (11 Sep) that Strassen-Winograd expressed as flopscope ops is
permitted with no recursion-level limit. Our estimator is now ~90% matmul, so if the meter really
charges 7 half-size products instead of 8, the whole budget roughly doubles. Checked here directly,
including numerical agreement so we know it is the same arithmetic.
"""
import flopscope as flops
import flopscope.numpy as fnp

n = 512


def strassen(A, Bm, levels):
    m = A.shape[0]
    if levels == 0 or m % 2 or m < 64:
        return A @ Bm
    h = m // 2
    a11, a12, a21, a22 = A[:h, :h], A[:h, h:], A[h:, :h], A[h:, h:]
    b11, b12, b21, b22 = Bm[:h, :h], Bm[:h, h:], Bm[h:, :h], Bm[h:, h:]
    r = levels - 1
    m1 = strassen(a11 + a22, b11 + b22, r)
    m2 = strassen(a21 + a22, b11, r)
    m3 = strassen(a11, b12 - b22, r)
    m4 = strassen(a22, b21 - b11, r)
    m5 = strassen(a11 + a12, b22, r)
    m6 = strassen(a21 - a11, b11 + b12, r)
    m7 = strassen(a12 - a22, b21 + b22, r)
    top = fnp.concatenate([m1 + m4 - m5 + m7, m3 + m5], axis=1)
    bot = fnp.concatenate([m2 + m4, m1 - m2 + m3 + m6], axis=1)
    return fnp.concatenate([top, bot], axis=0)


rng = fnp.random.default_rng(0)
A = rng.standard_normal((n, n), dtype=fnp.float32)
Bm = rng.standard_normal((n, n), dtype=fnp.float32)

def measure(fn):
    with flops.BudgetContext(flop_budget=2**50, wall_time_limit_s=600.0, quiet=True) as ctx:
        out = fn()
    return out, ctx.flops_used


ref, direct = measure(lambda: A @ Bm)
print(f"direct matmul {n}x{n}:            {direct:,} FLOPs")

for lv in (1, 2, 3, 4):
    out, t = measure(lambda: strassen(A, Bm, lv))
    err = float(fnp.max(fnp.abs(out - ref))) / float(fnp.max(fnp.abs(ref)))
    print(f"strassen, {lv} level(s):           {t:,} FLOPs   ratio {t/direct:.4f}   "
          f"max rel diff {err:.2e}")
