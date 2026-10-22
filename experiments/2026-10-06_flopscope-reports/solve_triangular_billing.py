"""Report 2 — fnp.linalg.solve bills an LU factorisation plus two triangular solves even when the matrix is triangular,
and there is no solve_triangular to declare the structure. A triangular system costs one triangular solve (n^2 per
right-hand side), i.e. about 2.7x less than what is billed; the only way to be billed for the work actually done is an
explicit inverse applied by matmul (which also loses accuracy). Request: a `fnp.linalg.solve_triangular(a, b, lower=…)`
op billed at n*n*nrhs (+ the usual add/mul split), or structure detection in `solve`.
Run: uv run --project external/whest-starterkit python experiments/2026-10-06_flopscope-reports/solve_triangular_billing.py"""
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

print("flopscope", flops.__version__, "| numpy", np.__version__)
rng = np.random.default_rng(0)
with flops.BudgetContext(flop_budget=2**50, quiet=True) as ctx:
    def billed(fn):
        b = int(ctx.flops_used); out = fn(); return int(ctx.flops_used) - b, out
    for n, k in [(256, 256), (448, 448), (448, 1024)]:
        R = np.triu(rng.standard_normal((n, n)).astype(np.float32)) + np.eye(n, dtype=np.float32) * n
        B = rng.standard_normal((n, k)).astype(np.float32)
        Rf, Bf = fnp.asarray(R), fnp.asarray(B)
        c_solve, X1 = billed(lambda: fnp.linalg.solve(Rf, Bf))
        c_inv, Ri = billed(lambda: fnp.linalg.inv(Rf))
        c_app, X2 = billed(lambda: Ri @ Bf)
        back_subst = n * n * k           # multiply-adds of one triangular solve with k right-hand sides (≈ 2·n²k/2 flops)
        lu = 2 * n**3 // 3               # LU of a general n×n matrix
        print(f"n={n} nrhs={k}: solve(R,B) billed {c_solve:,} = {c_solve/back_subst:.2f}× a triangular solve "
              f"(LU {lu:,} + fwd/back {2*back_subst:,} = {lu + 2*back_subst:,}); inv(R) {c_inv:,}; inv@B {c_app:,}; "
              f"max rel diff solve vs inv@B {float(np.max(np.abs(np.asarray(X1)-np.asarray(X2)))/np.max(np.abs(np.asarray(X1)))):.1e}")
    print("\nReference: numpy's triangular solve of an n×n system with k right-hand sides is n(n+1)k/2 multiply-adds; "
          "flopscope has no op that bills it.")
