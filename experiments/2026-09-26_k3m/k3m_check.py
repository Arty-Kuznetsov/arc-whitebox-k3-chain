"""Self-check of src/estimator_k3m.py (= estimator_k3l.py with the dead-row threshold default at -2.5).

    uv run --project external/whest-starterkit python experiments/2026-09-26_k3m/k3m_check.py

Two claims, both bitwise, on a fresh 1024x16 He-init MLP:
  (1) k3m(deadcut=-3.0) == k3l()            -> the file is a faithful copy of k3l (nothing else changed);
  (2) k3m()             == k3l(deadcut=-2.5) -> the default is the ONLY change, and it reproduces the
      arithmetic already measured on 100 MLPs by experiments/2026-09-26_cost-frontier/runs/v_k3l_cut25.py
      (adjusted 5.5209e-9, util 24.894%, 0 failures).
FLOPs must match exactly in both pairs; the residual/wall columns are informational (this Mac, under load).
"""
import sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
import estimator_k3l as L, estimator_k3m as M


def mlp(n, depth, seed):
    rng = np.random.default_rng(seed)
    Ws = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(depth)]

    class Mo:
        width = n
    Mo.depth, Mo.seed, Mo.weights = depth, seed, [fnp.asarray(W) for W in Ws]
    return Mo


def run(est, m):
    with flops.BudgetContext(flop_budget=2 ** 50, wall_time_limit_s=3600.0, quiet=True) as ctx:
        t0 = time.perf_counter()
        out = np.asarray(est.predict(m, 2 ** 41), dtype=np.float64)
        return out, ctx.flops_used, ctx.residual_wall_time_s, time.perf_counter() - t0


A = mlp(1024, 16, 11)
for tag, e_ref, e_new in (("(1) k3m(-3.0) vs k3l()", L.Estimator(), M.Estimator(deadcut=-3.0)),
                          ("(2) k3m() vs k3l(-2.5)", L.Estimator(deadcut=-2.5), M.Estimator())):
    o0, f0, r0, w0 = run(e_ref, A)
    o1, f1, r1, w1 = run(e_new, A)
    print(f"{tag}: bitwise {np.array_equal(o0, o1)}, flops equal {f0 == f1} "
          f"({100 * f1 / 2 ** 41:.3f}% of B), residual {r1:.3f}s (ref {r0:.3f}s), wall {w1:.1f}s (ref {w0:.1f}s)")
