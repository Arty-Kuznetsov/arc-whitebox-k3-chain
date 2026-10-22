"""Gates of src/estimator_k4a.py (experiments/2026-10-06_k4a/PLAN.md):
  G0: restructure=False, young=None -> bitwise k3z, equal FLOPs (1024x16 synthetic; 128x8 small config)
  G1: restructure=True, young=None  -> outputs equal k3z to float rounding; FLOPs ratio; wall
  G2: defaults (young on)           -> FLOPs ratio, finite outputs; the K4d move is measured against the torch chain separately
Run: uv run --project ../../external/whest-starterkit python gate.py [g0|g1|g2|all]   (from this folder)"""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
import estimator_k3z as Zm, estimator_k4a as A

def mlp(n, depth, seed):
    rng = np.random.default_rng(seed)
    Ws = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(depth)]
    class Mo:
        width = n
    Mo.depth, Mo.seed, Mo.weights = depth, seed, [fnp.asarray(Wm) for Wm in Ws]
    return Mo

def run(est, m):
    t0 = time.perf_counter()
    with flops.BudgetContext(flop_budget=2 ** 50, wall_time_limit_s=3600.0, quiet=True) as ctx:
        out = np.asarray(est.predict(m, 2 ** 41), dtype=np.float64)
        return out, ctx.flops_used, time.perf_counter() - t0

what = sys.argv[1] if len(sys.argv) > 1 else "all"
if __name__ != "__main__":
    what = "none"
if __name__ == "__main__":
    small = dict(thin=128, band=None, transport_l5=False)
    m = mlp(1024, 16, 11)
    o0, f0, w0 = run(Zm.Estimator(), m)
    print(f"k3z 1024x16: FLOPs {f0:.6e}  wall {w0:.1f} s", flush=True)
    if what in ("g0", "all"):
        o1, f1, w1 = run(A.Estimator(restructure=False, young=None), m)
        print(f"G0 1024x16: bitwise k3z {np.array_equal(o0, o1)}  flops equal {f0 == f1}  wall {w1:.1f} s", flush=True)
        ms = mlp(128, 8, 5)
        a0, g0_, _ = run(Zm.Estimator(**small), ms)
        a1, g1_, _ = run(A.Estimator(restructure=False, young=None, **small), ms)
        print(f"G0 128x8: bitwise {np.array_equal(a0, a1)}  flops equal {g0_ == g1_}", flush=True)
    if what in ("g1", "all"):
        o2, f2, w2 = run(A.Estimator(young=None), m)
        rel = np.abs(o2 - o0).max() / np.abs(o0).max()
        print(f"G1 1024x16: restructure only: FLOPs x{f2 / f0:.5f} (+{100 * (f2 / f0 - 1):.2f}%)  max rel move {rel:.2e}  "
              f"finite {np.isfinite(o2).all()}  wall {w2:.1f} s (k3z {w0:.1f})", flush=True)
    if what in ("g2", "all"):
        o3, f3, w3 = run(A.Estimator(), m)
        rel = np.abs(o3 - o0).max() / np.abs(o0).max()
        last = np.abs(o3[-1] - o0[-1]).mean() / np.abs(o0[-1]).mean()
        print(f"G2 1024x16: defaults (young on): FLOPs x{f3 / f0:.5f} (+{100 * (f3 / f0 - 1):.2f}%)  max rel move {rel:.2e}  "
              f"mean |d last row|/|last| {last:.2e}  finite {np.isfinite(o3).all()}  wall {w3:.1f} s", flush=True)
